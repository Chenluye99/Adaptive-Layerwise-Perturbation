# patch_qwen2.py  (HF transformers==4.54.0 aligned)
import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


class CustomQwen2DecoderLayer(GradientCheckpointingLayer):
    """
    HF 4.54.0 aligned:
    - Inject perturbation BEFORE input_layernorm.
    - Inherits GradientCheckpointingLayer so model.gradient_checkpointing_enable() works (hook in __call__).
    - micro-checkpoint only around injection when coef_learnable=True to avoid storing full-size noise.
    - Keeps Qwen2DecoderLayer semantics: returns hidden_states (Tensor).
    - Uses past_key_value (singular) to match HF 4.54.0 Qwen2Attention/Qwen2DecoderLayer.
    """

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = int(layer_idx)

        # ---- Read perturb configs (default off) ----
        self.smooth = bool(getattr(config, "use_perturbation", False))
        self.coef_learnable = bool(getattr(config, "coef_learnable", False))
        self.initial_coef = float(getattr(config, "perturb_std", 1e-2))
        if self.initial_coef <= 0.0:
            self.smooth = False
            self.initial_coef = 1e-2

        # Layer range
        perturb_start_layer = int(getattr(config, "perturb_start_layer", 0))
        perturb_end_layer = getattr(config, "perturb_end_layer", None)
        if perturb_end_layer is None:
            perturb_end_layer = int(getattr(config, "num_hidden_layers", 10**9))
        else:
            perturb_end_layer = int(perturb_end_layer)
        self.layer_needs_perturbation = (perturb_start_layer <= self.layer_idx < perturb_end_layer)

        # dtype
        dtype = getattr(config, "torch_dtype", torch.float32)
        if isinstance(dtype, str):
            if dtype == "bfloat16":
                dtype = torch.bfloat16
            elif dtype == "float16":
                dtype = torch.float16
            else:
                dtype = torch.float32

        log_init = math.log(self.initial_coef)

        # Only make it a Parameter if perturbation is enabled for this layer and learnable
        if self.smooth and self.layer_needs_perturbation and self.coef_learnable:
            self.log_coef = nn.Parameter(torch.tensor([log_init], dtype=dtype))
        else:
            self.register_buffer("log_coef", torch.tensor([log_init], dtype=dtype))

        with torch.no_grad():
            self.log_coef.fill_(log_init)

        # Seed set externally per micro-batch (by dp_actor)
        self._noise_seed: int = 0

        # ---- HF Qwen2 components (4.54.0) ----
        self.self_attn = modeling_qwen2.Qwen2Attention(config=config, layer_idx=self.layer_idx)
        self.mlp = modeling_qwen2.Qwen2MLP(config)
        self.input_layernorm = modeling_qwen2.Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = modeling_qwen2.Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # In 4.54.0, config.layer_types exists and is used by Qwen2Attention for sliding_window selection
        self.attention_type = config.layer_types[self.layer_idx]

    def _stateless_noise(self, h: torch.Tensor, seed: int) -> torch.Tensor:
        gen = torch.Generator(device=h.device)
        gen.manual_seed(int(seed) + int(self.layer_idx))
        return torch.randn_like(h, generator=gen)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,  # kept for BC; forwarded via **kwargs if needed
        past_key_value: Optional[Cache] = None,           # <-- 4.54.0 uses singular
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # BC: some callers may pass plural
        if past_key_value is None and "past_key_values" in kwargs:
            past_key_value = kwargs.pop("past_key_values")

        # ============================================================
        # Perturb BEFORE input_layernorm
        # ============================================================
        if self.smooth and self.training and self.layer_needs_perturbation:
            if self.coef_learnable and isinstance(self.log_coef, nn.Parameter):
                seed_tensor = torch.tensor(self._noise_seed, device=hidden_states.device, dtype=torch.int64)

                def _inject(h: torch.Tensor, log_coef: torch.Tensor, seed_t: torch.Tensor) -> torch.Tensor:
                    coef = log_coef.exp()
                    noise = self._stateless_noise(h, int(seed_t.item()))
                    return h + coef * noise

                hidden_states = checkpoint(
                    _inject,
                    hidden_states,
                    self.log_coef,
                    seed_tensor,
                    use_reentrant=False,
                )
            else:
                coef = self.log_coef.exp()
                noise = self._stateless_noise(hidden_states, self._noise_seed)
                hidden_states = hidden_states + coef * noise

        # ============================================================
        # HF 4.54.0 original decoder-layer body (keep structure)
        # ============================================================
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,            # passed through **kwargs/ignored if unused
            past_key_value=past_key_value,        # <-- match 4.54.0
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def apply_qwen2_patch():
    print("🚨 [Verl Patch] Applying Custom Qwen2 Perturbation Logic (HF 4.54.0 aligned)...")
    print("   -> Replacing transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer")

    # For FSDP wrap policy / _no_split_modules matching
    CustomQwen2DecoderLayer.__name__ = "Qwen2DecoderLayer"
    CustomQwen2DecoderLayer.__qualname__ = "Qwen2DecoderLayer"

    modeling_qwen2.Qwen2DecoderLayer = CustomQwen2DecoderLayer