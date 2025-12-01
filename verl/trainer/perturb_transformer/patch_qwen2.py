import torch
from torch import nn
import math
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from typing import Optional, Tuple, Union

# ---------------------------------------------------------------------------- #
# 1. 重写 Qwen2DecoderLayer
#    (去掉了 smooth 参数，改为从 config 读取)
# ---------------------------------------------------------------------------- #
class CustomQwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # --- [关键修改]: 从 config 中读取自定义参数 ---
        # 如果 config.json 里没有写，就默认关闭 (False/1)
        self.smooth = getattr(config, "use_perturbation", False)
        self.coef_learnable = getattr(config, "coef_learnable", False)
        self.initial_coef = getattr(config, "perturb_std", 1e-2)
        
        # 处理 coef (扰动系数)
        # 确保类型为模型的 dtype (通常是 bfloat16 或 float32)
        dtype = getattr(config, "torch_dtype", torch.float32)
        # 如果 config.torch_dtype 是字符串，需要转换
        if isinstance(dtype, str):
             if dtype == "bfloat16":
                 dtype = torch.bfloat16
             elif dtype == "float16":
                 dtype = torch.float16
             else:
                 dtype = torch.float32

        if self.coef_learnable:
            # 如果想让它可训练，需注册为 Parameter
            # 使用 log 空间优化，保证 std 始终非负
            self.log_coef = nn.Parameter(torch.tensor([math.log(self.initial_coef)], dtype=dtype))
        else:
            # 固定值则注册为 buffer
            self.register_buffer("log_coef", torch.tensor([math.log(self.initial_coef)], dtype=dtype))
        
        # 强制初始化
        with torch.no_grad():
            self.log_coef.fill_(math.log(self.initial_coef))
        # --------------------------------------------------------

        self.self_attn = modeling_qwen2.Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = modeling_qwen2.Qwen2MLP(config)
        self.input_layernorm = modeling_qwen2.Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = modeling_qwen2.Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # --- [关键修复]: 从 config 中读取 layer_types，以匹配原版 Qwen2DecoderLayer 行为 ---
        # 解决 AttributeError: 'Qwen2DecoderLayer' object has no attribute 'attention_type'
        if hasattr(config, "layer_types"):
            self.attention_type = config.layer_types[layer_idx]
        else:
            # Fallback for standard Qwen2 configs that might not have layer_types
            # Standard Qwen2 usually uses full attention everywhere
            self.attention_type = "full_attention" 

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        
        # === 优化后的逻辑：只采样一次 (Noise Injection) ===
        # 仅在开启 smooth 且处于训练模式时执行
        if self.smooth and self.training:
            # 1. 准备系数 (确保在正确的 device)
            # 从 log 空间恢复 std: std = exp(log_std)
            current_coef = self.log_coef.to(hidden_states.device).exp()
            
            # 2. 生成噪声并注入 (Element-wise 操作，非常快)
            # torch.rand_like 生成 [0, 1) 的均匀分布噪声
            noise = torch.rand_like(hidden_states)
            perturbed_states = hidden_states + current_coef * noise
            
            # 3. 执行一次 Forward
            # 直接返回结果
            # 显式传入 update_key_value=True，确保 KV Cache 逻辑正确
            return self._process(
                hidden_states=perturbed_states, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                past_key_values=past_key_values, 
                use_cache=use_cache, 
                cache_position=cache_position, 
                position_embeddings=position_embeddings,
                update_key_value=True,
                **kwargs
            )
        
        # === 标准逻辑 (无扰动) ===
        else:
            return self._process(
                hidden_states=hidden_states, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                past_key_values=past_key_values, 
                use_cache=use_cache, 
                cache_position=cache_position, 
                position_embeddings=position_embeddings,
                **kwargs
            )

    def _process(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        update_key_value: bool = True,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # Qwen2 implementation in Transformers 4.56+ uses 'past_key_values'
        
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Transformers 4.56.1 Qwen2DecoderLayer returns only hidden_states (Tensor)
        return hidden_states

# ---------------------------------------------------------------------------- #
# 2. 定义 Patch 应用函数
# ---------------------------------------------------------------------------- #
def apply_qwen2_patch():
    print("🚨 [Verl Patch] Applying Custom Qwen2 Perturbation Logic...")
    print("   -> Replacing transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer")
    
    # Hack for FSDP wrap policy: set the class name to match the original one
    # Qwen2Model._no_split_modules is ["Qwen2DecoderLayer"]
    CustomQwen2DecoderLayer.__name__ = "Qwen2DecoderLayer"
    CustomQwen2DecoderLayer.__qualname__ = "Qwen2DecoderLayer"
    
    # 核心：替换 transformers 库中的类定义
    modeling_qwen2.Qwen2DecoderLayer = CustomQwen2DecoderLayer
    
    # 注意：通常只需要替换 DecoderLayer 即可。
    # 如果你也需要在 Model 级别做操作（比如 input_ids 维度的扰动），
    # 你也需要重写 Qwen2Model 并在这里替换：
    # modeling_qwen2.Qwen2Model = CustomQwen2Model