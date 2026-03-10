#!/bin/bash
# 直接从 HuggingFace 加载模型（无需先下载到本地）。加载时会自动注入 use_perturbation / coef_learnable / perturb_std 等配置，无需修改 config.json。

set -e

LOSS_MODE="token"
PERTURB_STD="1e-5"
GEOMETRIC="false"
CLIP_RATIO_LOW="0.2"
CLIP_RATIO_HIGH="0.2"
CLIP_RATIO_C="10.0"
PERTURB_START="0"
PERTURB_END=""   # empty = last layer (inclusive)
PERTURB_LR="1e-6"
PERTURB_PATCH="qwen3" #qwen2/llama/qwen3
MODEL_BASE="Qwen"
MODEL_NAME="Qwen3-4B"

# Parse --name value arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --loss_mode)         LOSS_MODE="$2"; shift 2 ;;
    --perturb_std)       PERTURB_STD="$2"; shift 2 ;;
    --geometric)         GEOMETRIC="$2"; shift 2 ;;
    --clip_ratio_low)    CLIP_RATIO_LOW="$2"; shift 2 ;;
    --clip_ratio_high)   CLIP_RATIO_HIGH="$2"; shift 2 ;;
    --clip_ratio_c)      CLIP_RATIO_C="$2"; shift 2 ;;
    --perturb_start)     PERTURB_START="$2"; shift 2 ;;
    --perturb_end)       PERTURB_END="$2"; shift 2 ;;
    --perturb_lr)        PERTURB_LR="$2"; shift 2 ;;
    --perturb_patch)     PERTURB_PATCH="$2"; shift 2 ;;
    --model_base)        MODEL_BASE="$2"; shift 2 ;;
    --model_name)        MODEL_NAME="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "  (GPUs: always auto-detect all available, no --cuda)"
      echo "  --loss_mode MODE       token|sequence|cum-token|cum-turn (default: sequence)"
      echo "  --perturb_std STD      Perturbation initial std (default: 0.02)"
      echo "  --geometric true|false (default: false)"
      echo "  --clip_ratio_low LOW  (default: 0.2)"
      echo "  --clip_ratio_high HIGH (default: 0.28)"
      echo "  --clip_ratio_c C      (default: 10.0)"
      echo "  --perturb_start N     First layer to perturb, inclusive (default: 0)"
      echo "  --perturb_end N       Last layer to perturb, inclusive; omit or empty = last layer"
      echo "  --perturb_lr LR       Perturbation learning rate (default: 1e-2)"
      echo "  --perturb_patch PATCH qwen2|llama (default: qwen2)"
      echo "  --model_base BASE     HuggingFace org (default: Qwen)"
      echo "  --model_name NAME     Model name (default: Qwen2.5-Math-1.5B)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"; echo "Use --help for usage."; exit 1
      ;;
  esac
done

MODEL_PATH="${MODEL_BASE}/${MODEL_NAME}"

echo "=========================================="
echo "Experiment 1: GRPO Baseline"
echo "Loss Mode: ${LOSS_MODE}"
echo "Perturb Std: ${PERTURB_STD}"
echo "Geometric: ${GEOMETRIC}"
echo "Clip Ratio Low: ${CLIP_RATIO_LOW}"
echo "Clip Ratio High: ${CLIP_RATIO_HIGH}"
echo "Clip Ratio C: ${CLIP_RATIO_C}"
echo "Perturb layers: [${PERTURB_START}, ${PERTURB_END:-last}] (start inclusive, end inclusive)"
echo "Perturbation Learning Rate: ${PERTURB_LR}"
echo "Perturb Patch: ${PERTURB_PATCH}"
echo "Model: ${MODEL_PATH}"
echo "Using ${FREE_GPU_COUNT} GPUs: ${FREE_GPUS}"
echo "Start time: $(date)"
echo "=========================================="

export PERTURB_PATCH=${PERTURB_PATCH}
export WANDB_API_KEY="a17294c76f5787d04c92fd978d0f1a29133756e2"
export WANDB_ENTITY="mismatch"
export RAY_TMPDIR=/opt/dlami/nvme/ray_tmp

max_prompt_length=$((2048 * 1))
max_response_length=$((16384))
train_prompt_bsz=128
n_resp_per_prompt=8
train_prompt_mini_bsz=32
loss_agg_mode="token-mean"

# Data files (use absolute paths)
USE_PERTURBATION=True
project_name="mismatch_rl_research"
dataset_name="merged_openr1_guru" # openr1 or merged_openr1_guru
exp_name="all-perturb_${LOSS_MODE}_inistd${PERTURB_STD}_clip_${CLIP_RATIO_LOW}_${CLIP_RATIO_HIGH}_c${CLIP_RATIO_C}_lr${PERTURB_LR}_${MODEL_NAME}_${dataset_name}_n${n_resp_per_prompt}"
if [ "$GEOMETRIC" = "true" ]; then
    exp_name="${exp_name}_geo"
fi

CKPTS_DIR="/opt/dlami/nvme/chenluy_ckpoints/${project_name}/${exp_name}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Suppress pynvml deprecation warning
export PYTHONWARNINGS="ignore::FutureWarning"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="/home/chenluy/data/${dataset_name}/train.parquet" \
    data.val_files="/home/chenluy/data/${dataset_name}/test.parquet" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.use_perturbation=${USE_PERTURBATION} \
    actor_rollout_ref.actor.perturb_std=${PERTURB_STD} \
    actor_rollout_ref.actor.perturb_start=${PERTURB_START} \
    actor_rollout_ref.actor.perturb_end=${PERTURB_END:-null} \
    +actor_rollout_ref.actor.perturb_lr=${PERTURB_LR} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    actor_rollout_ref.actor.policy_loss.is_geometric=${GEOMETRIC} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C} \
    actor_rollout_ref.actor.perturb_patch=${PERTURB_PATCH} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    +algorithm.rollout_correction.rollout_is=null \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=${FREE_GPU_COUNT} \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_epochs=50 \
    2>&1 | tee logs/${exp_name}.log

echo "=========================================="
echo "Experiment completed: $(date)"
echo "=========================================="
