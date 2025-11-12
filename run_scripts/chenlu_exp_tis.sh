#!/bin/bash
# Unified Experiment: GRPO + Truncated Importance Sampling (TIS)
# Usage: bash run_exp_tis.sh [level] [mode] [threshold] [veto_threshold] [geometric]
#
# Examples:
#   bash run_exp_tis.sh token                    # Token-level TIS
#   bash run_exp_tis.sh sequence                 # Sequence-level TIS
#   bash run_exp_tis.sh cum-token                # Cumulative token TIS
#   bash run_exp_tis.sh cum-turn                 # Cumulative turn TIS
#   bash run_exp_tis.sh sequence truncate 5.0    # Custom threshold
#   bash run_exp_tis.sh token mask 3.0 0.001     # Mask mode with veto
#   bash run_exp_tis.sh cum-token truncate 5.0 0.0 true  # Geometric aggregation
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils_gpu.sh"

# Set GPU devices here (comma-separated, e.g., "0,1,2,3")
# If not set, will auto-detect free GPUs
CUDA_VISIBLE_DEVICES="0,1,2,3"

# Parse arguments
TIS_LEVEL="${1:-sequence}"             # token/sequence/cum-token/cum-turn
TIS_MODE="${2:-truncate}"              # truncate/mask
TIS_THRESHOLD="${3:-5.0}"              # Upper threshold
VETO_THRESHOLD="${4:-0.0}"             # Veto threshold (0.0=disabled)
GEOMETRIC="${5:-false}"                # Geometric aggregation

# Validate level
if [[ "$TIS_LEVEL" != "token" && "$TIS_LEVEL" != "sequence" && "$TIS_LEVEL" != "cum-token" && "$TIS_LEVEL" != "cum-turn" ]]; then
    echo "Error: Invalid TIS level. Use 'token', 'sequence', 'cum-token', or 'cum-turn'"
    exit 1
fi

# Validate mode
if [[ "$TIS_MODE" != "truncate" && "$TIS_MODE" != "mask" ]]; then
    echo "Error: Invalid TIS mode. Use 'truncate' or 'mask'"
    exit 1
fi

# Get GPUs: use CUDA_VISIBLE_DEVICES if set in script, otherwise auto-detect
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Use manually specified GPUs from script
    FREE_GPUS="$CUDA_VISIBLE_DEVICES"
    # Count GPUs: split by comma and count
    FREE_GPU_COUNT=$(echo "$FREE_GPUS" | tr ',' '\n' | wc -l)
    echo "Using manually specified GPUs: ${FREE_GPUS}"
else
    # Auto-detect free GPUs
    FREE_GPUS=$(get_free_gpus)
    FREE_GPU_COUNT=$(get_free_gpu_count)
    
    if [ -z "$FREE_GPUS" ]; then
        echo "Error: No free GPUs available"
        exit 1
    fi
    echo "Auto-detected free GPUs: ${FREE_GPUS}"
fi

echo "=========================================="
echo "Experiment 1: GRPO Baseline"
echo "Using ${FREE_GPU_COUNT} GPUs: ${FREE_GPUS}"
echo "Start time: $(date)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=${FREE_GPUS}
export WANDB_API_KEY="a17294c76f5787d04c92fd978d0f1a29133756e2"
export WANDB_ENTITY="mismatch"
export RAY_TMPDIR=/opt/dlami/nvme/ray_tmp

#source "${SCRIPT_DIR}/setup_env.sh"
MODEL_PATH="Qwen/Qwen2.5-Math-1.5B"

clip_ratio_low=0.2
clip_ratio_high=0.28
max_prompt_length=$((1024 * 1))
max_response_length=$((2048))
train_prompt_bsz=256
n_resp_per_prompt=8
train_prompt_mini_bsz=64
loss_agg_mode="token-mean"

# Data files
train_file="/home/zhang430/data/openr1/train.parquet"
val_file="/home/zhang430/data/openr1/test.parquet"

# Generate experiment name
EXP_NAME="grpo_tis_qwen2.5-1.5b-math_n${n_resp_per_prompt}_${TIS_LEVEL}_${TIS_MODE}_th${TIS_THRESHOLD}"
if (( $(echo "$VETO_THRESHOLD > 0" | bc -l) )); then
    EXP_NAME="${EXP_NAME}_veto${VETO_THRESHOLD}"
fi
if [ "$GEOMETRIC" = "true" ]; then
    EXP_NAME="${EXP_NAME}_geo"
fi

cd /home/chenluy/mismatch-perturbation-on-math

python3 -m verl.trainer.main_ppo \
    ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH=/home/zhang430/miniconda3/envs/verl_pert/lib:\${LD_LIBRARY_PATH} \
    algorithm.adv_estimator=grpo \
    data.train_files="${train_file}" \
    data.val_files="${val_file}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-1.5B \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    +algorithm.rollout_correction.rollout_is=${TIS_LEVEL} \
    +algorithm.rollout_correction.rollout_is_threshold=${TIS_THRESHOLD} \
    +algorithm.rollout_correction.rollout_is_mode=${TIS_MODE} \
    +algorithm.rollout_correction.rollout_is_veto_threshold=${VETO_THRESHOLD} \
    +algorithm.rollout_correction.rollout_is_geometric=${GEOMETRIC} \
    trainer.critic_warmup=0 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=mismatch_rl_research \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${FREE_GPU_COUNT} \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=50 \
    2>&1 | tee logs/${EXP_NAME}.log

echo "=========================================="
echo "Experiment completed: $(date)"
echo "=========================================="
