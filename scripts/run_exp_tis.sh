#!/bin/bash
# Unified Experiment: GRPO + Truncated Importance Sampling
# Usage: bash run_exp_grpo_tis.sh [token|sequence]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils_gpu.sh"

# Parse TIS level argument (default: sequence)
TIS_LEVEL="${1:-sequence}"

# Validate TIS level
if [[ "$TIS_LEVEL" != "token" && "$TIS_LEVEL" != "sequence" ]]; then
    echo "Error: Invalid TIS level. Use 'token' or 'sequence'"
    echo "Usage: bash $0 [token|sequence]"
    exit 1
fi

# Get free GPUs
FREE_GPUS=$(get_free_gpus)
FREE_GPU_COUNT=$(get_free_gpu_count)

if [ -z "$FREE_GPUS" ]; then
    echo "Error: No free GPUs available"
    exit 1
fi

echo "=========================================="
echo "Experiment: GRPO + ${TIS_LEVEL}-level TIS"
echo "Using ${FREE_GPU_COUNT} free GPUs: ${FREE_GPUS}"
echo "Start time: $(date)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=${FREE_GPUS}

# Use conda's libstdc++ to support Flash Attention
export LD_LIBRARY_PATH=/home/zhang430/miniconda3/envs/verl_pert/lib:$LD_LIBRARY_PATH

source "${SCRIPT_DIR}/setup_env.sh"

# Data files (use absolute paths)
train_file="/home/zhang430/data/openr1/train.parquet"
val_file="/home/zhang430/data/openr1/test.parquet"

# Set experiment name based on TIS level
if [ "$TIS_LEVEL" = "token" ]; then
    EXP_NAME="exp3_grpo_token_tis"
else
    EXP_NAME="exp4_grpo_seq_tis"
fi

cd /home/zhang430/code/mismatch_rl_research

python3 -m verl.trainer.main_ppo \
    ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH=/home/zhang430/miniconda3/envs/verl_pert/lib:\${LD_LIBRARY_PATH} \
    algorithm.adv_estimator=grpo \
    data.train_files="${train_file}" \
    data.val_files="${val_file}" \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    +algorithm.rollout_correction.rollout_is=${TIS_LEVEL} \
    +algorithm.rollout_correction.rollout_is_threshold=5.0 \
    trainer.critic_warmup=0 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=mismatch_rl_research \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${FREE_GPU_COUNT} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=50 \
    2>&1 | tee logs/${EXP_NAME}.log

echo "=========================================="
echo "Experiment completed: $(date)"
echo "=========================================="

