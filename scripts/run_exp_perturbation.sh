#!/bin/bash
# Unified Experiment: GRPO + Bypass + Perturbation
# Usage: bash run_exp_perturbation.sh [loss_mode] [perturb_std] [geometric]
#
# Examples:
#   bash run_exp_perturbation.sh                     # Default: sequence, 0.02
#   bash run_exp_perturbation.sh token               # Token-level
#   bash run_exp_perturbation.sh sequence            # Sequence-level
#   bash run_exp_perturbation.sh cum-token           # Cumulative token
#   bash run_exp_perturbation.sh cum-turn            # Cumulative turn
#   bash run_exp_perturbation.sh sequence 0.05       # Custom std
#   bash run_exp_perturbation.sh sequence 0.02 true  # With geometric aggregation
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils_gpu.sh"

# Parse arguments
LOSS_MODE="${1:-sequence}"       # token/sequence/cum-token/cum-turn
PERTURB_STD="${2:-0.02}"        # Perturbation std
GEOMETRIC="${3:-false}"         # Geometric aggregation

# Validate loss mode
if [[ "$LOSS_MODE" != "token" && "$LOSS_MODE" != "sequence" && "$LOSS_MODE" != "cum-token" && "$LOSS_MODE" != "cum-turn" ]]; then
    echo "Error: Invalid loss mode. Use 'token', 'sequence', 'cum-token', or 'cum-turn'"
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
echo "Experiment: GRPO + Bypass + Perturbation"
echo "Loss Mode: ${LOSS_MODE}"
echo "Perturb Std: ${PERTURB_STD}"
echo "Geometric: ${GEOMETRIC}"
echo "Using ${FREE_GPU_COUNT} free GPUs: ${FREE_GPUS}"
echo "Start time: $(date)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=${FREE_GPUS}
export LD_LIBRARY_PATH=/home/zhang430/miniconda3/envs/verl_pert/lib:$LD_LIBRARY_PATH

source "${SCRIPT_DIR}/setup_env.sh"

# Data files
train_file="/home/zhang430/data/openr1/train.parquet"
val_file="/home/zhang430/data/openr1/test.parquet"

# Generate experiment name
EXP_NAME="exp_perturb_${LOSS_MODE}_std${PERTURB_STD}"
if [ "$GEOMETRIC" = "true" ]; then
    EXP_NAME="${EXP_NAME}_geo"
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
    actor_rollout_ref.actor.perturb_std=${PERTURB_STD} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    actor_rollout_ref.actor.policy_loss.is_geometric=${GEOMETRIC} \
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
    +algorithm.rollout_correction.bypass_old_logprob_for_rollout=true \
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
