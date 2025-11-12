#!/bin/bash
# Experiment 1: GRPO Baseline (no rollout correction)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils_gpu.sh"

# Get free GPUs
FREE_GPUS=$(get_free_gpus)
FREE_GPU_COUNT=$(get_free_gpu_count)

if [ -z "$FREE_GPUS" ]; then
    echo "Error: No free GPUs available"
    exit 1
fi

echo "=========================================="
echo "Experiment 1: GRPO Baseline"
echo "Using ${FREE_GPU_COUNT} free GPUs: ${FREE_GPUS}"
echo "Start time: $(date)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=${FREE_GPUS}
export WANDB_API_KEY="a17294c76f5787d04c92fd978d0f1a29133756e2"
export WANDB_ENTITY="mismatch"
export DISABLE_FLASH_ATTN=1
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager

source "${SCRIPT_DIR}/setup_env.sh"

clip_ratio_low=0.2
clip_ratio_high=0.28
max_prompt_length=$((1024 * 1))
max_response_length=$((2048))
train_prompt_bsz=256
n_resp_per_prompt=8
train_prompt_mini_bsz=64
loss_agg_mode="token-mean"

# Data files (use absolute paths)
train_file="/home/zhang430/data/openr1/train.parquet" #TODO: change according to https://github.com/LLM360/Reasoning360/blob/main/scripts/tools/download_guru.py
val_file="/home/zhang430/data/openr1/test.parquet"

cd /home/zhang430/code/mismatch_rl_research

python3 -m verl.trainer.main_ppo \
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
    '+actor_rollout_ref.model.override_config={attn_implementation:eager,torch_dtype:float16}' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
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
    +algorithm.rollout_correction.rollout_is=null \
    trainer.critic_warmup=0 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=mismatch_rl_research \
    trainer.experiment_name=exp1_grpo_baseline \
    trainer.n_gpus_per_node=${FREE_GPU_COUNT} \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=50 \
    2>&1 | tee logs/exp1_grpo_baseline.log

echo "=========================================="
echo "Experiment completed: $(date)"
echo "=========================================="
