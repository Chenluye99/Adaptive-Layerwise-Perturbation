#!/bin/bash
# Experiment: GRPO Baseline (no rollout correction)
# Usage: bash run_exp_baseline.sh [eager]
#
# Prerequisites: conda activate verl_new
#
# Examples:
#   bash run_exp_baseline.sh          # Standard baseline with Flash Attention
#   bash run_exp_baseline.sh true     # Use eager attention
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils_gpu.sh"

FREE_GPUS="0,1,2,3,4,5,6,7"
FREE_GPU_COUNT=8
export CUDA_VISIBLE_DEVICES=${FREE_GPUS}
echo "Using fixed GPUs: ${CUDA_VISIBLE_DEVICES}"

# Set Ray temp directory (Docker-compatible)
if [ -d "/workspace/project" ]; then
    # Running in Docker
    export RAY_TMPDIR=/tmp/ray_tmp_${FREE_GPU_COUNT}gpu
else
    # Running locally
    export RAY_TMPDIR=/home/zhang430/.cache/ray_tmp_${FREE_GPU_COUNT}gpu
fi
export NCCL_P2P_DISABLE=1

# Set WandB credentials
export WANDB_API_KEY="de10a9d8ee68dcfcae3324b99a557c99ec7a1f32"

mkdir -p $RAY_TMPDIR
chmod -R 777 $RAY_TMPDIR 2>/dev/null || true

clip_ratio_low=0.2
clip_ratio_high=0.28
max_prompt_length=$((1024 * 1))
max_response_length=$((2048))

# Fixed configuration aligned with chenlu_exp_tis.sh (8 GPUs only)
train_prompt_bsz=256
n_resp_per_prompt=8
train_prompt_mini_bsz=32
ppo_micro_batch_size=4
rollout_log_prob_micro_bsz=4
ref_log_prob_micro_bsz=4
gpu_memory_util=0.7
ray_num_cpus=64
loss_agg_mode="token-mean"

echo "=========================================="
echo "Experiment: GRPO Baseline"
echo "Using ${FREE_GPU_COUNT} GPUs: ${FREE_GPUS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "----------------------------------------"
echo "Batch Configuration:"
echo "  train_prompt_bsz: ${train_prompt_bsz}"
echo "  n_resp_per_prompt: ${n_resp_per_prompt}"
echo "  train_prompt_mini_bsz: ${train_prompt_mini_bsz}"
echo "  ppo_micro_batch_size: ${ppo_micro_batch_size}"
echo "  rollout_log_prob_micro_bsz: ${rollout_log_prob_micro_bsz}"
echo "  ref_log_prob_micro_bsz: ${ref_log_prob_micro_bsz}"
echo "  gpu_memory_util: ${gpu_memory_util}"
echo "  ray_num_cpus: ${ray_num_cpus}"
echo "----------------------------------------"
echo "Start time: $(date)"
echo "=========================================="

# Detect if running in Docker container and set paths accordingly
if [ -d "/workspace/project" ]; then
    DATA_ROOT="/data/guru_rl92k"
else
    DATA_ROOT="/home/zhang430/data/guru_rl92k"
fi
DATA_PARENT=$(dirname "${DATA_ROOT}")

train_file="${DATA_ROOT}/train/train/math__combined_54.4k.parquet"
val_file="${DATA_PARENT}/guru_rl92k_prepared/math_eval.parquet"
val_files_json="[\"${val_file}\"]"

if [ ! -f "${train_file}" ]; then
    echo "Error: train file ${train_file} not found" >&2
    exit 1
fi

if [ ! -f "${val_file}" ]; then
    echo "Error: val file ${val_file} not found" >&2
    exit 1
fi

# Model configuration
MODEL_PATH="Qwen/Qwen2.5-Math-1.5B"
MODEL_ID=$(echo "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g')

DATASET_NAME="guru_rl92k_math"

# Generate experiment name
project_name="mismatch_rl_research"
EXP_NAME="exp_grpo_${MODEL_ID}_${DATASET_NAME}"
if [ "$EAGER" = "true" ]; then
    EXP_NAME="${EXP_NAME}_eager"
fi

# Checkpoint directory
CKPTS_DIR="${CKPTS_BASE}/${project_name}/${EXP_NAME}"

# Change to project directory (handle both Docker and local environments)
if [ -d "/workspace/project" ]; then
    # Running in Docker container
    cd /workspace/project
else
    # Running locally
    cd /home/zhang430/code/mismatch_rl
fi

# Create logs and outputs directories
mkdir -p logs outputs
chmod 777 logs outputs 2>/dev/null || true

python3 -m verl.trainer.main_ppo \
    hydra.run.dir=outputs/${EXP_NAME}/${now:%Y-%m-%d}/${now:%H-%M-%S} \
    algorithm.adv_estimator=grpo \
    data.train_files="[\"${train_file}\"]" \
    data.val_files="${val_files_json}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.nccl_timeout=1800 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size} \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_bsz} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_util} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_bsz} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    +algorithm.rollout_correction.rollout_is=null \
    trainer.critic_warmup=0 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${EXP_NAME} \
    +trainer.wandb_entity=mismatch \
    +trainer.wandb_mode=online \
    +trainer.wandb_tags=["grpo","baseline"] \
    +trainer.wandb_config.clip_ratio_low=${clip_ratio_low} \
    +trainer.wandb_config.clip_ratio_high=${clip_ratio_high} \
    +trainer.wandb_config.loss_agg_mode=${loss_agg_mode} \
    trainer.n_gpus_per_node=${FREE_GPU_COUNT} \
    trainer.nnodes=1 \
    +trainer.ray_init.num_gpus=${FREE_GPU_COUNT} \
    +trainer.ray_init.num_cpus=${ray_num_cpus} \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_epochs=50 \
    2>&1 | tee logs/${EXP_NAME}.log

echo "=========================================="
echo "Experiment completed: $(date)"
echo "=========================================="
