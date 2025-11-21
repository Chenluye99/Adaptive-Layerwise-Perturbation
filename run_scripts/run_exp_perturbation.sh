#!/bin/bash
# Unified Experiment: GRPO + Bypass + Perturbation
# Usage: bash run_exp_perturbation.sh [loss_mode] [perturb_std] [geometric] [dataset]
#
# Examples:
#   bash run_exp_perturbation.sh                     # Default: sequence, 0.02, guru
#   bash run_exp_perturbation.sh token               # Token-level (guru)
#   bash run_exp_perturbation.sh sequence            # Sequence-level (guru)
#   bash run_exp_perturbation.sh sequence 0.05       # Custom std (guru)
#   bash run_exp_perturbation.sh sequence 0.02 false openr1  # Use openr1 dataset
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils_gpu.sh"

# Parse arguments
LOSS_MODE="${1:-sequence}"       # token/sequence/cum-token/cum-turn
PERTURB_STD="${2:-0.02}"        # Perturbation std
GEOMETRIC="${3:-false}"         # Geometric aggregation
DATASET="${4:-guru}"             # guru or openr1

# Validate loss mode
if [[ "$LOSS_MODE" != "token" && "$LOSS_MODE" != "sequence" && "$LOSS_MODE" != "cum-token" && "$LOSS_MODE" != "cum-turn" ]]; then
    echo "Error: Invalid loss mode. Use 'token', 'sequence', 'cum-token', or 'cum-turn'"
    exit 1
fi

FREE_GPUS="0,1,2,3,4,5,6,7"
FREE_GPU_COUNT=8
export CUDA_VISIBLE_DEVICES=${FREE_GPUS}
echo "Using fixed GPUs: ${FREE_GPUS}"


# Set RAY_TMPDIR - use workspace directory in Docker, or home directory locally
if [ -d "/workspace/project" ]; then
    # Running in Docker container
    export RAY_TMPDIR=/workspace/project/.cache/ray_tmp_${FREE_GPU_COUNT}gpu
else
    # Running locally
    export RAY_TMPDIR=/home/zhang430/.cache/ray_tmp_${FREE_GPU_COUNT}gpu
fi
export NCCL_P2P_DISABLE=1

# Set WandB credentials
export WANDB_API_KEY="de10a9d8ee68dcfcae3324b99a557c99ec7a1f32"

mkdir -p $RAY_TMPDIR 2>/dev/null || true
chmod -R 777 $RAY_TMPDIR 2>/dev/null || true

echo "=========================================="
echo "Experiment: GRPO + Bypass + Perturbation"
echo "Dataset: ${DATASET}"
echo "Loss Mode: ${LOSS_MODE}"
echo "Perturb Std: ${PERTURB_STD}"
echo "Geometric: ${GEOMETRIC}"
echo "Using ${FREE_GPU_COUNT} GPUs: ${FREE_GPUS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "----------------------------------------"
echo "Start time: $(date)"
echo "=========================================="

clip_ratio_low=0.5
clip_ratio_high=3.0

# Shared parameters
max_prompt_length=$((2048 * 1))
max_response_length=$((2048))
train_prompt_bsz=512
n_resp_per_prompt=8
train_prompt_mini_bsz=32
ppo_micro_batch_size=8
rollout_log_prob_micro_bsz=8
ref_log_prob_micro_bsz=8
gpu_memory_util=0.7
ray_num_cpus=64
loss_agg_mode="token-mean"

# Detect if running in Docker container and set paths accordingly
if [ "$DATASET" = "openr1" ]; then
    # openr1 dataset paths
    if [ -d "/workspace/project" ]; then
        train_file="/data/openr1/train.parquet"
        val_file="/data/openr1/test.parquet"
    else
        train_file="/home/zhang430/data/openr1/train.parquet"
        val_file="/home/zhang430/data/openr1/test.parquet"
    fi
    DATASET_NAME="openr1"
else
    # guru_rl92k dataset paths
    if [ -d "/workspace/project" ]; then
        DATA_ROOT="/data/guru_rl92k"
    else
        DATA_ROOT="/home/zhang430/data/guru_rl92k"
    fi
    train_file="${DATA_ROOT}/train/math__combined_54.4k.parquet"
    val_file="[${DATA_ROOT}/online_eval/math__math_500.parquet,${DATA_ROOT}/online_eval/math__aime_repeated_8x_240.parquet]"
    DATASET_NAME="guru_rl92k_math"
fi

# Model configuration
MODEL_PATH="Qwen/Qwen2.5-Math-1.5B"
MODEL_ID=$(echo "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g')

# Generate experiment name
project_name="mismatch_rl_research"
EXP_NAME="exp_perturb_analysis_${MODEL_ID}_${DATASET_NAME}_${LOSS_MODE}_std${PERTURB_STD}_clip${clip_ratio_low}_${clip_ratio_high}_n${n_resp_per_prompt}_bz${train_prompt_bsz}_mini_bz${train_prompt_mini_bsz}"
if [ "$GEOMETRIC" = "true" ]; then
    EXP_NAME="${EXP_NAME}_geo"
fi

# Checkpoint directory (handle both local and Docker environments)
if [ -d "/workspace/project" ]; then
    # Running in Docker container
    CKPTS_DIR="/checkpoints/${project_name}/${EXP_NAME}"
else
    # Running locally
    CKPTS_DIR="/home/zhang430/checkpoints/${project_name}/${EXP_NAME}"
fi

# Change to project directory (handle both Docker and local environments)
if [ -d "/workspace/project" ]; then
    # Running in Docker container
    cd /workspace/project
else
    # Running locally
    cd /home/zhang430/code/mismatch_rl
fi

# Create logs, outputs, and checkpoint directories
mkdir -p logs outputs
chmod 777 logs outputs 2>/dev/null || true
mkdir -p "${CKPTS_DIR}"
chmod -R 777 "${CKPTS_DIR}" 2>/dev/null || true

python3 -m verl.trainer.main_ppo \
    hydra.run.dir=outputs/${EXP_NAME}/${now:%Y-%m-%d}/${now:%H-%M-%S} \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${train_prompt_bsz} \
    data.train_files=${train_file} \
    data.val_files=${val_file} \
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
    actor_rollout_ref.actor.perturb_std=${PERTURB_STD} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    actor_rollout_ref.actor.policy_loss.is_geometric=${GEOMETRIC} \
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
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_util} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_bsz} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    +algorithm.rollout_correction.rollout_is=null \
    +algorithm.rollout_correction.bypass_old_logprob_for_rollout=true \
    trainer.critic_warmup=0 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${EXP_NAME} \
    +trainer.wandb_entity=mismatch \
    +trainer.wandb_mode=online \
    +trainer.wandb_tags=["grpo","perturbation","${LOSS_MODE}"] \
    +trainer.wandb_config.loss_mode=${LOSS_MODE} \
    +trainer.wandb_config.perturb_std=${PERTURB_STD} \
    +trainer.wandb_config.is_geometric=${GEOMETRIC} \
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
