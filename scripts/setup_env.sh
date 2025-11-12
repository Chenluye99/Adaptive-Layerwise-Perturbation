#!/bin/bash
# Environment setup for all experiments
# Source this file at the beginning of each script

# Use verl_pert environment with modified verl
export CONDA_ENV="verl_pert"
export VERL_PATH="/home/zhang430/code/mismatch_rl_research/verl"
export WORK_DIR="/home/zhang430/code/mismatch_rl_research"

# Standard environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_API_KEY="de10a9d8ee68dcfcae3324b99a557c99ec7a1f32"
export WANDB_ENTITY="mismatch-perturbation"

# Add custom modules to Python path
export PYTHONPATH="${WORK_DIR}/src:${PYTHONPATH}"

echo "✓ Environment configured:"
echo "  - Conda env: ${CONDA_ENV}"
echo "  - VERL path: ${VERL_PATH}"
echo "  - Work dir: ${WORK_DIR}"

