
#!/bin/bash

# Configuration
base_output_dir="/home/chenluy/mismatch_all_perturb_agent/eval/gen_data/simpletir__maxturn5_batch256_ppomini32_lossmodevanilla_maskvoidturnsTrue_risTrue_islvlsequence_isth5.0_deepscaler_merge_train_Qwen2.5-7B"
mkdir -p $base_output_dir

K=32
world_size=8

# Base model path (for tokenizer and config)
BASE_MODEL_PATH="Qwen/Qwen2.5-7B"

# Model and dataset arrays
models=()
base_model_path="/opt/dlami/nvme/chenluy_ckpoints/test/simpletir__maxturn5_batch256_ppomini32_lossmodevanilla_maskvoidturnsTrue_risTrue_islvlsequence_isth5.0_deepscaler_merge_train_Qwen2.5-7B"

# merge the model
echo "=== Starting model merging ==="
for step in $(seq 20 20 20); do
    actor_dir="$base_model_path/global_step_$step/actor"
    merged_dir="$base_model_path/global_step_$step/merged"
    
    if [ -d "$merged_dir" ]; then
        echo "✓ Merged model exists for step $step, skipping..."
        continue
    fi
    
    echo "Preparing actor directory for step $step..."
    
    # Copy tokenizer files from base model to actor directory if not exists
    if [ ! -f "$actor_dir/tokenizer_config.json" ]; then
        echo "  Downloading and copying tokenizer files from $BASE_MODEL_PATH..."
        python3 -c "
from transformers import AutoTokenizer
import shutil
import os

tokenizer = AutoTokenizer.from_pretrained('$BASE_MODEL_PATH', trust_remote_code=True)
tokenizer.save_pretrained('$actor_dir')
print('✓ Tokenizer files copied to actor directory')
"
    fi
    
    echo "Merging model for step $step..."
    python3 /home/chenluy/mismatch_all_perturb_agent/scripts/model_merger.py \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --hf_model_path "$actor_dir" \
        --target_dir "$merged_dir"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully merged step $step"
    else
        echo "✗ Failed to merge step $step"
        exit 1
    fi
done

# Generate model paths for global_step_20 to global_step_220 (increment by 20)
for step in $(seq 20 20 20); do
    models+=("$base_model_path/global_step_$step/merged")
done

datasets=("weqweasdas/math500" "weqweasdas/minerva_math" "weqweasdas/olympiadbench" "weqweasdas/aime24" "Chenlu123/aime25" "Chenlu123/hmmt0225")

# Create base output directory
mkdir -p $base_output_dir

# Loop through models and datasets
for model_name in "${models[@]}"; do
    echo "Testing model: $model_name"
    
    for dataset in "${datasets[@]}"; do
        echo "Testing dataset: $dataset"
        
        # Create model/dataset specific output directory
        # Extract global_step_X/merged from the full path
        model_step_dir=$(echo "$model_name" | sed 's|.*/\(global_step_[0-9]*/merged\)|\1|')
        output_dir="$base_output_dir/$model_step_dir/$dataset"
        mkdir -p "$output_dir"
        
        echo "Output directory: $output_dir"
        
        # Extract model parent directory and model name
        model_parent_dir=$(dirname "$model_name")
        model_basename=$(basename "$model_name")
        
        # Run validation using train.sh
        echo "Starting validation using train.sh..."
        cd /home/chenluy/mismatch_all_perturb_agent
        
        # Disable wandb for validation
        export WANDB_MODE=disabled
        
        MODEL_PATH="$model_parent_dir" \
        DATA_PATH=/home/chenluy/mismatch_all_perturb_agent/datasets \
        CHECKPOINT_PATH="$output_dir" \
        NNODES=1 \
        GPUS_PER_NODE=8 \
        RESUME=False \
        CONFIG_NAME=simpletir_trainer \
        bash train.sh \
          --max_response_length 8000 \
          --max_prompt_length 16000 \
          --model_path "$model_parent_dir" \
          --model_name "$model_basename" \
          --max_turns 5 \
          --valid_dataset "$dataset" \
          --val_only True \
          --n_val $K \
          --output_acc_to_file True \
          --val_sample_size null \
          --val_batch_size 256 \
          --sp_size 1 \
          --total_epochs 1 \
          --val_temperature 1.0
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to run validation for $model_name on $dataset"
            continue
        fi
        
        echo "Completed evaluation for $model_name on $dataset"
        echo "Results saved to: $output_dir"
        echo "----------------------------------------"
    done
done

echo "All evaluations completed!"