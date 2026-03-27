
#!/bin/bash

export CUDA_HOME=/usr/local/cuda
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Configuration
model_name="all-perturb_sequence_inistd1e-5_clip_0.5_3.0_c10.0_lr1e-4_Qwen3-4B_dapo-math-17k_n8"
base_output_dir="/home/chenluy/mismatch-all_perturbation-on-math_new/data/gen_data/$model_name"
mkdir -p $base_output_dir

K=32
world_size=8
# gen_data.py: LLM(max_model_len=max_input_length). Must be >= prompt tokens + max_new_tokens.
# 32768 leaves headroom for long prompts + 16384 new tokens; lower if OOM (e.g. 24576).
MAX_NEW_TOKENS=16384
MAX_INPUT_LENGTH=32768

# Remove custom perturb params that standard vLLM Qwen3 loader does not accept.
filter_log_coef_from_merged() {
    local merged_dir="$1"
    python3 - "$merged_dir" <<'PY'
import json
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

merged_dir = Path(sys.argv[1])
safetensor_files = sorted(merged_dir.glob("*.safetensors"))
if not safetensor_files:
    print(f"[filter_log_coef] No safetensors found in {merged_dir}, skip.")
    raise SystemExit(0)

removed_count = 0
for shard_path in safetensor_files:
    state = load_file(str(shard_path))
    filtered = {k: v for k, v in state.items() if ".log_coef" not in k}
    removed = len(state) - len(filtered)
    if removed == 0:
        continue
    tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    save_file(filtered, str(tmp_path))
    tmp_path.replace(shard_path)
    removed_count += removed

index_path = merged_dir / "model.safetensors.index.json"
if index_path.exists():
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index_data.get("weight_map", {})
    index_data["weight_map"] = {k: v for k, v in weight_map.items() if ".log_coef" not in k}
    index_path.write_text(json.dumps(index_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(f"[filter_log_coef] {merged_dir}: removed {removed_count} log_coef tensors.")
PY
}

# Model and dataset arrays
models=()
base_model_path="/opt/dlami/nvme/chenluy_ckpoints/mismatch_rl_research/$model_name"

# # merge the model
# for step in $(seq 400 20 480); do
#     python /home/chenluy/mismatch-all_perturbation-on-math_new/scripts/legacy_model_merger.py merge \
#         --backend fsdp \
#         --local_dir $base_model_path/global_step_$step/actor \
#         --hf_model_path $base_model_path/global_step_$step/actor/huggingface \
#         --target_dir $base_model_path/global_step_$step/merged
#     filter_log_coef_from_merged "$base_model_path/global_step_$step/merged"
# done

# Generate model paths for global_step_20 to global_step_220 (increment by 20)
for step in $(seq 400 20 480); do
    models+=("$base_model_path/global_step_$step/merged")
done

datasets=("weqweasdas/olympiadbench" "weqweasdas/math500" "weqweasdas/minerva_math" "weqweasdas/aime24" "Chenlu123/aime25")

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
        
        # Generate data in parallel
        echo "Starting parallel data generation..."
        # we use gpu 4,5,6,7
        for i in 0 1 2 3 4 5 6 7; do
            CUDA_VISIBLE_DEVICES=$i python3 gen_data.py \
                --local_index $((i)) \
                --my_world_size $world_size \
                --model_name_or_path "$model_name" \
                --output_dir "$output_dir/" \
                --K $K \
                --max_input_length $MAX_INPUT_LENGTH \
                --max_new_tokens $MAX_NEW_TOKENS \
                --dataset_name_or_path "$dataset" &
        done
        
        # Wait for all parallel processes to complete
        wait
        echo "Data generation completed."
        
        # Merge the generated data
        echo "Merging data..."
        python3 merge_data.py \
            --base_path "$output_dir/" \
            --output_dir "$output_dir/merged_data.jsonl" \
            --num_datasets $world_size
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to merge data for $model_name on $dataset"
            continue
        fi
        
        # Compute scores
        echo "Computing scores..."
        python3 compute_score.py \
            --dataset_path "$output_dir/merged_data.jsonl" \
            --record_path "$output_dir/record.txt"
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to compute scores for $model_name on $dataset"
            continue
        fi
        
        echo "Completed evaluation for $model_name on $dataset"
        echo "Results saved to: $output_dir/record.txt"
        echo "----------------------------------------"

        echo "Compute minerval score... for minerva_math"
        if [ "$dataset" == "weqweasdas/minerva_math" ]; then
            python3 compute_score_minerval.py \
                --dataset_path "$output_dir/merged_data.jsonl" \
                --record_path "$output_dir/record_new.txt"
        fi
    done
done

echo "All evaluations completed!"