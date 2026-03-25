
#!/bin/bash

# Configuration
model_name="mis_baseline_sequence_mask_2.0_loss_sequence_0.5_3.0_Qwen3-4B_dapo-math-17k_n8_prompt_bsz_128_mini_bsz_32"
base_output_dir="/home/chenluy/mismatch-all_perturbation-on-math_new/data/gen_data/$model_name"
mkdir -p $base_output_dir

K=32
world_size=8

# Model and dataset arrays
models=()
base_model_path="/opt/dlami/nvme/chenluy_ckpoints/mismatch_rl_research/$model_name"

# merge the model
for step in $(seq 140 20 140); do
    python /home/chenluy/mismatch-all_perturbation-on-math_new/scripts/legacy_model_merger.py merge \
        --backend fsdp \
        --local_dir $base_model_path/global_step_$step/actor \
        --hf_model_path $base_model_path/global_step_$step/actor/huggingface \
        --target_dir $base_model_path/global_step_$step/merged
done

# Generate model paths for global_step_20 to global_step_220 (increment by 20)
for step in $(seq 140 20 140); do
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
                --max_new_tokens 16384 \
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