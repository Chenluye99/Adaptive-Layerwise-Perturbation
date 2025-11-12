#!/usr/bin/env python3
"""
Prepare weqweasdas/from_default_filtered_openr1_with_scores dataset
Filter by average score between 0 and 1
"""

import os
import argparse
import datasets
import pandas as pd
from pathlib import Path


def filter_by_score(example):
    """Filter examples with average score > 0 and < 1"""
    scores = example.get('scores', [])
    if not scores or len(scores) == 0:
        return False
    
    avg_score = sum(scores) / len(scores)
    return 0 < avg_score < 1


def process_dataset(split='train', local_dir='~/data/openr1', score_range=(0, 1)):
    """
    Load and process the dataset
    
    Args:
        split: Dataset split ('train' or 'test')
        local_dir: Local directory to save processed data
        score_range: Tuple of (min_score, max_score) for filtering
    """
    local_dir = os.path.expanduser(local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Loading {split} split from weqweasdas/from_default_filtered_openr1_with_scores...")
    
    # Load dataset
    dataset = datasets.load_dataset(
        'weqweasdas/from_default_filtered_openr1_with_scores',
        split=split,
        trust_remote_code=True
    )
    
    print(f"Original dataset size: {len(dataset)}")
    
    # Filter by score
    print(f"Filtering by score range {score_range}...")
    filtered_dataset = dataset.filter(filter_by_score)
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    
    # Process data format
    def process_example(example, idx):
        # Extract prompt and ground truth
        prompt = example.get('prompt', example.get('question', ''))
        answer = example.get('answer', '')
        scores = example.get('scores', [])
        
        # Format for verl
        data = {
            "data_source": "openr1_filtered",
            "prompt": [{
                "role": "user",
                "content": prompt,
            }],
            "ability": "reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'scores': scores,
                'avg_score': sum(scores) / len(scores) if scores else 0,
                'original_prompt': prompt,
                'original_answer': answer,
            }
        }
        return data
    
    # Apply processing
    processed_dataset = filtered_dataset.map(
        function=process_example,
        with_indices=True,
        remove_columns=filtered_dataset.column_names
    )

    print(processed_dataset[0])
    print(f"Processed dataset size: {len(processed_dataset)}")
    
    # Save to parquet
    output_file = os.path.join(local_dir, f'{split}.parquet')
    processed_dataset.to_parquet(output_file)
    print(f"✓ Saved to {output_file}")
    
    # Print statistics
    if len(processed_dataset) > 0:
        df = processed_dataset.to_pandas()
        avg_scores = [info['avg_score'] for info in df['extra_info']]
        print(f"\nDataset statistics:")
        print(f"  - Size: {len(processed_dataset)}")
        print(f"  - Avg score range: [{min(avg_scores):.3f}, {max(avg_scores):.3f}]")
        print(f"  - Mean avg score: {sum(avg_scores)/len(avg_scores):.3f}")
    
    return processed_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/openr1', 
                       help='Local directory to save processed data')
    parser.add_argument('--min_score', type=float, default=0,
                       help='Minimum average score')
    parser.add_argument('--max_score', type=float, default=1,
                       help='Maximum average score')
    
    args = parser.parse_args()
    
    print("="*60)
    print("OpenR1 Dataset Preparation")
    print("="*60)
    
    # Process train split
    print("\n[1/2] Processing train split...")
    train_dataset = process_dataset(
        split='train',
        local_dir=args.local_dir,
        score_range=(args.min_score, args.max_score)
    )
    
    # Process test split (use train if no test split exists)
    print("\n[2/2] Processing test split...")
    try:
        test_dataset = process_dataset(
            split='test',
            local_dir=args.local_dir,
            score_range=(args.min_score, args.max_score)
        )
    except ValueError as e:
        print(f"No test split found: {e}")
        print("Creating validation split from train (10%)...")
        
        # Use 10% of train as validation
        train_test_split = train_dataset.train_test_split(test_size=0.01, seed=42)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']
        
        # Save the new train split (90%)
        local_dir = os.path.expanduser(args.local_dir)
        train_file = os.path.join(local_dir, 'train.parquet')
        train_dataset.to_parquet(train_file)
        print(f"✓ Updated train split: {train_file} ({len(train_dataset)} samples)")
        
        # Save test split (10%)
        test_file = os.path.join(local_dir, 'test.parquet')
        test_dataset.to_parquet(test_file)
        print(f"✓ Created test split: {test_file} ({len(test_dataset)} samples)")
    
    print("\n" + "="*60)
    print("✓ Data preparation completed!")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  - {os.path.expanduser(args.local_dir)}/train.parquet ({len(train_dataset)} samples)")
    print(f"  - {os.path.expanduser(args.local_dir)}/test.parquet ({len(test_dataset)} samples)")
    print(f"\nYou can now run experiments with these data files.")

