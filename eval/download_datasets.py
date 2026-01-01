#!/usr/bin/env python3
"""
Download HuggingFace datasets and save them as parquet files
"""
import os
import argparse
from pathlib import Path
from datasets import load_dataset


def download_and_save_dataset(dataset_name, output_dir, split="train"):
    """
    Download a dataset from HuggingFace and save as parquet
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., 'weqweasdas/math500')
        output_dir: Output directory to save parquet files
        split: Dataset split to download (default: 'test')
    """
    print(f"Downloading dataset: {dataset_name}")
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_name, split=split)
        
        # Create output directory matching the dataset name structure
        # e.g., weqweasdas/math500 -> output_dir/weqweasdas/
        if "/" in dataset_name:
            org, name = dataset_name.split("/")
            dataset_output_dir = Path(output_dir) / org
        else:
            dataset_output_dir = Path(output_dir)
            name = dataset_name
        
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet
        output_path = dataset_output_dir / f"{name}.parquet"
        dataset.to_parquet(str(output_path))
        
        print(f"✓ Saved to: {output_path}")
        print(f"  Number of samples: {len(dataset)}")
        
        return output_path
        
    except Exception as e:
        print(f"✗ Error downloading {dataset_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets to local parquet files")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of HuggingFace dataset names"
    )
    parser.add_argument(
        "--output_dir",
        default="/home/chenluy/SimpleTIR/datasets",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download (default: test)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Downloading HuggingFace Datasets")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Split: {args.split}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print()
    
    success_count = 0
    failed_datasets = []
    
    for dataset_name in args.datasets:
        result = download_and_save_dataset(
            dataset_name,
            args.output_dir,
            split=args.split
        )
        if result:
            success_count += 1
        else:
            failed_datasets.append(dataset_name)
        print()
    
    print("=" * 60)
    print(f"Download complete: {success_count}/{len(args.datasets)} successful")
    if failed_datasets:
        print(f"Failed datasets: {', '.join(failed_datasets)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

