#!/bin/bash

# Download all required datasets from HuggingFace

cd /home/chenluy/mismatch_all_perturb_agent/eval

echo "Downloading datasets from HuggingFace..."

python3 download_datasets.py \
  --datasets \
    "weqweasdas/math500" \
    "weqweasdas/minerva_math" \
    "weqweasdas/olympiadbench" \
    "Chenlu123/hmmt0225" \
  --output_dir /home/chenluy/mismatch_all_perturb_agent/datasets \
  --split train

echo ""
echo "Dataset download complete!"
echo "Files saved to: /home/chenluy/mismatch_all_perturb_agent/datasets"
echo ""
echo "Directory structure:"
ls -lh /home/chenluy/mismatch_all_perturb_agent/datasets/weqweasdas/ 2>/dev/null || echo "weqweasdas directory not found"
ls -lh /home/chenluy/mismatch_all_perturb_agent/datasets/Chenlu123/ 2>/dev/null || echo "Chenlu123 directory not found"

