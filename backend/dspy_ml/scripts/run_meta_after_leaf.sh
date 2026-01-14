#!/bin/bash
# Wait for leaf clustering to complete, then run meta clustering

cd "$(dirname "$0")/../.."

echo "Waiting for leaf clustering to complete..."
while [ ! -f dspy_ml/data/guideline_tree_v2.json ]; do
    echo "  Still waiting... ($(date))"
    sleep 30
done

echo "✓ Leaf clustering completed! Starting meta clustering..."

source .venv/bin/activate

nohup python3 dspy_ml/scripts/cluster_guideline_tree_meta.py \
    --dataset dspy_ml/data/drills_dataset_v2.json \
    --leaf-tree dspy_ml/data/guideline_tree_v2.json \
    --output-tree dspy_ml/data/guideline_tree_meta_v2.json \
    --output-best dspy_ml/data/best_guidelines_meta_v2.json \
    --log dspy_ml/data/guideline_tree_meta_v5.log \
    > dspy_ml/data/guideline_tree_meta_v5_stdout.txt 2>&1 &

echo "Meta clustering started in background (PID: $!)"
echo "Monitor progress with: tail -f dspy_ml/data/guideline_tree_meta_v5.log"

