#!/bin/bash
# Layer Search Workflow - Complete Example
# This script demonstrates the recommended workflow for layer search and evaluation
# Usage: bash run_layer_search_workflow.sh

set -euo pipefail

BASE="/home/fr/fr_fr/fr_ml642/Thesis"
EVALDIR="$BASE/DNABERT2/evalEmbeddings"

echo "========================================"
echo "Layer Search Workflow"
echo "========================================"
echo ""

# Step 1: Run layer search to find the best layer
echo "STEP 1: Running layer search on TAPT LAMAR..."
echo "  This will probe all transformer layers and identify the best one."
echo "  Submitting job: run_layer_search.slurm"
echo ""

LAYER_SEARCH_JOB=$(sbatch "$EVALDIR/run_layer_search.slurm" | awk '{print $4}')
echo "  Layer search job submitted: SLURM_JOB_ID=$LAYER_SEARCH_JOB"
echo "  Waiting for completion..."
echo ""

srun --dependency=afterok:$LAYER_SEARCH_JOB echo "✓ Layer search complete!"

# Find the output directory of the layer search job
LAYER_SEARCH_OUT=$(ls -td "$EVALDIR/results/linear_probe_embedding_quality"/*layer_search* 2>/dev/null | head -1)
if [ -z "$LAYER_SEARCH_OUT" ]; then
    echo "ERROR: Could not find layer_search output directory"
    exit 1
fi

echo "  Results saved to: $LAYER_SEARCH_OUT"
echo ""

# Extract the best layer from results
if [ -f "$LAYER_SEARCH_OUT/layer_search.json" ]; then
    BEST_LAYER=$(grep -o '"best_layer": [0-9]*' "$LAYER_SEARCH_OUT/layer_search.json" | grep -o '[0-9]*')
    BEST_AUROC=$(grep -o '"best_auroc": [0-9.]*' "$LAYER_SEARCH_OUT/layer_search.json" | grep -o '[0-9.]*')
    echo "  ✓ Best layer found: $BEST_LAYER with AUROC = $BEST_AUROC"
else
    echo "ERROR: layer_search.json not found"
    exit 1
fi

echo ""
echo "========================================"
echo "STEP 2: Run full evaluation with best layer"
echo "========================================"
echo ""
echo "  Edit run_full_evaluation_with_best_layer.slurm:"
echo "    Change: BEST_LAYER=6"
echo "    To:     BEST_LAYER=$BEST_LAYER"
echo ""
echo "  Then submit:"
echo "    sbatch $EVALDIR/run_full_evaluation_with_best_layer.slurm"
echo ""
echo "========================================"
echo ""
echo "To automate this, uncomment the following lines:"
echo ""
cat << 'EOF'
# Automatically update BEST_LAYER in the full evaluation script
sed -i "s/BEST_LAYER=.*/BEST_LAYER=$BEST_LAYER/" "$EVALDIR/run_full_evaluation_with_best_layer.slurm"

# Submit the full evaluation job
EVAL_JOB=$(sbatch "$EVALDIR/run_full_evaluation_with_best_layer.slurm" | awk '{print $4}')
echo "  Full evaluation job submitted: SLURM_JOB_ID=$EVAL_JOB"

# Wait for completion
srun --dependency=afterok:$EVAL_JOB echo "✓ Full evaluation complete!"

echo ""
echo "All jobs complete! Check results in:"
echo "  $EVALDIR/results/linear_probe_embedding_quality/"
EOF

echo ""
echo "Layer search workflow information:"
echo "  - Layer search results: $LAYER_SEARCH_OUT/layer_search.json"
echo "  - Layer search plot: $LAYER_SEARCH_OUT/layer_auroc_curve.png"
echo "  - Best layer: $BEST_LAYER"
