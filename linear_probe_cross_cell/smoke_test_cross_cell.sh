#!/bin/bash
# smoke_test_cross_cell.sh
# Quick local test: 2 LAMAR + 1 DNABERT-2 + one_hot on 3 tasks, no plots.
set -euo pipefail

BASE="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"
SCRIPT="$BASE/linear_probe_cross_cell/linear_probe_cross_cell.py"

source "$BASE/.venv/bin/activate"

export TOKENIZERS_PARALLELISM=false

python "$SCRIPT" \
  --models lamar_pretrained dnabert2_pretrained one_hot \
  --max_rbps 3 \
  --max_samples_per_rbp 300 \
  --batch_size 16 \
  --num_folds 3 \
  --output_dir "$BASE/linear_probe_cross_cell/results/smoke_test" \
  --cache_dir  "$BASE/linear_probe_cross_cell/results/smoke_test/cache" \
  --skip_plots

echo "Smoke test passed."
