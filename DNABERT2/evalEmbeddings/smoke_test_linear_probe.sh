#!/bin/bash
set -euo pipefail

BASE="/home/fr/fr_fr/fr_ml642/Thesis"
SCRIPT="$BASE/DNABERT2/evalEmbeddings/linear_probe_embedding_quality.py"

RUN_TAG="smoke_$(date +%Y%m%d_%H%M%S)"

if [[ "${1:-}" == "dnabert2" ]]; then
  source /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh
  conda activate /home/fr/fr_fr/fr_ml642/.conda/envs/dnabert2

  OUT_DIR="$BASE/DNABERT2/evalEmbeddings/results/linear_probe_embedding_quality/$RUN_TAG"
  CACHE_DIR="$BASE/DNABERT2/evalEmbeddings/results/linear_probe_embedding_quality/cache/$RUN_TAG"

  python "$SCRIPT" \
    --data_roots "$BASE/DNABERT2/data" "$BASE/data/finetune_data_koo" \
    --embedding_models one_hot base_dnabert2 \
    --max_rbps 1 \
    --max_samples_per_rbp 128 \
    --num_folds 2 \
    --batch_size 16 \
    --skip_plots \
    --output_dir "$OUT_DIR" \
    --cache_dir "$CACHE_DIR"

elif [[ "${1:-}" == "lamar" ]]; then
  source /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh
  conda activate /home/fr/fr_fr/fr_ml642/.conda/envs/lamar_fixed

  OUT_DIR="$BASE/LAMAR/evalEmbeddings/results/linear_probe_embedding_quality/$RUN_TAG"
  CACHE_DIR="$BASE/LAMAR/evalEmbeddings/results/linear_probe_embedding_quality/cache/$RUN_TAG"

  python "$SCRIPT" \
    --data_roots "$BASE/DNABERT2/data" "$BASE/data/finetune_data_koo" \
    --base_model "zhihan1996/DNABERT-2-117M" \
    --tapt_lamar_model "$BASE/LAMAR/src/pretrain/saving_model/tapt_lamar/checkpoint-98000" \
    --pretrained_lamar_model "$BASE/LAMAR/weights" \
    --embedding_models tapt_lamar pretrained_lamar \
    --max_rbps 1 \
    --max_samples_per_rbp 96 \
    --num_folds 2 \
    --batch_size 8 \
    --skip_plots \
    --output_dir "$OUT_DIR" \
    --cache_dir "$CACHE_DIR"

else
  echo "Usage: $0 [dnabert2|lamar]"
  exit 1
fi

echo "Smoke test finished successfully."
