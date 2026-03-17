#!/usr/bin/env bash
# run_linear_probe.sh  —  run linear_probe_embedding_quality.py locally on a GPU node
# Models evaluated: one_hot, base_dnabert2, random_dnabert2
#
# Usage:
#   bash run_linear_probe.sh                        # full run
#   bash run_linear_probe.sh --max_rbps 5           # quick smoke test
#   bash run_linear_probe.sh --enable_layer_search  # with layer search

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── activate venv ──────────────────────────────────────────────────────────
VENV="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/.venv/bin/activate"
if [[ -f "$VENV" ]]; then
    # shellcheck disable=SC1090
    source "$VENV"
    echo "[run_linear_probe] activated venv: $VENV"
else
    echo "[WARN] venv not found at $VENV — continuing with current environment"
fi

# ── paths ──────────────────────────────────────────────────────────────────
WS="/gpfs/bwfor/work/ws/fr_ml642-thesis_work"

TAPT_CKPT="/home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/pretrain/models/dnabert2_tapt_v3/checkpoint-2566"
FALLBACK_TOKENIZER="zhihan1996/DNABERT-2-117M"

OUTPUT_DIR="${WS}/home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/pretrain/models/dnabert2_tapt_v3/checkpoint-2566"
CACHE_DIR="${OUTPUT_DIR}/cache"

# ── run ────────────────────────────────────────────────────────────────────
python "${SCRIPT_DIR}/linear_probe_embedding_quality.py" \
    --tapt_dnabert2_model "${TAPT_CKPT}" \
    --fallback_tokenizer  "${FALLBACK_TOKENIZER}" \
    --embedding_models    tapt_dnabert2 \
    --enable_layer_search \
    --layer_search_models tapt_dnabert2 \
    --output_dir          "${OUTPUT_DIR}" \
    --cache_dir           "${CACHE_DIR}" \
    --max_length          512 \
    --batch_size          32 \
    --num_folds           5 \
    --seed                42 \
    "$@"   # forward any extra CLI args (e.g. --max_rbps 5)
