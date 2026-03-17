#!/usr/bin/env bash
# run_linear_probe.sh  —  launch linear_probe_lamar.py with all four LAMAR variants
# Usage:  bash run_linear_probe.sh [extra args]
# Example with layer search:   bash run_linear_probe.sh --enable_layer_search
# Example quick test (5 RBPs): bash run_linear_probe.sh --max_rbps 5

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

TOKENIZER_PATH="${WS}/Thesis/LAMAR/src/pretrain/saving_model/tapt_1024_standard_collator_1gpu/checkpoint-232000"

PRETRAINED_WEIGHTS="${WS}/Thesis/LAMAR/weights"

TAPT_CHECKPOINT="${WS}/Thesis/LAMAR/src/pretrain/saving_model/tapt_1024_standard_collator/checkpoint-134000"

TAPT_STANDARD_1GPU="${WS}/Thesis/LAMAR/src/pretrain/saving_model/tapt_1024_standard_collator_1gpu/checkpoint-232000"

TAPT_CUSTOM_1GPU="${WS}/Thesis/LAMAR/src/pretrain/saving_model/tapt_1024_custom_collator_1gpu/checkpoint-232000"

TAPT_512="${WS}/Thesis/LAMAR/src/pretrain/saving_model/tapt_512_standard_collator_1gpu/checkpoint-265000}"

OUTPUT_DIR="${WS}/Thesis/LAMAR/evalEmbeddings/results/linear_probe_full_1gpu"
CACHE_DIR="${OUTPUT_DIR}/cache"

# ── run ────────────────────────────────────────────────────────────────────
python "${SCRIPT_DIR}/linear_probe_lamar.py" \
    --tokenizer_path        "${TOKENIZER_PATH}" \
    --tapt_512_checkpoint           "${TAPT_512}" \
    --output_dir  "${OUTPUT_DIR}" \
    --cache_dir   "${CACHE_DIR}" \
    --max_length  1024 \
    --batch_size  32 \
    --num_folds   5 \
    --seed        42 \
    "$@"          # forward any extra CLI args (e.g. --enable_layer_search, --max_rbps 5)
