#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_dnabert2_local.sh  —  run DNABERT2 finetuning locally on a GPU node
#
# Loops over ALL pairs for the given VARIANT × EXPERIMENT and skips any pair
# whose results.json already exists (safe to re-run / resume).
#
# Usage:
#   bash run_dnabert2_local.sh <variant> <experiment>
#
#   variant    : pretrained | random
#   experiment : cross_cell | cross_length
#
# Examples:
#   bash run_dnabert2_local.sh pretrained cross_cell
#   bash run_dnabert2_local.sh random     cross_length
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

VARIANT="${1:-}"
EXPERIMENT="${2:-}"

if [[ -z "$VARIANT" ]] || [[ -z "$EXPERIMENT" ]]; then
    echo "Usage: $0 <variant> <experiment>"
    echo "  variant    : pretrained | random"
    echo "  experiment : cross_cell | cross_length"
    exit 1
fi

if [[ "$VARIANT" != "pretrained" ]] && [[ "$VARIANT" != "random" ]]; then
    echo "ERROR: variant must be 'pretrained' or 'random', got: $VARIANT"
    exit 1
fi

if [[ "$EXPERIMENT" != "cross_cell" ]] && [[ "$EXPERIMENT" != "cross_length" ]]; then
    echo "ERROR: experiment must be 'cross_cell' or 'cross_length', got: $EXPERIMENT"
    exit 1
fi

# ── Activate venv ──────────────────────────────────────────────────────────────
VENV="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/.venv/bin/activate"
if [[ -f "$VENV" ]]; then
    # shellcheck disable=SC1090
    source "$VENV"
    echo "[run_dnabert2_local] activated venv: $VENV"
else
    echo "[WARN] venv not found at $VENV — continuing with current environment"
fi

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export THESIS_ROOT="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"
export TOKENIZERS_PARALLELISM=false
cd "$SCRIPT_DIR"

# ── Resolve pair list & settings per experiment ────────────────────────────────
if [[ "$EXPERIMENT" == "cross_cell" ]]; then
    PAIRS_JSON="$THESIS_ROOT/data/cross_cell/valid_pairs.json"
    MAX_LENGTH=128
elif [[ "$EXPERIMENT" == "cross_length" ]]; then
    PAIRS_JSON="$THESIS_ROOT/data/cross_length/valid_prefixes.json"
    MAX_LENGTH=256
fi

if [[ ! -f "$PAIRS_JSON" ]]; then
    echo "ERROR: Pair list not found: $PAIRS_JSON"
    exit 1
fi

# ── Resolve random init flag ───────────────────────────────────────────────────
RANDOM_ARG=""
if [[ "$VARIANT" == "random" ]]; then
    RANDOM_ARG="--use_random_init"
fi

OUTPUT_DIR="$SCRIPT_DIR/results/${EXPERIMENT}/dnabert2_${VARIANT}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

# ── Load pair list ─────────────────────────────────────────────────────────────
mapfile -t PAIRS < <(python3 -c "import json; [print(p) for p in json.load(open('$PAIRS_JSON'))]")
N_TOTAL="${#PAIRS[@]}"

echo "════════════════════════════════════════════════════════════"
echo "  DNABERT2 local finetuning"
echo "  variant    : $VARIANT"
echo "  experiment : $EXPERIMENT"
echo "  max_length : $MAX_LENGTH"
echo "  pairs      : $N_TOTAL"
echo "  output_dir : $OUTPUT_DIR"
echo "  python     : $(which python)"
echo "════════════════════════════════════════════════════════════"

# ── CUDA sanity check ──────────────────────────────────────────────────────────
python -c "
import torch
print(f'  CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU            : {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""

# ── Main loop ──────────────────────────────────────────────────────────────────
DONE=0
SKIPPED=0
FAILED=0

for PAIR_NAME in "${PAIRS[@]}"; do
    RESULTS_FILE="$OUTPUT_DIR/$PAIR_NAME/results.json"

    if [[ -f "$RESULTS_FILE" ]]; then
        echo "[SKIP  $((DONE+SKIPPED+FAILED+1))/$N_TOTAL] $PAIR_NAME  (already done)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[START $((DONE+SKIPPED+FAILED+1))/$N_TOTAL] $PAIR_NAME"

    if python finetune_cross.py \
        --model_type  dnabert2 \
        --experiment  "$EXPERIMENT" \
        --pair_name   "$PAIR_NAME" \
        $RANDOM_ARG \
        --max_length  "$MAX_LENGTH" \
        --output_dir  "$OUTPUT_DIR" \
        --seed        42 \
        2>&1 | tee "$SCRIPT_DIR/logs/dnabert2_${VARIANT}_${EXPERIMENT}_${PAIR_NAME}.log"; then
        DONE=$((DONE + 1))
        AUC=$(python3 -c "
import json
d = json.load(open('$RESULTS_FILE'))
print(f\"{d['test_metrics'].get('eval_auc', 'N/A'):.4f}\")
" 2>/dev/null || echo "N/A")
        echo "[DONE  $((DONE+SKIPPED+FAILED))/$N_TOTAL] $PAIR_NAME  test_auc=$AUC"
    else
        FAILED=$((FAILED + 1))
        echo "[FAIL  $((DONE+SKIPPED+FAILED))/$N_TOTAL] $PAIR_NAME"
    fi
    echo ""
done

echo "════════════════════════════════════════════════════════════"
echo "  Finished: $DONE done, $SKIPPED skipped, $FAILED failed"
echo "  Results : $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"
