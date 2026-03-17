#!/bin/bash
# ──────────────────────────────────────────────────────────────
# test_local.sh  —  smoke-test finetune_cross.py before submitting
# batch jobs. Runs ONE pair per model×variant×experiment with
# --max_train_samples 32 so each test finishes in <2 min on GPU.
#
# Environment notes:
#   DNABERT2 → Thesis/.venv  (activated automatically below)
#   LAMAR    → conda env lamar_fixed  (activated automatically below)
#
# Usage:
#   bash test_local.sh              # test both LAMAR + DNABERT2
#   bash test_local.sh lamar        # test LAMAR only
#   bash test_local.sh dnabert2     # test DNABERT2 only
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export THESIS_ROOT="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"
export TOKENIZERS_PARALLELISM=false

# Test pair names (first in each list — small, representative)
CROSS_CELL_PAIR="AGGF1_train_K562_test_HepG2_fixlen_101"
CROSS_LENGTH_PAIR="AATF_K562_ENCSR819XBT"

LAMAR_DIR="$THESIS_ROOT/LAMAR"
PRETRAIN_PATH="$LAMAR_DIR/weights"

TEST_DIR="$SCRIPT_DIR/test_output"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

MODEL_FILTER="${1:-all}"

PASS=0
FAIL=0

_print_result() {
    local json_path="$1"
    python3 -c "
import json, sys
d = json.load(open('$json_path'))
print(f\"    val_auc  : {d['val_metrics'].get('eval_auc', 'N/A')}\")
print(f\"    test_auc : {d['test_metrics'].get('eval_auc', 'N/A')}\")
" 2>/dev/null || echo "    (could not parse results.json)"
}

# ── LAMAR tests ─────────────────────────────────────────────
if [[ "$MODEL_FILTER" == "all" ]] || [[ "$MODEL_FILTER" == "lamar" ]]; then
    # Activate LAMAR conda env
    source /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh
    conda activate lamar_finetune 2>/dev/null || conda activate lamar_fixed
    LAMAR_PYTHON=$(which python)

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  [1/4] LAMAR pretrained × cross_cell"
    echo "════════════════════════════════════════════════════════"
    if $LAMAR_PYTHON finetune_cross.py \
        --model_type lamar \
        --experiment cross_cell \
        --pair_name "$CROSS_CELL_PAIR" \
        --pretrain_path "$PRETRAIN_PATH" \
        --max_length 1024 \
        --max_train_samples 32 \
        --output_dir "$TEST_DIR/lamar_pretrained_cc" \
        --seed 42; then
        echo "  ✓ PASSED"
        _print_result "$TEST_DIR/lamar_pretrained_cc/$CROSS_CELL_PAIR/results.json"
        PASS=$((PASS+1))
    else
        echo "  ✗ FAILED"; FAIL=$((FAIL+1))
    fi

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  [2/4] LAMAR random × cross_length"
    echo "════════════════════════════════════════════════════════"
    if $LAMAR_PYTHON finetune_cross.py \
        --model_type lamar \
        --experiment cross_length \
        --pair_name "$CROSS_LENGTH_PAIR" \
        --max_length 1024 \
        --max_train_samples 32 \
        --output_dir "$TEST_DIR/lamar_random_cl" \
        --seed 42; then
        echo "  ✓ PASSED"
        _print_result "$TEST_DIR/lamar_random_cl/$CROSS_LENGTH_PAIR/results.json"
        PASS=$((PASS+1))
    else
        echo "  ✗ FAILED"; FAIL=$((FAIL+1))
    fi
fi

# ── DNABERT2 tests ───────────────────────────────────────────
if [[ "$MODEL_FILTER" == "all" ]] || [[ "$MODEL_FILTER" == "dnabert2" ]]; then
    # Activate .venv
    VENV="$THESIS_ROOT/.venv/bin/activate"
    if [[ -f "$VENV" ]]; then
        # shellcheck disable=SC1090
        source "$VENV"
        echo "[test_local] activated venv: $VENV"
    else
        echo "[WARN] venv not found at $VENV — using current environment"
    fi

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  [3/4] DNABERT2 pretrained × cross_cell"
    echo "════════════════════════════════════════════════════════"
    if python finetune_cross.py \
        --model_type dnabert2 \
        --experiment cross_cell \
        --pair_name "$CROSS_CELL_PAIR" \
        --max_length 128 \
        --max_train_samples 32 \
        --output_dir "$TEST_DIR/dnabert2_pretrained_cc" \
        --seed 42; then
        echo "  ✓ PASSED"
        _print_result "$TEST_DIR/dnabert2_pretrained_cc/$CROSS_CELL_PAIR/results.json"
        PASS=$((PASS+1))
    else
        echo "  ✗ FAILED"; FAIL=$((FAIL+1))
    fi

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  [4/4] DNABERT2 random × cross_length"
    echo "════════════════════════════════════════════════════════"
    if python finetune_cross.py \
        --model_type dnabert2 \
        --experiment cross_length \
        --pair_name "$CROSS_LENGTH_PAIR" \
        --use_random_init \
        --max_length 256 \
        --max_train_samples 32 \
        --output_dir "$TEST_DIR/dnabert2_random_cl" \
        --seed 42; then
        echo "  ✓ PASSED"
        _print_result "$TEST_DIR/dnabert2_random_cl/$CROSS_LENGTH_PAIR/results.json"
        PASS=$((PASS+1))
    else
        echo "  ✗ FAILED"; FAIL=$((FAIL+1))
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Smoke test complete: $PASS passed, $FAIL failed"
echo "  Output: $TEST_DIR"
echo "════════════════════════════════════════════════════════"
[[ $FAIL -eq 0 ]]
