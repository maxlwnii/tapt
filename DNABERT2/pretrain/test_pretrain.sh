#!/usr/bin/env bash
# test_pretrain.sh — smoke-test pretrain_dnabert2_v3.py
#
# Uses a tiny subset of real data (100 train / 20 val, 1 epoch) on CPU or
# whatever GPU is available.  Completes in a few minutes.
#
# Usage:  bash test_pretrain.sh [--fp16 | --bf16]
#
# Pass --gpu_only to skip the test when no CUDA device is detected.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"

# ── activate venv if available ─────────────────────────────────────────────────
VENV="$BASE/.venv/bin/activate"
[[ -f "$VENV" ]] && source "$VENV"

PREPROCESS="$SCRIPT_DIR/preprocess_val_db2.py"
PRETRAIN="$SCRIPT_DIR/pretrain_dnabert2_v3.py"

TRAIN_FILE="$BASE/preprocess/preprocessed_data_metadata_train.json"
VAL_RAW="$BASE/data/eval_db2.txt"
VAL_JSON="/tmp/val_db2_test.json"
OUT_DIR="/tmp/dnabert2_tapt_v3_test"

# ── 1. create tiny val JSON ────────────────────────────────────────────────────
echo "════════════════════════════════════════════"
echo "  Step 1: preprocess val data (20 windows)"
echo "════════════════════════════════════════════"
python "$PREPROCESS" \
    --input       "$VAL_RAW" \
    --output      "$VAL_JSON" \
    --window      512 \
    --max_n       0.1 \
    --max_samples 20

echo "  val JSON: $VAL_JSON"

# ── 2. check CUDA availability ─────────────────────────────────────────────────
CUDA_AVAILABLE=$(python -c "import torch; print(int(torch.cuda.is_available()))")
echo ""
if [[ "$CUDA_AVAILABLE" == "1" ]]; then
    echo "CUDA detected — running test on GPU"
    EXTRA_FLAGS="--fp16"
else
    echo "No CUDA — running test on CPU (slow but functional)"
    EXTRA_FLAGS=""
fi

# ── 3. run tiny pretraining ────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  Step 2: pretrain (100 train / 20 val / 1 epoch)"
echo "════════════════════════════════════════════"

python "$PRETRAIN" \
    --model            zhihan1996/DNABERT-2-117M \
    --max_length       512 \
    --train_file       "$TRAIN_FILE" \
    --val_file         "$VAL_JSON" \
    --max_train        100 \
    --max_val          20 \
    --num_workers      1 \
    --mlm_prob         0.15 \
    --output_dir       "$OUT_DIR" \
    --epochs           1 \
    --batch_size       4 \
    --eval_batch       4 \
    --accum            1 \
    --lr               5e-5 \
    --patience         1 \
    --save_total       1 \
    --logging_steps    10 \
    --seed             42 \
    --report_to        none \
    $EXTRA_FLAGS

# ── 4. verify output ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  Step 3: verify outputs"
echo "════════════════════════════════════════════"
python - <<'EOF'
import os, json

out = "/tmp/dnabert2_tapt_v3_test"
required = ["config.json", "tokenizer_config.json", "train_results.json"]
missing = [f for f in required if not os.path.exists(os.path.join(out, f))]

if missing:
    print(f"FAIL — missing files: {missing}")
    exit(1)

with open(os.path.join(out, "train_results.json")) as f:
    r = json.load(f)
print(f"  train_loss  : {r.get('train_loss', 'N/A')}")

eval_file = os.path.join(out, "eval_results.json")
if os.path.exists(eval_file):
    with open(eval_file) as f:
        e = json.load(f)
    print(f"  eval_loss   : {e.get('eval_loss', 'N/A')}")
    print(f"  perplexity  : {e.get('perplexity', 'N/A')}")

print("  PASSED ✓")
EOF

echo ""
echo "All pretraining smoke-tests passed."
echo "Output directory: $OUT_DIR"
