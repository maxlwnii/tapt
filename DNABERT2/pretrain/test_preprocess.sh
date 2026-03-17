#!/usr/bin/env bash
# test_preprocess.sh — smoke-test preprocess_val_db2.py
#
# Runs in dry-run mode (--test) and then writes a tiny output to /tmp.
# No GPU / conda environment required; uses whatever python is active.
#
# Usage:  bash test_preprocess.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"

VENV="$BASE/.venv/bin/activate"
[[ -f "$VENV" ]] && source "$VENV"

PREPROCESS="$SCRIPT_DIR/preprocess_val_db2.py"
INPUT="$BASE/data/eval_db2.txt"
OUT_TMP="/tmp/val_db2_test.json"

echo "════════════════════════════════════════"
echo "  1. dry-run (--test, no file written)"
echo "════════════════════════════════════════"
python "$PREPROCESS" \
    --input   "$INPUT" \
    --output  "$OUT_TMP" \
    --window  512 \
    --max_n   0.1 \
    --test

echo ""
echo "═══════════════════════════════════════════"
echo "  2. write output (capped at 20 samples)"
echo "═══════════════════════════════════════════"
python "$PREPROCESS" \
    --input       "$INPUT" \
    --output      "$OUT_TMP" \
    --window      512 \
    --max_n       0.1 \
    --max_samples 20

echo ""
echo "  Verifying output …"
python - <<'EOF'
import json, sys
with open("/tmp/val_db2_test.json") as f:
    data = json.load(f)
print(f"  records    : {len(data)}")
print(f"  keys       : {list(data[0].keys())}")
print(f"  seq_len    : {data[0]['seq_len']}")
print(f"  first seq  : {data[0]['sequence'][:40]}…")
assert all(len(r["sequence"]) == 512 for r in data), "FAIL: not all windows are 512!"
assert all(r["seq_len"] == 512 for r in data), "FAIL: seq_len mismatch!"
print("  PASSED ✓")
EOF

echo ""
echo "All preprocessing tests passed."
