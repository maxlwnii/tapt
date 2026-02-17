#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  DNABERT2 Fine-tuning — Local Smoke Test
#
#  Tests BOTH pretrained and random-init DNABERT2 fine-tuning
#  + HPO on a SMALL subset for BOTH datasets (CSV and Koo).
#
#  Models tested:
#    1. DNABERT2 pretrained  (zhihan1996/DNABERT-2-117M)
#    2. DNABERT2 random init (same architecture, random weights)
#
#  Duration: ~10–15 min on GPU, ~30 min on CPU
# ============================================================

# ── Conda ──────────────────────────────────────────────────────
CONDA_ENV="/home/fr/fr_fr/fr_ml642/.conda/envs/dnabert2"
if [[ -d "$CONDA_ENV" ]]; then
    source "$(conda info --base 2>/dev/null || echo /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

# ── Paths ──────────────────────────────────────────────────────
if [[ -d "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis" ]]; then
    THESIS="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"
elif [[ -d "/home/fr/fr_fr/fr_ml642/Thesis" ]]; then
    THESIS="/home/fr/fr_fr/fr_ml642/Thesis"
else
    echo "ERROR: Cannot find Thesis directory." >&2; exit 1
fi

FINETUNE_DIR="$THESIS/DNABERT2/finetune"
HPO_SCRIPT="$FINETUNE_DIR/hpo_dnabert2.py"
TRAIN_SCRIPT="$THESIS/DNABERT2/train.py"
OUTPUT_DIR="$FINETUNE_DIR/test_output"

export TOKENIZERS_PARALLELISM=false
export THESIS_ROOT="$THESIS"

RBP="U2AF1_K562_200"  # smallest dataset (~2.3k train)

echo "═══════════════════════════════════════════════════════════"
echo "  DNABERT2 Fine-tuning — Smoke Test (pretrained + random)"
echo "  RBP     : $RBP"
echo "  THESIS  : $THESIS"
echo "  Conda   : ${CONDA_DEFAULT_ENV:-unknown}"
echo "═══════════════════════════════════════════════════════════"

# Detect GPU
HAS_GPU=$(python3 -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo "0")
FP16=""
[[ "$HAS_GPU" == "1" ]] && FP16="--fp16"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

PASS=0
FAIL=0
TOTAL=0

# ═══════════════════════════════════════════════════════════════
#  TEST 1: Fine-tune PRETRAINED on CSV (train.py)
# ═══════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: Fine-tune PRETRAINED on CSV (train.py) ─"

python3 "$TRAIN_SCRIPT" \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --data_path "$THESIS/data/finetune_data_koo/$RBP" \
    --kmer -1 \
    --run_name test_pretrained_csv_${RBP} \
    --model_max_length 25 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --early_stopping_patience 3 \
    $FP16 \
    --save_steps 200 --eval_steps 200 \
    --output_dir "$OUTPUT_DIR/pretrained_csv_finetune" \
    --warmup_steps 50 --logging_steps 50 \
    --overwrite_output_dir --log_level info \
    --save_model --eval_and_save_results \
&& { echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1)); }

# ═══════════════════════════════════════════════════════════════
#  TEST 2: Fine-tune RANDOM INIT on CSV (train.py)
# ═══════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: Fine-tune RANDOM INIT on CSV (train.py) "

python3 "$TRAIN_SCRIPT" \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --use_random_init \
    --data_path "$THESIS/data/finetune_data_koo/$RBP" \
    --kmer -1 \
    --run_name test_random_csv_${RBP} \
    --model_max_length 25 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --early_stopping_patience 3 \
    $FP16 \
    --save_steps 200 --eval_steps 200 \
    --output_dir "$OUTPUT_DIR/random_csv_finetune" \
    --warmup_steps 50 --logging_steps 50 \
    --overwrite_output_dir --log_level info \
    --save_model --eval_and_save_results \
&& { echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1)); }

# ═══════════════════════════════════════════════════════════════
#  TEST 3: HPO PRETRAINED on CSV (2 trials)
# ═══════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: HPO PRETRAINED on CSV (2 trials) ───────"

python3 "$HPO_SCRIPT" \
    --rbp_name "$RBP" --dataset csv \
    --output_dir "$OUTPUT_DIR/hpo_pretrained_csv" \
    --n_trials 2 --max_train_samples 200 $FP16 \
&& { echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1)); }

# ═══════════════════════════════════════════════════════════════
#  TEST 4: HPO RANDOM INIT on CSV (2 trials)
# ═══════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: HPO RANDOM INIT on CSV (2 trials) ──────"

python3 "$HPO_SCRIPT" \
    --rbp_name "$RBP" --dataset csv --use_random_init \
    --output_dir "$OUTPUT_DIR/hpo_random_csv" \
    --n_trials 2 --max_train_samples 200 $FP16 \
&& { echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1)); }

# ═══════════════════════════════════════════════════════════════
#  TEST 5: HPO PRETRAINED on Koo (2 trials)
# ═══════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: HPO PRETRAINED on Koo (2 trials) ───────"

python3 "$HPO_SCRIPT" \
    --rbp_name "$RBP" --dataset koo \
    --output_dir "$OUTPUT_DIR/hpo_pretrained_koo" \
    --n_trials 2 --max_train_samples 200 $FP16 \
&& { echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1)); }

# ═══════════════════════════════════════════════════════════════
#  TEST 6: HPO RANDOM INIT on Koo (2 trials)
# ═══════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: HPO RANDOM INIT on Koo (2 trials) ──────"

python3 "$HPO_SCRIPT" \
    --rbp_name "$RBP" --dataset koo --use_random_init \
    --output_dir "$OUTPUT_DIR/hpo_random_koo" \
    --n_trials 2 --max_train_samples 200 $FP16 \
&& { echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1)); }

# ═══════════════════════════════════════════════════════════════
#  TEST 7: Verify output artefacts
# ═══════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: Verify output artefacts ────────────────"

ARTEFACT_OK=true

for variant in pretrained random; do
    ft_dir="$OUTPUT_DIR/${variant}_csv_finetune/results"
    if [[ -d "$ft_dir" ]]; then
        echo "  ✓ ${variant} CSV fine-tune results exist"
    else
        echo "  ✗ Missing ${variant} CSV fine-tune results"; ARTEFACT_OK=false
    fi
done

for variant in pretrained random; do
    for ds in csv koo; do
        summary="$OUTPUT_DIR/hpo_${variant}_${ds}/${RBP}_${ds}/hpo_summary.json"
        if [[ -f "$summary" ]]; then
            echo "  ✓ HPO ${variant} ${ds} summary exists"
            python3 -c "import json; d=json.load(open('$summary')); print(f'    Best AUC: {d[\"best_value\"]:.4f}')"
        else
            echo "  ✗ Missing HPO ${variant} ${ds} summary"; ARTEFACT_OK=false
        fi
    done
done

if $ARTEFACT_OK; then
    echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1))
else
    echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1))
fi

# ═══════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  RESULTS:  $PASS passed,  $FAIL failed  (out of $TOTAL)"
echo "  Output:   $OUTPUT_DIR"
if [[ $FAIL -eq 0 ]]; then
    echo "  ✓ ALL TESTS PASSED"
else
    echo "  ✗ $FAIL TEST(S) FAILED"
    exit 1
fi
echo "═══════════════════════════════════════════════════════════"
