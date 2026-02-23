#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  LAMAR Fine-tuning — Local Smoke Test
#
#  Tests ALL 3 LAMAR variants on a SMALL subset:
#    1. LAMAR_512     (tapt_lamar checkpoint-98000)
#    2. LAMAR pretrained (LAMAR/weights — normal pretrained)
#    3. LAMAR random  (no pretrain weights, random init)
#
#  Each variant is tested with HPO on both datasets (Koo + CSV).
#  Duration: ~15–25 min on GPU, ~40 min on CPU
# ============================================================

# ── Conda ──────────────────────────────────────────────────────
CONDA_ENV="/home/fr/fr_fr/fr_ml642/.conda/envs/lamar_finetune"
if [[ ! -d "$CONDA_ENV" ]]; then
    CONDA_ENV="/home/fr/fr_fr/fr_ml642/.conda/envs/lamar_fixed"
fi
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

LAMAR_DIR="$THESIS/LAMAR"
FINETUNE_SCRIPTS="$LAMAR_DIR/finetune_scripts"
HPO_SCRIPT="$FINETUNE_SCRIPTS/hpo_lamar.py"
FINETUNE_SCRIPT="$FINETUNE_SCRIPTS/finetune_rbp.py"
OUTPUT_DIR="$FINETUNE_SCRIPTS/test_output"

# ── Weight paths for the 3 variants ───────────────────────────
LAMAR_512_WEIGHTS="$LAMAR_DIR/src/pretrain/saving_model/tapt_lamar/checkpoint-98000/model.safetensors"
LAMAR_PRETRAINED_WEIGHTS="$LAMAR_DIR/weights"
# LAMAR_RANDOM: no weights (omit --pretrain_path)

# Tokenizer
TOKENIZER_PATH="$LAMAR_DIR/src/pretrain/saving_model/tapt_lamar/checkpoint-100000"
if [[ ! -d "$TOKENIZER_PATH" ]]; then
    TOKENIZER_PATH="$THESIS/LAMAR/src/pretrain/saving_model/tapt_lamar/checkpoint-100000"
fi

export THESIS_ROOT="$THESIS"
export TOKENIZERS_PARALLELISM=false

RBP_KOO="U2AF1_K562_200"    # smallest koo dataset
RBP_CSV="QKI_K562_IDR"      # smallest csv dataset
RBP="$RBP_KOO"              # default for direct finetune tests

echo "═══════════════════════════════════════════════════════════"
echo "  LAMAR Fine-tuning — Smoke Test (3 variants)"
echo "  RBP koo     : $RBP_KOO"
echo "  RBP csv     : $RBP_CSV"
echo "  LAMAR_512   : $LAMAR_512_WEIGHTS"
echo "  Pretrained  : $LAMAR_PRETRAINED_WEIGHTS"
echo "  Random      : (no weights)"
echo "  Tokenizer   : $TOKENIZER_PATH"
echo "  Conda       : ${CONDA_DEFAULT_ENV:-unknown}"
echo "═══════════════════════════════════════════════════════════"

# Verify weight files exist
for wf in "$LAMAR_512_WEIGHTS" "$LAMAR_PRETRAINED_WEIGHTS"; do
    if [[ ! -e "$wf" ]]; then
        echo "WARNING: Weight file not found: $wf"
    fi
done

# Detect GPU
HAS_GPU=$(python3 -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo "0")
FP16=""
[[ "$HAS_GPU" == "1" ]] && FP16="--fp16"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

PASS=0
FAIL=0
TOTAL=0

# ═════════════════════════════════════════════════════════════
#  TEST 1: Finetune LAMAR_512 on Koo data (1 fold, 2 epochs)
# ═════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: Finetune LAMAR_512 on Koo data ────────"

python3 "$FINETUNE_SCRIPT" \
    --rbp_name "$RBP" \
    --data_path "$THESIS/data/finetune_data_koo/$RBP" \
    --output_dir "$OUTPUT_DIR/ft_lamar512_koo" \
    --pretrain_path "$LAMAR_512_WEIGHTS" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --epochs 2 --batch_size 8 --lr 3e-5 --cv_folds 1 \
    --nlabels 2 --warmup_ratio 0.05 --logging_steps 50 \
    --save_epochs 50 --early_stopping_patience 3 $FP16 \
&& { echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1)); }

# ═════════════════════════════════════════════════════════════
#  TEST 2: Finetune LAMAR pretrained on CSV data (1 fold)
# ═════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: Finetune LAMAR pretrained on CSV ──────"

python3 "$FINETUNE_SCRIPT" \
    --rbp_name "$RBP_CSV" \
    --data_path "$THESIS/DNABERT2/data/$RBP_CSV" \
    --output_dir "$OUTPUT_DIR/ft_pretrained_csv" \
    --pretrain_path "$LAMAR_PRETRAINED_WEIGHTS" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --epochs 2 --batch_size 8 --lr 3e-5 --cv_folds 1 \
    --nlabels 2 --warmup_ratio 0.05 --logging_steps 50 \
    --save_epochs 50 --early_stopping_patience 3 $FP16 \
&& { echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1)); }

# ═════════════════════════════════════════════════════════════
#  TEST 3: Finetune LAMAR random init on Koo data (1 fold)
# ═════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: Finetune LAMAR RANDOM on Koo data ─────"

python3 "$FINETUNE_SCRIPT" \
    --rbp_name "$RBP" \
    --data_path "$THESIS/data/finetune_data_koo/$RBP" \
    --output_dir "$OUTPUT_DIR/ft_random_koo" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --epochs 2 --batch_size 8 --lr 3e-5 --cv_folds 1 \
    --nlabels 2 --warmup_ratio 0.05 --logging_steps 50 \
    --save_epochs 50 --early_stopping_patience 3 $FP16 \
&& { echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1)); }

# ═════════════════════════════════════════════════════════════
#  TEST 4: HPO LAMAR_512 on Koo (2 trials, 200 samples)
# ═════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: HPO LAMAR_512 on Koo (2 trials) ───────"

python3 "$HPO_SCRIPT" \
    --rbp_name "$RBP_KOO" --dataset koo \
    --pretrain_path "$LAMAR_512_WEIGHTS" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR/hpo_lamar512_koo" \
    --n_trials 2 --max_train_samples 200 $FP16 \
&& { echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1)); }

# ═════════════════════════════════════════════════════════════
#  TEST 5: HPO LAMAR pretrained on CSV (2 trials, 200 samples)
# ═════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: HPO LAMAR pretrained on CSV (2 trials) ─"

python3 "$HPO_SCRIPT" \
    --rbp_name "$RBP_CSV" --dataset csv \
    --pretrain_path "$LAMAR_PRETRAINED_WEIGHTS" \
    --output_dir "$OUTPUT_DIR/hpo_pretrained_csv" \
    --n_trials 2 --max_train_samples 200 $FP16 \
&& { echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1)); }

# ═════════════════════════════════════════════════════════════
#  TEST 6: HPO LAMAR random on Koo (2 trials, 200 samples)
# ═════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: HPO LAMAR RANDOM on Koo (2 trials) ────"

python3 "$HPO_SCRIPT" \
    --rbp_name "$RBP_KOO" --dataset koo \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR/hpo_random_koo" \
    --n_trials 2 --max_train_samples 200 $FP16 \
&& { echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1)); }

# ═════════════════════════════════════════════════════════════
#  TEST 7: HPO LAMAR_512 on CSV (2 trials, 200 samples)
# ═════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: HPO LAMAR_512 on CSV (2 trials) ───────"

python3 "$HPO_SCRIPT" \
    --rbp_name "$RBP_CSV" --dataset csv \
    --pretrain_path "$LAMAR_512_WEIGHTS" \
    --output_dir "$OUTPUT_DIR/hpo_lamar512_csv" \
    --n_trials 2 --max_train_samples 200 $FP16 \
&& { echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1)); }

# ═════════════════════════════════════════════════════════════
#  TEST 8: HPO LAMAR random on CSV (2 trials, 200 samples)
# ═════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: HPO LAMAR RANDOM on CSV (2 trials) ────"

python3 "$HPO_SCRIPT" \
    --rbp_name "$RBP_CSV" --dataset csv \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR/hpo_random_csv" \
    --n_trials 2 --max_train_samples 200 $FP16 \
&& { echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1)); }

# ═════════════════════════════════════════════════════════════
#  TEST 9: HPO LAMAR pretrained on Koo (2 trials, 200 samples)
# ═════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: HPO LAMAR pretrained on Koo (2 trials) ─"

python3 "$HPO_SCRIPT" \
    --rbp_name "$RBP_KOO" --dataset koo \
    --pretrain_path "$LAMAR_PRETRAINED_WEIGHTS" \
    --output_dir "$OUTPUT_DIR/hpo_pretrained_koo" \
    --n_trials 2 --max_train_samples 200 $FP16 \
&& { echo "  ✓ TEST $TOTAL passed"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST $TOTAL FAILED"; FAIL=$((FAIL+1)); }

# ═════════════════════════════════════════════════════════════
#  TEST 10: Verify output artefacts
# ═════════════════════════════════════════════════════════════
TOTAL=$((TOTAL+1))
echo ""
echo "─── TEST $TOTAL: Verify output artefacts ───────────────"

ARTEFACT_OK=true

for variant in lamar512 pretrained random; do
    for ds in koo csv; do
        if [[ "$ds" == "koo" ]]; then _RBP="$RBP_KOO"; else _RBP="$RBP_CSV"; fi
        summary="$OUTPUT_DIR/hpo_${variant}_${ds}/${_RBP}_${ds}/hpo_summary.json"
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

# ═════════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════════
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
