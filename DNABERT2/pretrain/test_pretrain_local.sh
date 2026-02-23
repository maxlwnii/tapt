#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  DNABERT2 TAPT — Comprehensive Local Smoke Test
#
#  Tests EVERY parameter used in the real SLURM pretraining
#  scripts (pretrain_dnabert2_full.slurm and
#  pretrain_dnabert2_standard_mlm.slurm) on a tiny data
#  subset so failures surface *before* a 48-hour A100 job.
#
#  Duration: ~5–10 minutes on a consumer GPU, ~15 min on CPU.
# ============================================================

# ── Conda environment (match SLURM scripts) ──────────────────
# Activate the dnabert2 env.  On HPC nodes the conda init may
# not be sourced automatically, so we do it explicitly.
CONDA_ENV="/home/fr/fr_fr/fr_ml642/.conda/envs/dnabert2"
if [[ -d "$CONDA_ENV" ]]; then
    if command -v conda &>/dev/null; then
        echo "Activating conda env: dnabert2"
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
        conda activate "$CONDA_ENV"
    else
        # fallback: try the standard miniforge path
        source /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh 2>/dev/null || true
        conda activate "$CONDA_ENV"
    fi
fi

# ── Paths (adjust to your machine) ────────────────────────────
if [[ -d "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis" ]]; then
    BASE="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"
elif [[ -d "/home/fr/fr_fr/fr_ml642/Thesis" ]]; then
    BASE="/home/fr/fr_fr/fr_ml642/Thesis"
else
    echo "ERROR: Cannot find Thesis directory. Set BASE manually." >&2
    exit 1
fi

SCRIPT="$BASE/DNABERT2/pretrain/pretrain_dnabert2.py"
TRAIN_FILE="$BASE/preprocess/preprocessed_data_metadata_train.json"
VAL_FILE="$BASE/preprocess/preprocessed_data_metadata_val.json"
OUTPUT_DIR="$BASE/DNABERT2/pretrain/models/test_local"

export PYTHONPATH="$BASE:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

echo "═══════════════════════════════════════════════════════════"
echo "  DNABERT2 TAPT — Comprehensive smoke test"
echo "  BASE   : $BASE"
echo "  Python : $(which python3)"
echo "  Conda  : ${CONDA_DEFAULT_ENV:-unknown}"
echo "═══════════════════════════════════════════════════════════"

# ── Verify files exist ────────────────────────────────────────
for f in "$SCRIPT" "$TRAIN_FILE" "$VAL_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing file: $f" >&2
        exit 1
    fi
done

# ── Python & package sanity ───────────────────────────────────
echo ""
echo "─── Package versions ─────────────────────────────────────"
python3 -c "
import sys
print(f'  Python       : {sys.version}')
import torch; print(f'  PyTorch      : {torch.__version__}')
import transformers; print(f'  transformers : {transformers.__version__}')
import datasets; print(f'  datasets     : {datasets.__version__}')
"

# ── CUDA info (mirrors SLURM sanity check exactly) ────────────
echo ""
echo "─── CUDA sanity check ────────────────────────────────────"
python3 -c "
import torch
print(f'  CUDA available : {torch.cuda.is_available()}')
print(f'  GPU count      : {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'  GPU name       : {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  Running on CPU (expect slower execution)')
" 2>/dev/null || echo "  (could not detect GPU)"

# ── Detect GPU for fp16 ──────────────────────────────────────
HAS_GPU=$(python3 -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo "0")
FP16_FLAG=""
if [[ "$HAS_GPU" == "1" ]]; then
    FP16_FLAG="--fp16"
fi

# ── Clean previous test output ────────────────────────────────
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# ─────────────────────────────────────────────────────────────
#  NOTE ON PARAMETER MAPPING
#
#  Every real SLURM flag is replicated below.  The ONLY
#  intentional differences for the smoke test are:
#
#   Real SLURM              Smoke test
#   ─────────────────────── ──────────────────────────────
#   --num_train_epochs 10   --max_steps 20
#   --per_device_*_bs 32/64 --per_device_*_bs 4
#   --grad_accum_steps 2    --gradient_accumulation_steps 1
#   (no sample limit)       --max_train_samples 200
#                           --max_eval_samples 50
#   --eval_strategy epoch   --eval_strategy steps (+ eval/save_steps 10)
#   --logging_steps 50      --logging_steps 5
#   --save_total_limit 5    --save_total_limit 2
#   --report_to tensorboard --report_to none
#   --dataloader_num_w 4    --dataloader_num_workers 2
#
#  All other parameters are IDENTICAL to the real jobs.
# ─────────────────────────────────────────────────────────────

PASS=0
FAIL=0

# ═══════════════════════════════════════════════════════════════
#  TEST 1: Adaptive masking — ALL real flags
#          (mirrors pretrain_dnabert2_full.slurm)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "─── TEST 1: Adaptive masking (full param mirror) ─────────"
echo ""

python3 "$SCRIPT" \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --max_seq_length 512 \
    \
    --train_file "$TRAIN_FILE" \
    --validation_file "$VAL_FILE" \
    --max_train_samples 200 \
    --max_eval_samples 50 \
    --preprocessing_num_workers 2 \
    \
    --use_adaptive_masking \
    --target_mlm_prob 0.15 \
    --eclip_mlm_lo 0.20 \
    --eclip_mlm_hi 0.25 \
    --min_flanking_prob 0.05 \
    \
    --output_dir "${OUTPUT_DIR}/adaptive" \
    --max_steps 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --optim adamw_torch \
    \
    --early_stopping_patience 3 \
    \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_steps 10 \
    --logging_steps 5 \
    --save_total_limit 2 \
    \
    $FP16_FLAG \
    --gradient_checkpointing \
    --dataloader_num_workers 2 \
    --dataloader_pin_memory \
    \
    --seed 42 \
    --report_to none \
&& { echo "  ✓ TEST 1 passed (adaptive masking, full params)"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST 1 FAILED"; FAIL=$((FAIL+1)); }

# ═══════════════════════════════════════════════════════════════
#  TEST 2: Standard masking — ALL real flags
#          (mirrors pretrain_dnabert2_standard_mlm.slurm)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "─── TEST 2: Standard masking (full param mirror) ─────────"
echo ""

python3 "$SCRIPT" \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --max_seq_length 512 \
    \
    --train_file "$TRAIN_FILE" \
    --validation_file "$VAL_FILE" \
    --max_train_samples 200 \
    --max_eval_samples 50 \
    --preprocessing_num_workers 2 \
    \
    --target_mlm_prob 0.15 \
    \
    --output_dir "${OUTPUT_DIR}/standard" \
    --max_steps 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --optim adamw_torch \
    \
    --early_stopping_patience 3 \
    \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_steps 10 \
    --logging_steps 5 \
    --save_total_limit 2 \
    \
    $FP16_FLAG \
    --gradient_checkpointing \
    --dataloader_num_workers 2 \
    --dataloader_pin_memory \
    \
    --seed 42 \
    --report_to none \
&& { echo "  ✓ TEST 2 passed (standard masking, full params)"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST 2 FAILED"; FAIL=$((FAIL+1)); }

# ═══════════════════════════════════════════════════════════════
#  TEST 3: Adaptive masking WITHOUT gradient checkpointing
#          (fallback test — ensure training works both ways)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "─── TEST 3: Adaptive masking without grad checkpoint ─────"
echo ""

python3 "$SCRIPT" \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --max_seq_length 512 \
    \
    --train_file "$TRAIN_FILE" \
    --validation_file "$VAL_FILE" \
    --max_train_samples 100 \
    --max_eval_samples 30 \
    --preprocessing_num_workers 2 \
    \
    --use_adaptive_masking \
    --target_mlm_prob 0.15 \
    --eclip_mlm_lo 0.20 \
    --eclip_mlm_hi 0.25 \
    --min_flanking_prob 0.05 \
    \
    --output_dir "${OUTPUT_DIR}/adaptive_no_gc" \
    --max_steps 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --optim adamw_torch \
    \
    --early_stopping_patience 3 \
    \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_steps 10 \
    --logging_steps 5 \
    --save_total_limit 2 \
    \
    $FP16_FLAG \
    --dataloader_num_workers 2 \
    --dataloader_pin_memory \
    \
    --seed 42 \
    --report_to none \
&& { echo "  ✓ TEST 3 passed (no gradient checkpointing)"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST 3 FAILED"; FAIL=$((FAIL+1)); }

# ═══════════════════════════════════════════════════════════════
#  TEST 4: Epoch-based eval strategy
#          (real SLURM uses --eval_strategy epoch — verify it)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "─── TEST 4: Epoch-based evaluation strategy ──────────────"
echo ""

python3 "$SCRIPT" \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --max_seq_length 512 \
    \
    --train_file "$TRAIN_FILE" \
    --validation_file "$VAL_FILE" \
    --max_train_samples 100 \
    --max_eval_samples 30 \
    --preprocessing_num_workers 2 \
    \
    --use_adaptive_masking \
    --target_mlm_prob 0.15 \
    --eclip_mlm_lo 0.20 \
    --eclip_mlm_hi 0.25 \
    --min_flanking_prob 0.05 \
    \
    --output_dir "${OUTPUT_DIR}/adaptive_epoch" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --optim adamw_torch \
    \
    --early_stopping_patience 3 \
    \
    --eval_strategy epoch \
    --logging_steps 5 \
    --save_total_limit 2 \
    \
    $FP16_FLAG \
    --gradient_checkpointing \
    --dataloader_num_workers 2 \
    --dataloader_pin_memory \
    \
    --seed 42 \
    --report_to none \
&& { echo "  ✓ TEST 4 passed (epoch-based eval)"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST 4 FAILED"; FAIL=$((FAIL+1)); }

# ═══════════════════════════════════════════════════════════════
#  TEST 5: Verify masking distribution
#          (unit test: ~15% total, eCLIP > flanking)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "─── TEST 5: Verify masking distribution ──────────────────"
echo ""

BASE="$BASE" python3 - <<'PYEOF'
"""
Quick check that the adaptive collator produces ~15% total masking
with higher rates in eCLIP regions.

Uses realistic mixed-nucleotide DNA sequences so BPE tokenization produces
enough content tokens, then derives peak positions from actual token counts.
"""
import json, torch, numpy as np, sys, os, random

sys.path.insert(0, os.path.join(
    os.environ.get("BASE", "."), "DNABERT2", "pretrain"))
from transformers import AutoTokenizer
from pretrain_dnabert2 import EclipAdaptiveMLMCollator

# ── Load tokenizer ────────────────────────────────────────────
print("  Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True, model_max_length=512,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Build realistic mixed-nucleotide sequences ───────────────
random.seed(42)
np.random.seed(42)

def rand_dna(n):
    return "".join(random.choices("ACGT", k=n))

seqs = [rand_dna(512) for _ in range(3)]

# Tokenize and inspect actual lengths
tokenized = tokenizer(seqs, padding="max_length", max_length=512, truncation=True)
n_content = []
for ids in tokenized["input_ids"]:
    n = sum(1 for t in ids if t != tokenizer.pad_token_id)
    n_content.append(n)

print(f"  Content tokens per sequence: {n_content}")

# ── Place eCLIP peaks within actual content region ────────────
peak_defs = []
for i, nc in enumerate(n_content):
    if i == 0:
        # Peak in middle 40% of content
        ps = nc // 4
        pe = ps + nc * 2 // 5
        peak_defs.append(([ps], [pe]))
    elif i == 2:
        # Peak in first 25% of content (skip special token at 0)
        ps = 1
        pe = 1 + nc // 4
        peak_defs.append(([ps], [pe]))
    else:
        peak_defs.append(([], []))

print(f"  Peak defs (token positions): {peak_defs}")

# ── Build examples ────────────────────────────────────────────
examples = []
for i in range(3):
    ex = {
        "input_ids":         tokenized["input_ids"][i],
        "attention_mask":    tokenized["attention_mask"][i],
        "eclip_token_starts": list(peak_defs[i][0]),
        "eclip_token_ends":   list(peak_defs[i][1]),
    }
    examples.append(ex)

# ── Create collator (match real SLURM params exactly) ─────────
collator = EclipAdaptiveMLMCollator(
    tokenizer=tokenizer, mlm=True,
    target_mlm_prob=0.15,
    eclip_mlm_range=(0.20, 0.25),
    min_flanking_prob=0.05,
)

# ── Run trials ───────────────────────────────────────────────
n_trials = 500
total_rates = []
eclip_rates_0 = []
flank_rates_0 = []

for _ in range(n_trials):
    batch_examples = [dict(ex) for ex in examples]
    batch = collator(batch_examples)
    labels = batch["labels"]
    input_ids = batch["input_ids"]

    for i in range(3):
        masked = (labels[i] != -100)
        pad = (input_ids[i] == tokenizer.pad_token_id)
        non_pad = ~pad
        valid = non_pad.sum().item()
        if valid > 0:
            rate = masked.sum().item() / valid
            total_rates.append(rate)

        # For example 0, measure eclip vs flank separately
        if i == 0 and peak_defs[0][0]:
            ps = peak_defs[0][0][0]
            pe = peak_defs[0][1][0]
            eclip_total  = min(pe, n_content[0]) - ps
            if eclip_total > 0:
                eclip_region = masked[ps:pe]
                eclip_masked = eclip_region[:eclip_total].sum().item()
                eclip_rates_0.append(eclip_masked / eclip_total)

            flank_total = valid - eclip_total
            if flank_total > 0:
                flank_masked = masked[:ps].sum().item() + masked[pe:n_content[0]].sum().item()
                flank_rates_0.append(flank_masked / flank_total)

avg_total = np.mean(total_rates)
avg_eclip = np.mean(eclip_rates_0) if eclip_rates_0 else 0
avg_flank = np.mean(flank_rates_0) if flank_rates_0 else 0

print()
print(f"  Average total masking rate : {avg_total:.4f}  (target: 0.15)")
print(f"  Average eCLIP masking rate : {avg_eclip:.4f}  (range: 0.20-0.25)")
print(f"  Average flank masking rate : {avg_flank:.4f}  (should be < eCLIP)")
if avg_flank > 0:
    print(f"  eCLIP / flank ratio        : {avg_eclip / avg_flank:.2f}x")
print()

# ── Assertions ────────────────────────────────────────────────
ok = True
if not (0.12 <= avg_total <= 0.18):
    print(f"  ✗ FAIL: total rate {avg_total:.4f} outside [0.12, 0.18]")
    ok = False
else:
    print(f"  ✓ Total rate {avg_total:.4f} is within [0.12, 0.18]")

if eclip_rates_0 and flank_rates_0:
    if avg_eclip > avg_flank:
        print(f"  ✓ eCLIP rate ({avg_eclip:.4f}) > flank rate ({avg_flank:.4f})")
    else:
        print(f"  ✗ FAIL: eCLIP rate ({avg_eclip:.4f}) should exceed flank rate ({avg_flank:.4f})")
        ok = False
else:
    print(f"  ⚠ Could not measure eCLIP/flank rates (lists empty)")
    ok = False

if ok:
    print("\n  ✓ Masking distribution looks correct!")
else:
    print("\n  ✗ Some masking checks failed — investigate.")
    sys.exit(1)
PYEOF

if [[ $? -eq 0 ]]; then
    echo "  ✓ TEST 5 passed (masking distribution)"
    PASS=$((PASS+1))
else
    echo "  ✗ TEST 5 FAILED"
    FAIL=$((FAIL+1))
fi

# ═══════════════════════════════════════════════════════════════
#  TEST 6: Verify checkpoint artefacts
#          (ensure model, tokenizer, trainer_state saved)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "─── TEST 6: Verify saved artefacts ───────────────────────"
echo ""

ARTEFACT_OK=true
for f in "config.json" "pytorch_model.bin" "tokenizer.json" "training_args.bin"; do
    if [[ -f "${OUTPUT_DIR}/adaptive/$f" ]]; then
        echo "  ✓ Found: adaptive/$f"
    else
        # Some configs use model.safetensors instead of pytorch_model.bin
        alt=""
        if [[ "$f" == "pytorch_model.bin" ]]; then
            alt="model.safetensors"
        fi
        if [[ -n "$alt" && -f "${OUTPUT_DIR}/adaptive/$alt" ]]; then
            echo "  ✓ Found: adaptive/$alt (alternative)"
        else
            echo "  ✗ Missing: adaptive/$f"
            ARTEFACT_OK=false
        fi
    fi
done

for f in "config.json" "pytorch_model.bin" "tokenizer.json" "training_args.bin"; do
    if [[ -f "${OUTPUT_DIR}/standard/$f" ]]; then
        echo "  ✓ Found: standard/$f"
    else
        alt=""
        if [[ "$f" == "pytorch_model.bin" ]]; then
            alt="model.safetensors"
        fi
        if [[ -n "$alt" && -f "${OUTPUT_DIR}/standard/$alt" ]]; then
            echo "  ✓ Found: standard/$alt (alternative)"
        else
            echo "  ✗ Missing: standard/$f"
            ARTEFACT_OK=false
        fi
    fi
done

if $ARTEFACT_OK; then
    echo "  ✓ TEST 6 passed (all artefacts present)"
    PASS=$((PASS+1))
else
    echo "  ✗ TEST 6 FAILED (missing artefacts)"
    FAIL=$((FAIL+1))
fi

# ═══════════════════════════════════════════════════════════════
#  TEST 7: Verify saved model can be reloaded
# ═══════════════════════════════════════════════════════════════
echo ""
echo "─── TEST 7: Reload saved model ──────────────────────────"
echo ""

python3 -c "
from transformers import BertForMaskedLM, BertConfig, AutoTokenizer
import os

out = '${OUTPUT_DIR}/adaptive'
print(f'  Loading from: {out}')
# Use BertConfig + BertForMaskedLM directly to avoid the Auto* class
# mismatch between DNABERT2's custom BertConfig and the standard one.
config = BertConfig.from_pretrained(out)
model = BertForMaskedLM.from_pretrained(out, config=config)
tok = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
n = sum(p.numel() for p in model.parameters()) / 1e6
print(f'  ✓ Model loaded: {n:.1f}M params, vocab={len(tok)}')
" \
&& { echo "  ✓ TEST 7 passed (model reload)"; PASS=$((PASS+1)); } \
|| { echo "  ✗ TEST 7 FAILED"; FAIL=$((FAIL+1)); }

# ═══════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  RESULTS:  $PASS passed,  $FAIL failed  (out of 7)"
echo ""
echo "  Test output: $OUTPUT_DIR"
echo ""

if [[ $FAIL -eq 0 ]]; then
    echo "  ✓ ALL TESTS PASSED — safe to submit SLURM jobs:"
    echo ""
    echo "    sbatch $BASE/DNABERT2/pretrain/pretrain_dnabert2_full.slurm"
    echo "    sbatch $BASE/DNABERT2/pretrain/pretrain_dnabert2_standard_mlm.slurm"
else
    echo "  ✗ $FAIL TEST(S) FAILED — fix before submitting!"
    exit 1
fi
echo "═══════════════════════════════════════════════════════════"
