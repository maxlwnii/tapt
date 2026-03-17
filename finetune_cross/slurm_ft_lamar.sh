#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maximilian.lewinfr@gmail.com

# ──────────────────────────────────────────────────────────────
# Finetuning — LAMAR (cross-cell / cross-length)
#
# Parameterised via --export when calling sbatch:
#   VARIANT     = random | pretrained | tapt_512 | tapt_512_std | tapt_1024
#   EXPERIMENT  = cross_cell | cross_length
#   CROSS_LENGTH_SUBSET_SIZE = 0 (all) or >0 (deterministic random subset)
#   CROSS_LENGTH_SUBSET_SEED = RNG seed for subset sampling
#   CROSS_LENGTH_USE_EXISTING_RBPS = 1 to restrict to RBPs that already exist
#                                   in key cross_length LAMAR result folders
#
# Usage:
#   sbatch --array=0-82  --export=VARIANT=pretrained,EXPERIMENT=cross_cell  slurm_ft_lamar.sh
#   sbatch --array=0-263 --export=VARIANT=tapt_1024,EXPERIMENT=cross_length slurm_ft_lamar.sh
# ──────────────────────────────────────────────────────────────

# ── Environment ──────────────────────────────────────────────
source /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh
conda activate lamar_finetune 2>/dev/null || conda activate lamar_fixed

# Strip .venv from PATH to avoid environment shadowing
if [[ -n "$VIRTUAL_ENV" ]]; then
    PATH="${PATH//${VIRTUAL_ENV}\/bin:/}"
    unset VIRTUAL_ENV
fi

PYTHON=$(which python)

export TOKENIZERS_PARALLELISM=false
export THESIS_ROOT="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"

SCRIPT_DIR="$THESIS_ROOT/finetune_cross"
cd "$SCRIPT_DIR"
mkdir -p logs

# ── Validate parameters ─────────────────────────────────────
if [[ -z "$VARIANT" ]] || [[ -z "$EXPERIMENT" ]]; then
    echo "ERROR: VARIANT and EXPERIMENT must be set via --export"
    echo "  e.g. sbatch --export=VARIANT=pretrained,EXPERIMENT=cross_cell ..."
    exit 1
fi

# ── Resolve pretrain path ────────────────────────────────────
LAMAR_DIR="$THESIS_ROOT/LAMAR"
case "$VARIANT" in
    random)
        PRETRAIN_ARG=""
        MAX_LENGTH=1024
        ;;
    pretrained)
        PRETRAIN_ARG="--pretrain_path $LAMAR_DIR/weights"
        MAX_LENGTH=1024
        ;;
    tapt_512)
        PRETRAIN_ARG="--pretrain_path $LAMAR_DIR/src/pretrain/saving_model/tapt_lamar/checkpoint-98000/model.safetensors"
        MAX_LENGTH=512
        ;;
    tapt_512_std)
        PRETRAIN_ARG="--pretrain_path $LAMAR_DIR/src/pretrain/saving_model/tapt_512_standard_collator_1gpu/checkpoint-265000/model.safetensors"
        MAX_LENGTH=512
        ;;
    tapt_1024)
        PRETRAIN_ARG="--pretrain_path $LAMAR_DIR/src/pretrain/saving_model/tapt_1024_standard_collator/checkpoint-134000/model.safetensors"
        MAX_LENGTH=1024
        ;;
    *)
        echo "ERROR: Unknown VARIANT=$VARIANT (expected: random|pretrained|tapt_512|tapt_512_std|tapt_1024)"
        exit 1
        ;;
esac

# ── Resolve pair list ────────────────────────────────────────
if [[ "$EXPERIMENT" == "cross_cell" ]]; then
    PAIRS_JSON="$THESIS_ROOT/data/cross_cell/valid_pairs.json"
elif [[ "$EXPERIMENT" == "cross_length" ]]; then
    PAIRS_JSON="$THESIS_ROOT/data/cross_length/valid_prefixes.json"
else
    echo "ERROR: Unknown EXPERIMENT=$EXPERIMENT (expected: cross_cell|cross_length)"
    exit 1
fi

if [[ ! -f "$PAIRS_JSON" ]]; then
    echo "ERROR: $PAIRS_JSON not found. Run preprocessing first."
    exit 1
fi

SUBSET_SIZE="${CROSS_LENGTH_SUBSET_SIZE:-0}"
SUBSET_SEED="${CROSS_LENGTH_SUBSET_SEED:-42}"
USE_EXISTING_RBPS="${CROSS_LENGTH_USE_EXISTING_RBPS:-0}"

PAIR_NAME=$($PYTHON -c "
import json, sys, numpy as np
from pathlib import Path
pairs = sorted(json.load(open('${PAIRS_JSON}')))
idx = int(sys.argv[1])
subset_size = int('${SUBSET_SIZE}')
seed = int('${SUBSET_SEED}')
if '${EXPERIMENT}' == 'cross_length' and int('${USE_EXISTING_RBPS}') == 1:
    thesis = Path('${THESIS_ROOT}')
    roots = [
        thesis / 'finetune_cross/results/cross_length/lamar_pretrained',
        thesis / 'finetune_cross/results/cross_length/lamar_random',
        thesis / 'finetune_cross/results/cross_length/lamar_tapt_512',
        thesis / 'finetune_cross/results/cross_length/lamar_tapt_1024',
    ]
    sets = []
    for root in roots:
        rbps = {
            d.name for d in root.iterdir()
            if d.is_dir() and (d / 'results.json').exists()
        } if root.exists() else set()
        sets.append(rbps)
    shared = sorted(set.intersection(*sets)) if sets else []
    if not shared:
        print('ERROR: no shared existing RBPs found in cross_length result folders', file=sys.stderr)
        sys.exit(1)
    pairs = [p for p in pairs if p in set(shared)]
if '${EXPERIMENT}' == 'cross_length' and subset_size > 0:
    if subset_size > len(pairs):
        print(f'ERROR: subset_size {subset_size} > available pairs {len(pairs)}', file=sys.stderr)
        sys.exit(1)
    rng = np.random.default_rng(seed)
    pairs = sorted(rng.choice(pairs, size=subset_size, replace=False).tolist())
if idx >= len(pairs):
    print(f'ERROR: index {idx} >= {len(pairs)} pairs', file=sys.stderr)
    sys.exit(1)
print(pairs[idx])
" "$SLURM_ARRAY_TASK_ID")

if [[ $? -ne 0 ]] || [[ -z "$PAIR_NAME" ]]; then
    echo "ERROR: Could not get pair name for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# ── Output directory ─────────────────────────────────────────
OUTPUT_DIR="$SCRIPT_DIR/results/${EXPERIMENT}/lamar_${VARIANT}"

RESULTS_FILE="${OUTPUT_DIR}/${PAIR_NAME}/results.json"
if [[ -f "$RESULTS_FILE" ]]; then
    echo "[SKIP] Already complete: ${RESULTS_FILE}"
    exit 0
fi

echo "============================================================"
echo "LAMAR ${VARIANT} — ${EXPERIMENT} — ${PAIR_NAME}"
echo "  output: ${OUTPUT_DIR}/${PAIR_NAME}"
echo "  python: $($PYTHON --version 2>&1)"
echo "============================================================"

# ── Run finetuning ───────────────────────────────────────────
$PYTHON finetune_cross.py \
    --model_type lamar \
    --experiment "$EXPERIMENT" \
    --pair_name "$PAIR_NAME" \
    $PRETRAIN_ARG \
    --max_length "$MAX_LENGTH" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42
