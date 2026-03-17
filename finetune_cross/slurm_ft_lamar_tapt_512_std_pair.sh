#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --job-name=ft_lm_512std
#SBATCH --output=/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/finetune_cross/logs/ft_lm_512std_%j.out
#SBATCH --error=/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/finetune_cross/logs/ft_lm_512std_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maximilian.lewinfr@gmail.com

set -euo pipefail

source /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh
conda activate lamar_finetune 2>/dev/null || conda activate lamar_fixed

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PATH="${PATH//${VIRTUAL_ENV}\/bin:/}"
    unset VIRTUAL_ENV
fi

export TOKENIZERS_PARALLELISM=false
export THESIS_ROOT="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"

SCRIPT_DIR="$THESIS_ROOT/finetune_cross"
cd "$SCRIPT_DIR"
mkdir -p logs

if [[ -z "${PAIR_NAME:-}" ]]; then
    echo "ERROR: PAIR_NAME must be set via --export=PAIR_NAME=..."
    exit 1
fi

OUTPUT_DIR="$SCRIPT_DIR/results/cross_cell/lamar_tapt_512_std"
LAMAR_TAPT_512_STD="$THESIS_ROOT/LAMAR/src/pretrain/saving_model/tapt_512_standard_collator_1gpu/checkpoint-265000/model.safetensors"

echo "============================================================"
echo "LAMAR tapt_512_std — cross_cell — ${PAIR_NAME}"
echo "  output: ${OUTPUT_DIR}/${PAIR_NAME}"
echo "============================================================"

python finetune_cross.py \
    --model_type lamar \
    --experiment cross_cell \
    --pair_name "$PAIR_NAME" \
    --pretrain_path "$LAMAR_TAPT_512_STD" \
    --max_length 512 \
    --output_dir "$OUTPUT_DIR" \
    --seed 42
