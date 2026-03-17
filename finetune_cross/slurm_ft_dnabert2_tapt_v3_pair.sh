#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --job-name=ft_db2_taptv3
#SBATCH --output=/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/finetune_cross/logs/ft_db2_taptv3_%j.out
#SBATCH --error=/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/finetune_cross/logs/ft_db2_taptv3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maximilian.lewinfr@gmail.com

set -euo pipefail

source /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh
conda activate dnabert2

export TOKENIZERS_PARALLELISM=false
export THESIS_ROOT="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"

SCRIPT_DIR="$THESIS_ROOT/finetune_cross"
cd "$SCRIPT_DIR"
mkdir -p logs

if [[ -z "${PAIR_NAME:-}" ]]; then
    echo "ERROR: PAIR_NAME must be set via --export=PAIR_NAME=..."
    exit 1
fi

OUTPUT_DIR="$SCRIPT_DIR/results/cross_cell/dnabert2_tapt_v3"
DNABERT2_TAPT_V3="$THESIS_ROOT/DNABERT2/pretrain/models/dnabert2_tapt_v3/checkpoint-2566"

echo "============================================================"
echo "DNABERT2 tapt_v3 — cross_cell — ${PAIR_NAME}"
echo "  output: ${OUTPUT_DIR}/${PAIR_NAME}"
echo "============================================================"

python finetune_cross.py \
    --model_type dnabert2 \
    --experiment cross_cell \
    --pair_name "$PAIR_NAME" \
    --dnabert2_model_path "$DNABERT2_TAPT_V3" \
    --max_length 128 \
    --output_dir "$OUTPUT_DIR" \
    --seed 42
