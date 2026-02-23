#!/bin/bash
#SBATCH --job-name=hpo_lamar_pre
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --array=0-17
#SBATCH --output=logs/hpo_lamar_pretrained_%A_%a.out
#SBATCH --error=logs/hpo_lamar_pretrained_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maximilian.lewinfr@gmail.com

# ============================================================
#  LAMAR Pretrained (LAMAR/weights) — HPO
#
#  Array: 0-17 = 10 koo RBPs (0-9) + 8 csv RBPs (10-17)
#  Submit: sbatch slurm_hpo_lamar_pretrained.sh
# ============================================================

source /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh
conda activate lamar_finetune 2>/dev/null || conda activate lamar_fixed

export TOKENIZERS_PARALLELISM=false
export THESIS_ROOT="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"

LAMAR_DIR="$THESIS_ROOT/LAMAR"
cd "$LAMAR_DIR/finetune_scripts"
mkdir -p logs

# Pretrain weights: normal pretrained LAMAR (LAMAR/weights safetensors file)
PRETRAIN_PATH="$LAMAR_DIR/weights"
TOKENIZER_PATH="$LAMAR_DIR/src/pretrain/saving_model/tapt_lamar/checkpoint-100000"

# ── Map array index to RBP + dataset ──────────────────────────
RBPS_KOO=(
    HNRNPK_K562_200  PTBP1_K562_200  PUM2_K562_200    QKI_K562_200
    RBFOX2_K562_200  SF3B4_K562_200  SRSF1_K562_200   TARDBP_K562_200
    TIA1_K562_200    U2AF1_K562_200
)
RBPS_CSV=(
    GTF2F1_K562_IDR  HNRNPL_K562_IDR  HNRNPM_HepG2_IDR  ILF3_HepG2_IDR
    KHSRP_K562_IDR   MATR3_K562_IDR   PTBP1_HepG2_IDR   QKI_K562_IDR
)

NUM_KOO=${#RBPS_KOO[@]}  # 10

if [[ $SLURM_ARRAY_TASK_ID -lt $NUM_KOO ]]; then
    RBP="${RBPS_KOO[$SLURM_ARRAY_TASK_ID]}"
    DATASET="koo"
else
    IDX=$((SLURM_ARRAY_TASK_ID - NUM_KOO))
    RBP="${RBPS_CSV[$IDX]}"
    DATASET="csv"
fi

OUTPUT_DIR="$LAMAR_DIR/finetune_scripts/hpo_results/lamar_pretrained"

echo "========================================================"
echo "  LAMAR Pretrained HPO"
echo "  Weights : $PRETRAIN_PATH"
echo "  RBP     : $RBP"
echo "  Dataset : $DATASET"
echo "  ArrayID : $SLURM_ARRAY_TASK_ID"
echo "  Node    : $(hostname)"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Started : $(date)"
echo "========================================================"

python hpo_lamar.py \
    --rbp_name "$RBP" \
    --dataset "$DATASET" \
    --pretrain_path "$PRETRAIN_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_trials 30 \
    --fp16 \
    --seed 42

echo "Job finished at: $(date)"
