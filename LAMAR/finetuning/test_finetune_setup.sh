#!/bin/bash
# Quick test script to verify fine-tuning setup works correctly
# Run this before submitting SLURM jobs

set -e

echo "========================================"
echo "Testing LAMAR Fine-tuning Setup"
echo "========================================"

# Activate conda environment
source /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh
conda activate lamar_finetune

module load devel/cuda/12
module load compiler/gnu/12

echo "✓ Conda environment activated: $CONDA_DEFAULT_ENV"

# Define paths
WORKSPACE_ROOT="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"
LAMAR_ROOT="${WORKSPACE_ROOT}/LAMAR"
PRETRAIN_WEIGHTS="${LAMAR_ROOT}/src/pretrain/saving_model/tapt_256_standard_collator_early_stopping_20/checkpoint-217000/model.safetensors"
TOKENIZER_PATH="${LAMAR_ROOT}/src/pretrain/saving_model/tapt_lamar/checkpoint-100000"
DATA_ROOT="${WORKSPACE_ROOT}/DNBERT2/data"
OUTPUT_ROOT="${LAMAR_ROOT}/finetuning/test_output"

# Pick first RBP for testing
TEST_RBP=$(ls -d ${DATA_ROOT}/*/ | head -n 1)
RBP_NAME=$(basename "${TEST_RBP}")

echo ""
echo "Test Configuration:"
echo "  RBP: ${RBP_NAME}"
echo "  Pretrain weights: ${PRETRAIN_WEIGHTS}"
echo "  Tokenizer: ${TOKENIZER_PATH}"
echo "  Output: ${OUTPUT_ROOT}"
echo ""

# Check if files exist
if [ ! -f "${PRETRAIN_WEIGHTS}" ]; then
    echo "❌ ERROR: Model weights not found at ${PRETRAIN_WEIGHTS}"
    exit 1
fi

if [ ! -d "${TOKENIZER_PATH}" ]; then
    echo "❌ ERROR: Tokenizer not found at ${TOKENIZER_PATH}"
    exit 1
fi

if [ ! -d "${TEST_RBP}" ]; then
    echo "❌ ERROR: Test RBP directory not found at ${TEST_RBP}"
    exit 1
fi

echo "✓ All files exist"
echo ""

# Navigate to finetune_scripts directory
cd "${LAMAR_ROOT}/finetune_scripts"

echo "Starting test fine-tuning (2 epochs)..."
echo "========================================"

# Run with minimal epochs for quick test
python finetune_rbp.py \
    --rbp_name "${RBP_NAME}" \
    --data_path "${TEST_RBP}" \
    --output_dir "${OUTPUT_ROOT}/test_${RBP_NAME}" \
    --pretrain_path "${PRETRAIN_WEIGHTS}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --epochs 2 \
    --batch_size 8 \
    --lr 3e-5 \
    --cv_folds 1 \
    --nlabels 2 \
    --warmup_ratio 0.05 \
    --logging_steps 50 \
    --save_epochs 1 \
    --fp16

echo ""
echo "========================================"
echo "✓ Test completed successfully!"
echo "========================================"
echo ""
echo "Check results in: ${OUTPUT_ROOT}/test_${RBP_NAME}"
echo ""
echo "If everything looks good, submit SLURM jobs with:"
echo "  cd ${LAMAR_ROOT}/finetuning/logs/slurm"
echo "  sbatch lamar_tapt_256_standard_collator.slurm"
echo "  sbatch lamar_tapt_256_early_stopping.slurm"
echo ""
