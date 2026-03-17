#!/bin/bash
# Submit cross-length finetuning arrays for the same finetunable model set
# used in linear probing (excluding one_hot).
#
# Models:
#   LAMAR:    pretrained, tapt_1024, tapt_512, tapt_512_std, random
#   DNABERT2: pretrained, tapt, tapt_v3, random
#
# Each submission is an array job over a deterministic random subset of 44 RBPs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

ARRAY_MAX=43
SUBSET_SIZE=44
SUBSET_SEED=42
USE_EXISTING_RBPS=1

echo "============================================================"
echo "  Submitting cross-length arrays for 44 RBPs"
echo "  subset_size=${SUBSET_SIZE}, subset_seed=${SUBSET_SEED}"
echo "  use_existing_rbps=${USE_EXISTING_RBPS}"
echo "  array indices=0-${ARRAY_MAX}"
echo "============================================================"
echo ""

echo "-- LAMAR variants --"
for VARIANT in pretrained tapt_1024 tapt_512 tapt_512_std random; do
    JID=$(sbatch --parsable \
        --array=0-${ARRAY_MAX} \
        --job-name="ft_lm_${VARIANT}_cl44" \
        --output="logs/ft_lamar_${VARIANT}_cl44_%A_%a.out" \
        --error="logs/ft_lamar_${VARIANT}_cl44_%A_%a.err" \
        --export=VARIANT=${VARIANT},EXPERIMENT=cross_length,CROSS_LENGTH_SUBSET_SIZE=${SUBSET_SIZE},CROSS_LENGTH_SUBSET_SEED=${SUBSET_SEED},CROSS_LENGTH_USE_EXISTING_RBPS=${USE_EXISTING_RBPS} \
        slurm_ft_lamar.sh)
    echo "  submitted lamar_${VARIANT}: job ${JID}"
done

echo ""
echo "-- DNABERT2 variants --"
for VARIANT in pretrained tapt tapt_v3 random; do
    JID=$(sbatch --parsable \
        --array=0-${ARRAY_MAX} \
        --job-name="ft_db2_${VARIANT}_cl44" \
        --output="logs/ft_dnabert2_${VARIANT}_cl44_%A_%a.out" \
        --error="logs/ft_dnabert2_${VARIANT}_cl44_%A_%a.err" \
        --export=VARIANT=${VARIANT},EXPERIMENT=cross_length,CROSS_LENGTH_SUBSET_SIZE=${SUBSET_SIZE},CROSS_LENGTH_SUBSET_SEED=${SUBSET_SEED},CROSS_LENGTH_USE_EXISTING_RBPS=${USE_EXISTING_RBPS} \
        slurm_ft_dnabert2.sh)
    echo "  submitted dnabert2_${VARIANT}: job ${JID}"
done

echo ""
echo "Done. Submitted 9 array jobs x 44 tasks = 396 tasks total."
echo "Monitor with: squeue -u $USER"
