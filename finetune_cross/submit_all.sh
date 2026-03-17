#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Submit all cross-cell + cross-length finetuning jobs.
#
# 12 job arrays total:
#   4 LAMAR  variants × 2 experiments (cross_cell + cross_length)
#   2 DNABERT2 variants × 2 experiments
#
# Cross-cell:   83 tasks  (array 0-82)
# Cross-length: 264 tasks (array 0-263)
# Total tasks:  6 variants × (83 + 264) = 2082
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

N_CELL=82     # 0-82 inclusive = 83 tasks
N_LEN=263     # 0-263 inclusive = 264 tasks

echo "═══════════════════════════════════════════════════════════"
echo "  Submitting all cross-cell + cross-length finetuning jobs"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── LAMAR ────────────────────────────────────────────────────
echo "── LAMAR (4 variants × 2 experiments = 8 array jobs) ──"
for VARIANT in random pretrained tapt_512 tapt_1024; do
    JID1=$(sbatch --parsable \
        --array=0-${N_CELL} \
        --job-name="ft_lm_${VARIANT}_cc" \
        --output="logs/ft_lamar_${VARIANT}_crosscell_%A_%a.out" \
        --error="logs/ft_lamar_${VARIANT}_crosscell_%A_%a.err" \
        --export=VARIANT=${VARIANT},EXPERIMENT=cross_cell \
        slurm_ft_lamar.sh)
    echo "  LAMAR ${VARIANT} × cross_cell:   job ${JID1} ($((N_CELL+1)) tasks)"

    JID2=$(sbatch --parsable \
        --array=0-${N_LEN} \
        --job-name="ft_lm_${VARIANT}_cl" \
        --output="logs/ft_lamar_${VARIANT}_crosslen_%A_%a.out" \
        --error="logs/ft_lamar_${VARIANT}_crosslen_%A_%a.err" \
        --export=VARIANT=${VARIANT},EXPERIMENT=cross_length \
        slurm_ft_lamar.sh)
    echo "  LAMAR ${VARIANT} × cross_length: job ${JID2} ($((N_LEN+1)) tasks)"
done
echo ""

# ── DNABERT2 ─────────────────────────────────────────────────
echo "── DNABERT2 (2 variants × 2 experiments = 4 array jobs) ──"
for VARIANT in pretrained random; do
    JID1=$(sbatch --parsable \
        --array=0-${N_CELL} \
        --job-name="ft_db2_${VARIANT}_cc" \
        --output="logs/ft_dnabert2_${VARIANT}_crosscell_%A_%a.out" \
        --error="logs/ft_dnabert2_${VARIANT}_crosscell_%A_%a.err" \
        --export=VARIANT=${VARIANT},EXPERIMENT=cross_cell \
        slurm_ft_dnabert2.sh)
    echo "  DNABERT2 ${VARIANT} × cross_cell:   job ${JID1} ($((N_CELL+1)) tasks)"

    JID2=$(sbatch --parsable \
        --array=0-${N_LEN} \
        --job-name="ft_db2_${VARIANT}_cl" \
        --output="logs/ft_dnabert2_${VARIANT}_crosslen_%A_%a.out" \
        --error="logs/ft_dnabert2_${VARIANT}_crosslen_%A_%a.err" \
        --export=VARIANT=${VARIANT},EXPERIMENT=cross_length \
        slurm_ft_dnabert2.sh)
    echo "  DNABERT2 ${VARIANT} × cross_length: job ${JID2} ($((N_LEN+1)) tasks)"
done
echo ""

TOTAL=$(( (N_CELL+1)*6 + (N_LEN+1)*6 ))
echo "═══════════════════════════════════════════════════════════"
echo "  Done! 12 array jobs submitted."
echo "  Total tasks: ${TOTAL}"
echo "  Monitor: squeue -u \$USER"
echo "═══════════════════════════════════════════════════════════"
