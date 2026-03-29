#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SBATCH_SCRIPT="$SCRIPT_DIR/run_linear_probe_single.slurm"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"

MODELS=(
  dnabert2_singlenuc_random
  dnabert2_tapt_v4_5132
  dnabert2_tapt_v4_28226
)

LAYERS=(6 last)

echo "========================================================"
echo "Submitting linear probe jobs (split by model and layer)"
echo "SBATCH script: $SBATCH_SCRIPT"
echo "========================================================"

for model in "${MODELS[@]}"; do
  for layer in "${LAYERS[@]}"; do
    job_name="lp_${model}_${layer}"
    job_id=$(sbatch \
      --job-name="$job_name" \
      --export="MODEL_NAME=${model},LAYER_MODE=${layer}" \
      "$SBATCH_SCRIPT" | awk '{print $NF}')
    echo "submitted ${job_name} -> ${job_id}"
  done
done

echo "Done."
