#!/usr/bin/env bash
# =============================================================================
#  submit_all_lamar_cnn.sh
#  Submit one SLURM job per LAMAR variant (~18 RBPs each, ~40h).
#  Variants that already have all 18 RBPs completed are skipped.
#
#  Usage:
#    bash submit_all_lamar_cnn.sh          # submit all pending variants
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_lamar_cnn.slurm"
RESULTS_CSV="${SCRIPT_DIR}/results/cnn_all_variants/LAMAR_CNN_results.csv"

mkdir -p "${SCRIPT_DIR}/logs"

# ── All 18 RBP task_ids (must match discover_tasks output in LAMAR_CNN.py) ───
ALL_TASK_IDS=(
    data/GTF2F1_K562_IDR
    data/HNRNPL_K562_IDR
    data/HNRNPM_HepG2_IDR
    data/ILF3_HepG2_IDR
    data/KHSRP_K562_IDR
    data/MATR3_K562_IDR
    data/PTBP1_HepG2_IDR
    data/QKI_K562_IDR
    finetune_data_koo/HNRNPK_K562_200
    finetune_data_koo/PTBP1_K562_200
    finetune_data_koo/PUM2_K562_200
    finetune_data_koo/QKI_K562_200
    finetune_data_koo/RBFOX2_K562_200
    finetune_data_koo/SF3B4_K562_200
    finetune_data_koo/SRSF1_K562_200
    finetune_data_koo/TARDBP_K562_200
    finetune_data_koo/TIA1_K562_200
    finetune_data_koo/U2AF1_K562_200
)

TOTAL=${#ALL_TASK_IDS[@]}

# ── Variants to evaluate ─────────────────────────────────────────────────────
VARIANTS=(
    lamar_tapt_1024
    lamar_tapt_standard_1gpu
    lamar_tapt_512
    lamar_tapt_512_std
    lamar_random
)

# ── Count pending tasks for a given variant by reading the CSV ───────────────
count_pending() {
    local variant="$1"
    if [[ ! -f "${RESULTS_CSV}" ]]; then
        echo "${TOTAL}"
        return
    fi
    local done_count
    done_count=$(awk -F',' -v v="${variant}" 'NR>1 && $1==v {count++} END {print count+0}' "${RESULTS_CSV}")
    echo $(( TOTAL - done_count ))
}

echo "========================================================"
echo "  Submitting LAMAR CNN jobs (one per variant)"
echo "  Results CSV: ${RESULTS_CSV}"
echo "========================================================"

for VARIANT in "${VARIANTS[@]}"; do
    PENDING=$(count_pending "${VARIANT}")
    DONE=$(( TOTAL - PENDING ))
    if [[ "${PENDING}" -eq 0 ]]; then
        echo "  [SKIP] ${VARIANT} – all ${TOTAL} tasks already done"
        continue
    fi
    JOB_ID=$(sbatch \
        --job-name="lamar_cnn_${VARIANT}" \
        --export="MODEL_VARIANT=${VARIANT}" \
        "${SBATCH_SCRIPT}" | awk '{print $NF}')
    echo "  submitted ${VARIANT}  →  job ${JOB_ID}  (${DONE} done, ${PENDING} pending)"
done

echo ""
echo "  Logs: ${SCRIPT_DIR}/logs/"
echo "  Results: ${SCRIPT_DIR}/results/cnn_all_variants/LAMAR_CNN_results.csv"
