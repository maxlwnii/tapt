#!/usr/bin/env bash
# =============================================================================
#  submit_all.sh
#  Submit one SLURM job per model variant (all 18 RBPs per job).
#
#  Usage:
#    bash submit_all.sh              # submit all 11 variants
#    bash submit_all.sh lamar        # submit only LAMAR variants
#    bash submit_all.sh dnabert2     # submit only DNABERT2 variants
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LAMAR_SBATCH="${SCRIPT_DIR}/slurm_ft_lamar.sh"
DB2_SBATCH="${SCRIPT_DIR}/slurm_ft_dnabert2.sh"

mkdir -p "${SCRIPT_DIR}/logs"

FILTER="${1:-all}"

# ── LAMAR variants  (36h each) ────────────────────────────────────────────────
LAMAR_VARIANTS=(
    lamar_pretrained
    lamar_tapt_1024
    lamar_tapt_512
    lamar_tapt_512_std
    lamar_tapt_standard_1gpu
    lamar_tapt_custom_1gpu
    lamar_random
)

# ── DNABERT-2 variants  (12h each) ───────────────────────────────────────────
DB2_VARIANTS=(
    dnabert2_tapt
)

OUTPUT_DIR="${SCRIPT_DIR}/results/rbp"

# ── All 18 RBPs (must mirror slurm_ft_*.sh) ──────────────────────────────────
ALL_RBPS=(
    HNRNPK_K562_200
    PTBP1_K562_200
    PUM2_K562_200
    QKI_K562_200
    RBFOX2_K562_200
    SF3B4_K562_200
    SRSF1_K562_200
    TARDBP_K562_200
    TIA1_K562_200
    U2AF1_K562_200
    GTF2F1_K562_IDR
    HNRNPL_K562_IDR
    HNRNPM_HepG2_IDR
    ILF3_HepG2_IDR
    KHSRP_K562_IDR
    MATR3_K562_IDR
    PTBP1_HepG2_IDR
    QKI_K562_IDR
)

# Returns number of RBPs still missing results.json for a given variant.
count_pending() {
    local variant="$1"
    local pending=0
    for rbp in "${ALL_RBPS[@]}"; do
        [[ -f "${OUTPUT_DIR}/${variant}/${rbp}/results.json" ]] || (( pending++ )) || true
    done
    echo "${pending}"
}

echo "========================================================"
echo "  Submitting finetune_rbp jobs"
echo "  Filter: ${FILTER}"
echo "========================================================"

if [[ "${FILTER}" == "all" || "${FILTER}" == "lamar" ]]; then
    echo ""
    echo "  --- LAMAR variants (--time=36:00:00) ---"
    for VARIANT in "${LAMAR_VARIANTS[@]}"; do
        PENDING=$(count_pending "${VARIANT}")
        DONE=$(( ${#ALL_RBPS[@]} - PENDING ))
        if [[ "${PENDING}" -eq 0 ]]; then
            echo "  [SKIP] ${VARIANT} – all ${#ALL_RBPS[@]} RBPs already done"
            continue
        fi
        JOB_ID=$(sbatch \
            --job-name="ft_rbp_${VARIANT}" \
            --export="MODEL_VARIANT=${VARIANT}" \
            "${LAMAR_SBATCH}" | awk '{print $NF}')
        echo "  submitted ${VARIANT}  →  job ${JOB_ID}  (${DONE} done, ${PENDING} pending)"
    done
fi

if [[ "${FILTER}" == "all" || "${FILTER}" == "dnabert2" ]]; then
    echo ""
    echo "  --- DNABERT-2 variants (--time=12:00:00) ---"
    for VARIANT in "${DB2_VARIANTS[@]}"; do
        PENDING=$(count_pending "${VARIANT}")
        DONE=$(( ${#ALL_RBPS[@]} - PENDING ))
        if [[ "${PENDING}" -eq 0 ]]; then
            echo "  [SKIP] ${VARIANT} – all ${#ALL_RBPS[@]} RBPs already done"
            continue
        fi
        JOB_ID=$(sbatch \
            --job-name="ft_rbp_${VARIANT}" \
            --export="MODEL_VARIANT=${VARIANT}" \
            "${DB2_SBATCH}" | awk '{print $NF}')
        echo "  submitted ${VARIANT}  →  job ${JOB_ID}  (${DONE} done, ${PENDING} pending)"
    done
fi

echo ""
echo "  All jobs submitted."
echo "  Logs: ${SCRIPT_DIR}/logs/"
echo "  Results will appear in: ${SCRIPT_DIR}/results/rbp/<variant>/<rbp>/"
