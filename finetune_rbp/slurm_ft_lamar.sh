#!/usr/bin/env bash
# =============================================================================
#  slurm_ft_lamar.sh
#  Fine-tune ONE LAMAR model variant on all 18 RBPs.
#
#  Submit with:
#    sbatch --export=MODEL_VARIANT=lamar_pretrained   slurm_ft_lamar.sh
#    sbatch --export=MODEL_VARIANT=lamar_tapt_1024    slurm_ft_lamar.sh
#    sbatch --export=MODEL_VARIANT=lamar_tapt_512     slurm_ft_lamar.sh
#    sbatch --export=MODEL_VARIANT=lamar_tapt_512_std slurm_ft_lamar.sh
#    sbatch --export=MODEL_VARIANT=lamar_tapt_standard_1gpu slurm_ft_lamar.sh
#    sbatch --export=MODEL_VARIANT=lamar_tapt_custom_1gpu   slurm_ft_lamar.sh
#    sbatch --export=MODEL_VARIANT=lamar_random       slurm_ft_lamar.sh
#
#SBATCH --job-name=ft_rbp_lamar
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --output=/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/finetune_rbp/logs/ft_lamar_%x_%j.out
#SBATCH --error=/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/finetune_rbp/logs/ft_lamar_%x_%j.err
# =============================================================================
set -euo pipefail

# ── Sanity check ──────────────────────────────────────────────────────────────
if [[ -z "${MODEL_VARIANT:-}" ]]; then
    echo "[ERROR] MODEL_VARIANT is not set."
    echo "  Usage: sbatch --export=MODEL_VARIANT=lamar_pretrained slurm_ft_lamar.sh"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
THESIS_ROOT="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"
SCRIPT_DIR="${THESIS_ROOT}/finetune_rbp"
VENV="${THESIS_ROOT}/.venv"

# ── Activate Python environment ───────────────────────────────────────────────
source "${VENV}/bin/activate"
echo "[env] $(python --version)"

# ── Export required paths ─────────────────────────────────────────────────────
export THESIS_ROOT
export PYTHONPATH="${THESIS_ROOT}/LAMAR:${PYTHONPATH:-}"

# ── GPU check ─────────────────────────────────────────────────────────────────
echo "[GPU check]"
python -c "import torch; print('  CUDA:', torch.cuda.is_available()); \
           print('  Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ── Output dir ────────────────────────────────────────────────────────────────
OUTPUT_DIR="${SCRIPT_DIR}/results/rbp"
mkdir -p "${OUTPUT_DIR}" "${SCRIPT_DIR}/logs"

# ── All 18 RBPs ───────────────────────────────────────────────────────────────
# 10 from finetune_data_koo (suffix _200)
KOO_RBPS=(
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
)

# 8 from DNABERT2/data (IDR-based)
CSV_RBPS=(
    GTF2F1_K562_IDR
    HNRNPL_K562_IDR
    HNRNPM_HepG2_IDR
    ILF3_HepG2_IDR
    KHSRP_K562_IDR
    MATR3_K562_IDR
    PTBP1_HepG2_IDR
    QKI_K562_IDR
)

ALL_RBPS=("${KOO_RBPS[@]}" "${CSV_RBPS[@]}")

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  Model variant: ${MODEL_VARIANT}"
echo "  Total RBPs:    ${#ALL_RBPS[@]}"
echo "======================================================================"

for RBP in "${ALL_RBPS[@]}"; do
    RESULTS_FILE="${OUTPUT_DIR}/${MODEL_VARIANT}/${RBP}/results.json"
    if [[ -f "${RESULTS_FILE}" ]]; then
        echo "[SKIP] ${MODEL_VARIANT}/${RBP} – results.json already exists"
        continue
    fi

    echo ""
    echo "----------------------------------------------------------------------"
    echo "  Running: ${MODEL_VARIANT}  /  ${RBP}"
    echo "----------------------------------------------------------------------"

    python "${SCRIPT_DIR}/finetune_rbp.py" \
        --variant    "${MODEL_VARIANT}"    \
        --rbp_name   "${RBP}"             \
        --output_dir "${OUTPUT_DIR}"
done

echo ""
echo "[done] ${MODEL_VARIANT} – all RBPs completed."
echo "  Results in: ${OUTPUT_DIR}/${MODEL_VARIANT}/"
