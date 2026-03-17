#!/usr/bin/env bash
# =============================================================================
#  run_unsupervised_eval.sh
#  Unsupervised embedding-quality evaluation for selected model variants.
#
#  Usage (interactive, GPU node already allocated):
#      bash run_unsupervised_eval.sh [--models NAMES...] [--no_pca_scatter]
#
#  Usage (SLURM, submit with sbatch):
#      sbatch run_unsupervised_eval.sh
#
#  SLURM directives (active when submitted via sbatch):
#SBATCH --job-name=unsup_eval
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/p_eickhoff_isoscore/logs/unsup_eval_%j.out
#SBATCH --error=/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/p_eickhoff_isoscore/logs/unsup_eval_%j.err
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
THESIS_ROOT="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"
SCRIPT_DIR="${THESIS_ROOT}/p_eickhoff_isoscore"
VENV="${THESIS_ROOT}/.venv"

# ── Activate Python environment ───────────────────────────────────────────────
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
    echo "[env] activated .venv: $(python --version)"
else
    echo "[WARN] .venv not found at ${VENV} – using system Python"
fi

# ── Export PYTHONPATH so LAMAR package is importable ─────────────────────────
export PYTHONPATH="${THESIS_ROOT}/LAMAR:${PYTHONPATH:-}"

# ── Verify GPU visibility ─────────────────────────────────────────────────────
echo "[GPU check]"
python -c "import torch; print('  CUDA available:', torch.cuda.is_available()); \
           print('  device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ── Default arguments (override via positional CLI args) ─────────────────────
DATA_ROOTS=(
    "${THESIS_ROOT}/DNABERT2/data"
    "${THESIS_ROOT}/data/finetune_data_koo"
)

OUTPUT_DIR="${SCRIPT_DIR}/results/unsupervised_eval2"
BATCH_SIZE=48         # sequences per forward pass (reduce to 16 if OOM)
N_BINS=6              # quantile bins for sequence-length sensitivity
N_SENSITIVITY=3       # number of RBP tasks used for sensitivity analysis
EXTRA_FLAGS=""        # e.g. "--pca_scatter" or "--force_reextract"

# -- Parse simple flags from command line -------------------------------------
ALL_MODELS=(
    dnabert2_tapt_v3
    lamar_tapt_512_std
)
MODELS=("${ALL_MODELS[@]}")   # run all by default

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            shift
            MODELS=()
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --pca_scatter)    EXTRA_FLAGS="${EXTRA_FLAGS} --pca_scatter"; shift ;;
        --force_reextract) EXTRA_FLAGS="${EXTRA_FLAGS} --force_reextract"; shift ;;
        --batch_size)     BATCH_SIZE="$2"; shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "[WARN] unknown arg: $1"; shift ;;
    esac
done

echo ""
echo "[config]"
echo "  output_dir  : ${OUTPUT_DIR}"
echo "  batch_size  : ${BATCH_SIZE}"
echo "  models      : ${MODELS[*]}"
echo "  data_roots  : ${DATA_ROOTS[*]}"
echo ""

mkdir -p "${OUTPUT_DIR}/logs" "${SCRIPT_DIR}/logs"

# ── Run ───────────────────────────────────────────────────────────────────────
time python "${SCRIPT_DIR}/unsupervised_eval.py" \
    --data_roots  "${DATA_ROOTS[@]}"              \
    --output_dir  "${OUTPUT_DIR}"                 \
    --models      "${MODELS[@]}"                  \
    --batch_size  "${BATCH_SIZE}"                 \
    --n_bins      "${N_BINS}"                     \
    --n_sensitivity_tasks "${N_SENSITIVITY}"      \
    ${EXTRA_FLAGS}

echo ""
echo "[done] results in: ${OUTPUT_DIR}"
echo "        plots   : ${OUTPUT_DIR}/plots/"
