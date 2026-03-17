#!/usr/bin/env bash
#SBATCH --job-name=lp_taptv3_lamar512
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/logs/linear_probe_taptv3_lamar512_%j.out
#SBATCH --error=/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/logs/linear_probe_taptv3_lamar512_%j.err

set -euo pipefail

THESIS_ROOT="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"
VENV="${THESIS_ROOT}/.venv/bin/activate"

mkdir -p "${THESIS_ROOT}/logs"

if [[ -f "${VENV}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV}"
    echo "[env] activated: ${VENV}"
    python --version
else
    echo "[WARN] venv not found: ${VENV}"
fi

export PYTHONPATH="${THESIS_ROOT}/LAMAR:${PYTHONPATH:-}"

echo "[GPU check]"
python - <<'PY'
import torch
print('  CUDA available:', torch.cuda.is_available())
print('  Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY

DATA_ROOT_1="${THESIS_ROOT}/DNABERT2/data"
DATA_ROOT_2="${THESIS_ROOT}/data/finetune_data_koo"

DNABERT_TAPT_V3_CKPT="${THESIS_ROOT}/DNABERT2/pretrain/models/dnabert2_tapt_v3/checkpoint-2566"
LAMAR_TAPT_512_STD_CKPT="${THESIS_ROOT}/LAMAR/src/pretrain/saving_model/tapt_512_standard_collator_1gpu/checkpoint-265000"

# Fresh output roots for this dedicated run
DNABERT_OUT="${THESIS_ROOT}/DNABERT2/evalEmbeddings/results/linear_probe_tapt_v3_only"
DNABERT_CACHE="${DNABERT_OUT}/cache"
LAMAR_OUT="${THESIS_ROOT}/LAMAR/evalEmbeddings/results/linear_probe_tapt_512_std_only"
LAMAR_CACHE="${LAMAR_OUT}/cache"

mkdir -p "${DNABERT_OUT}" "${DNABERT_CACHE}" "${LAMAR_OUT}" "${LAMAR_CACHE}"

echo ""
echo "[1/2] DNABERT2 TAPT-v3 linear probe"
python "${THESIS_ROOT}/DNABERT2/evalEmbeddings/linear_probe_embedding_quality.py  --enable_layer_search" \
    --data_roots "${DATA_ROOT_1}" "${DATA_ROOT_2}" \
    --tapt_dnabert2_model "${DNABERT_TAPT_V3_CKPT}" \
    --embedding_models tapt_dnabert2 \
    --output_dir "${DNABERT_OUT}" \
    --cache_dir "${DNABERT_CACHE}" \
    --max_length 512 \
    --batch_size 32 \
    --num_folds 5 \
    --seed 42

echo ""
echo "[2/2] LAMAR TAPT-512-std linear probe"
python "${THESIS_ROOT}/LAMAR/evalEmbeddings/linear_probe_lamar.py --enable_layer_search" \
    --data_roots "${DATA_ROOT_1}" "${DATA_ROOT_2}" \
    --models lamar_tapt_512_std \
    --tapt_512_checkpoint "${LAMAR_TAPT_512_STD_CKPT}" \
    --output_dir "${LAMAR_OUT}" \
    --cache_dir "${LAMAR_CACHE}" \
    --max_length 512 \
    --batch_size 32 \
    --num_folds 5 \
    --seed 42

echo ""
echo "[done]"
echo "  DNABERT2 output: ${DNABERT_OUT}"
echo "  LAMAR output   : ${LAMAR_OUT}"
