#!/bin/bash
#SBATCH --job-name=lamar_clip_eval
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=72:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/lamar_clip_256_%j.out
#SBATCH --error=logs/lamar_clip_256_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maximilian.lewinfr@gmail.com

# Info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

# Modules
module purge
module load compiler/gnu/12
module load devel/cuda/12

# Conda env
source /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh
conda activate /home/fr/fr_fr/fr_ml642/.conda/envs/lamar_fixed

# Env vars
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR:$PYTHONPATH

# Paths
cd /gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings
mkdir -p logs

# Diagnostics
python -c "import torch, tensorflow as tf; print('Torch CUDA:', torch.cuda.is_available(), 'count', torch.cuda.device_count()); print('TF built with CUDA:', tf.test.is_built_with_cuda()); print('TF GPUs:', tf.config.list_physical_devices('GPU'))"

# Run
echo "Starting LAMAR CLIP Evaluation"
python3 LAMAR_CNN_256.py

echo "Job finished at: $(date)"
