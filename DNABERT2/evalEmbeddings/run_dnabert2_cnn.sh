#!/bin/bash
#SBATCH --job-name=dnabert2_cnn
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/dnabert2_cnn_%j.out
#SBATCH --error=logs/dnabert2_cnn_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maximilian.lewinfr@gmail.com

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

module purge
module load compiler/gnu/12
module load devel/cuda/12

source /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh
conda activate /home/fr/fr_fr/fr_ml642/.conda/envs/dnabert2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/DNABERT2:$PYTHONPATH

cd /gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/DNABERT2/evalEmbeddings

python -c "import torch, tensorflow as tf; print('Torch CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "Starting DNABERT2 CNN (best_layer=4)"
python3 DNABERT2_CNN.py --best_layer 4

echo "Job finished at: $(date)"
