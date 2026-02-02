#!/bin/bash
# filepath: /home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/lamar_cnn_eval.slurm
#SBATCH --job-name=lamar_cnn_eval
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/lamar_cnn_eval_%j.out
#SBATCH --error=logs/lamar_cnn_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maximilian.lewinfr@gmail.com

module purge
module load compiler/gnu/12
module load devel/cuda/12

source /home/fr/fr_fr/fr_ml642/.conda/etc/profile.d/conda.sh
conda activate lamar_fixed

export PYTHONPATH=/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR:$PYTHONPATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false

mkdir -p logs

cd /gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings
python LAMAR_CNN_updated.py