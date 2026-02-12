#!/bin/bash

# List of RBPs with their sequence lengths
declare -A SEQ_LENGTHS=(
    ["GTF2F1_K562_IDR"]=101
    ["HNRNPL_K562_IDR"]=101
    ["HNRNPM_HepG2_IDR"]=101
    ["ILF3_HepG2_IDR"]=101
    ["KHSRP_K562_IDR"]=101
    ["MATR3_K562_IDR"]=101
    ["PTBP1_HepG2_IDR"]=101
    ["QKI_K562_IDR"]=101
)

RBPS=(
    "GTF2F1_K562_IDR"
    "HNRNPL_K562_IDR"
    "HNRNPM_HepG2_IDR"
    "ILF3_HepG2_IDR"
    "KHSRP_K562_IDR"
    "MATR3_K562_IDR"
    "PTBP1_HepG2_IDR"
    "QKI_K562_IDR"
)

# Template
TEMPLATE="#!/bin/bash
#SBATCH --job-name=dnabert2_RBP_NAME_SUFFIX
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --output=logs_dnabert2/dnabert2_RBP_NAME_SUFFIX_%j.out
#SBATCH --error=logs_dnabert2/dnabert2_RBP_NAME_SUFFIX_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maximilian.lewinfr@gmail.com

# Activate DNABERT2 environment
source /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh
conda activate dnabert2

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export DATA_PATH=./data/RBP_NAME
export MAX_LENGTH=MAX_LEN
export LR=3e-5

cd /gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/DNABERT_2_project

python train.py \\
    --model_name_or_path zhihan1996/DNABERT-2-117M \\
    --data_path \${DATA_PATH} \\
    --kmer -1 \\
    --fold FOLD \\
    USE_RANDOM \\
    --run_name DNABERT2_RBP_NAME_SUFFIX \\
    --model_max_length \${MAX_LENGTH} \\
    --per_device_train_batch_size 8 \\
    --per_device_eval_batch_size 16 \\
    --gradient_accumulation_steps 1 \\
    --learning_rate \${LR} \\
    --num_train_epochs 5 \\
    --fp16 \\
    --save_steps 200 \\
    --eval_steps 200 \\
    --output_dir output/dnabert2_RBP_NAME_SUFFIX \\
    --warmup_steps 50 \\
    --logging_steps 100 \\
    --overwrite_output_dir \\
    --log_level info

echo \"Job finished at: \$(date)\"
"

# Generate files
for RBP in "${RBPS[@]}"; do
    SEQ_LEN=${SEQ_LENGTHS[$RBP]}
    MAX_LEN=$((SEQ_LEN / 4))
    for FOLD in {0..4}; do
        for USE_RANDOM in "" "--use_random_init"; do
            if [ -z "$USE_RANDOM" ]; then
                SUFFIX="_fold${FOLD}"
            else
                SUFFIX="_fold${FOLD}_random"
            fi
            FILENAME="logs_dnabert2/dnabert2_${RBP}${SUFFIX}.slurm"
            echo "$TEMPLATE" | sed "s/RBP_NAME/${RBP}/g; s/MAX_LEN/${MAX_LEN}/g; s/FOLD/${FOLD}/g; s/USE_RANDOM/${USE_RANDOM}/g; s/SUFFIX/${SUFFIX}/g" > "$FILENAME"
            echo "Created $FILENAME with MAX_LENGTH=$MAX_LEN, fold=$FOLD, random=$USE_RANDOM"
        done
    done
done