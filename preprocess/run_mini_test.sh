#!/bin/bash
# Minimal test runner for pretrain.py using the mini datasets
work_dir=/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis
source /gpfs/bwfor/software/common/devel/miniforge/24.9.2-0/etc/profile.d/conda.sh
conda activate /home/fr/fr_fr/fr_ml642/.conda/envs/lamar_fixed

cd ${work_dir}/LAMAR/src/pretrain

torchrun --nnodes 1 --nproc_per_node 1 pretrain.py \
  --tokenizer_path=${work_dir}/LAMAR/tokenizer/single_nucleotide \
  --model_max_length=256 \
  --model_name=${work_dir}/LAMAR/src/pretrain/config_150M.json \
  --positional_embedding_type=rotary \
  --hidden_size=768 \
  --intermediate_size=3072 \
  --num_attention_heads=12 \
  --num_hidden_layers=12 \
  --data_for_pretrain_path=${work_dir}/preprocess/preprocessed_data_mini_metadata_train.json \
  --data_for_validation_path=${work_dir}/preprocess/preprocessed_data_mini_metadata_val.json \
  --batch_size=2 \
  --peak_lr=5e-5 \
  --warmup_ratio=0.01 \
  --max_steps=5 \
  --grad_clipping_norm=1.0 \
  --accum_steps=1 \
  --output_dir=${work_dir}/pretrain/saving_model/mini_test_out \
  --save_steps=1000 \
  --logging_steps=1 \
  --lr_scheduler_type=linear \
  --optim=adamw_torch \
  --dataloader_num_workers=0 \
  --seed=42

echo "Mini test finished: $(date)"
