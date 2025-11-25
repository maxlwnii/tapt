work_dir=/home/fr/fr_fr/fr_ml642/Thesis/
export PYTHONPATH="${work_dir}/LAMAR:$PYTHONPATH"

export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4

torchrun \
--nnodes 1 \
--nproc_per_node 1 \
pretrain.py \
--tokenizer_path=${work_dir}/LAMAR/tokenizer/single_nucleotide \
--model_max_length=512 \
--model_name=${work_dir}/LAMAR/config/config_150M.json \
--positional_embedding_type=rotary \
--hidden_size=768 \
--intermediate_size=3072 \
--num_attention_heads=12 \
--num_hidden_layers=12 \
--data_for_pretrain_path=${work_dir}/preprocess/preprocessed_data_metadata_train.json \
--data_for_validation_path=${work_dir}/preprocess/preprocessed_data_metadata_val.json \
--batch_size=16 \
--peak_lr=5e-5 \
--warmup_ratio=0.01 \
--max_steps=35000 \
--grad_clipping_norm=1.0 \
--accum_steps=4 \
--output_dir=${work_dir}/pretrain/saving_model/tapt_laa \
--save_steps=2000 \
--logging_steps=100 \
--fp16 \
--flash_attention \
--data_collator_patch