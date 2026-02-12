export TOKENIZERS_PARALLELISM=false
export DATA_PATH=./data/PTBP1_HepG2_IDR
export MAX_LENGTH=25
export LR=3e-5

python train.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --data_path ${DATA_PATH} \
    --kmer -1 \
    --fold 5 \
    --use_random_init \
    --run_name DNABERT2_PTBP1_HepG2_IDR__fold5_random \
    --model_max_length ${MAX_LENGTH} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs 50 \
    --early_stopping_patience 5 \
    --fp16 \
    --save_steps 200 \
    --eval_steps 200 \
    --output_dir output/dnabert2_PTBP1_HepG2_IDR__fold5_random \
    --warmup_steps 50 \
    --logging_steps 100 \
    --overwrite_output_dir \
    --log_level info