#!/usr/bin/env bash
set -euo pipefail

# Quick smoke-test for run_mlm.py using the preprocessed 256 sequences in Thesis/preprocess
# Adjust BASE if your Thesis folder is in a different path
BASE="/home/fr/fr_fr/fr_ml642/Thesis/DNABERT2_project/pretrain"
PREPROCESS_DIR="$BASE/preprocess"
SCRIPT_DIR="$BASE/DNABERT_2_project/pretrain/scripts"
OUT_DIR="$BASE/tmp_pretrain_test"

TRAIN_FILE="$PREPROCESS_DIR/preprocessed_data_256_sequences.txt"
# If you have a separate validation file, set VALIDATION_FILE; otherwise we'll let the script split the train file
VALIDATION_FILE=""

# Export PYTHONPATH so local packages (LAMAR) can be imported
export PYTHONPATH="${BASE}:${PYTHONPATH:-}"

mkdir -p "$OUT_DIR"

# Small run to verify script and custom collator import
python "$SCRIPT_DIR/run_mlm.py" \
  --do_train \
  --train_file "$TRAIN_FILE" \
  --model_type bert \
  --tokenizer_name bert-base-uncased \
  --validation_split_percentage 5 \
  --max_seq_length 512 \
  --output_dir "$OUT_DIR" \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --num_train_epochs 1 \
  --max_train_samples 200 \
  --max_eval_samples 50

echo "Quick pretrain smoke-test finished. Check logs and $OUT_DIR for outputs."
