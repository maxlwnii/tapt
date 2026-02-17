# LAMAR Fine-tuning & Hyperparameter Optimization

This directory contains scripts for fine-tuning **LAMAR** (ESM-based RNA language model) on RBP binding-site classification and running automated hyperparameter optimization with Optuna.

---

## Overview

| Script | Purpose |
|---|---|
| `finetune_rbp.py` | LAMAR fine-tuning with encoder weight loading, warmup, CV |
| `hpo_lamar.py` | Optuna HPO search for LAMAR fine-tuning |
| `test_finetune_lamar.sh` | Local smoke test (verify everything works before SLURM jobs) |

## Model

- **LAMAR** — Custom ESM-based architecture (`EsmForSequenceClassification`)
- Config: hidden_size=768, 12 layers, 12 heads, 3072 intermediate, rotary embeddings
- Encoder weights loaded from a pre-trained safetensors checkpoint
- Default tokenizer: `Thesis/pretrain/saving_model/tapt_lamar/checkpoint-100000/`

### Pretrained Weights

```
Thesis/LAMAR/src/pretrain/saving_model/tapt_256_standard_collator_early_stopping_20/checkpoint-217000/model.safetensors
```

## Datasets

The scripts support **two** input formats:

### 1. Koo Arrow data (`--dataset koo`)
- Path: `data/finetune_data_koo/<RBP>/`
- Format: HuggingFace `DatasetDict` (Arrow)
- Splits: `train/`, `validation/`, `test/`
- Columns: `seq, label`
- Sequences: RNA (A/C/G/U), length 101
- U→T conversion handled automatically

### 2. CSV data (`--dataset csv`)
- Path: `DNABERT2/data/<RBP>/`
- Files: `train.csv`, `dev.csv`, `test.csv`
- Columns: `sequence, label`
- Sequences: DNA (A/C/G/T), length ~101 nt

### Available RBPs

```
GTF2F1_K562_IDR    HNRNPL_K562_IDR    HNRNPM_HepG2_IDR    ILF3_HepG2_IDR
KHSRP_K562_IDR     MATR3_K562_IDR     PTBP1_HepG2_IDR     QKI_K562_IDR
```

---

## Fine-tuning Script — `finetune_rbp.py`

The original LAMAR fine-tuning script with support for:

- **Encoder weight loading** from pre-trained safetensors checkpoints
- **Freeze / unfreeze** warmup phase (freeze encoder → train classifier → unfreeze → full fine-tuning)
- **Cross-validation** (`--cv_folds`)
- **Both data formats** (Arrow and CSV)
- **Early stopping** on validation metric

### Key CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--rbp_name` | str | *required* | RBP name |
| `--data_path` | str | *required* | Path to data directory (Arrow or CSV) |
| `--pretrain_path` | str | `` | Path to `model.safetensors` |
| `--tokenizer_path` | str | auto | Path to tokenizer directory |
| `--output_dir` | str | *required* | Output directory |
| `--epochs` | int | 10 | Total training epochs |
| `--batch_size` | int | 4 | Per-device batch size |
| `--lr` | float | 3e-5 | Learning rate |
| `--cv_folds` | int | 5 | Number of CV folds (1 = no CV) |
| `--warmup_ratio` | float | 0.05 | Warmup ratio for LR scheduler |
| `--early_stopping_patience` | int | 3 | Patience for early stopping |
| `--fp16` | flag | False | Enable mixed-precision training |

### Usage

```bash
python finetune_rbp.py \
    --rbp_name PTBP1_HepG2_IDR \
    --data_path /path/to/data/finetune_data_koo/PTBP1_HepG2_IDR \
    --pretrain_path /path/to/model.safetensors \
    --output_dir ./output/lamar_finetune \
    --epochs 10 \
    --batch_size 8 \
    --lr 3e-5 \
    --cv_folds 5 \
    --fp16
```

---

## HPO Script — `hpo_lamar.py`

### Search Space

| Parameter | Range | Scale |
|---|---|---|
| `learning_rate` | 1e-5 – 1e-4 | log-uniform |
| `per_device_train_batch_size` | {4, 8, 16} | categorical |
| `weight_decay` | 0.0 – 0.1 | step 0.01 |
| `warmup_ratio` | 0.0 – 0.15 | step 0.05 |
| `num_train_epochs` | 3 – 10 | integer |
| `gradient_accumulation_steps` | {1, 2} | categorical |
| `freeze_encoder` | {True, False} | boolean |
| `warmup_epochs` | 0 – 2 | integer (only when freeze_encoder=True) |

### LAMAR-specific Features

- **Encoder freezing**: When `freeze_encoder=True`, the encoder is frozen and only the classifier head trains for `warmup_epochs`, then the full model is fine-tuned.
- Data is tokenized **once** and reused across all trials for efficiency.
- Pretrained encoder weights loaded from safetensors; classifier head is randomly initialized.

### Objective

Maximize **validation AUC** (area under ROC curve).

### Features

- **Optuna TPE sampler** with median pruner for early stopping of bad trials
- **SQLite storage** — studies can be resumed (`--study_name`)
- **Per-trial** `result.json` saved with full metrics + hyperparameters
- **Summary** `hpo_summary.json` with best trial info
- **Subset mode** (`--max_train_samples`) for quick local tests

### Usage

```bash
# Full HPO run (30 trials, Koo data)
python hpo_lamar.py \
    --rbp_name PTBP1_HepG2_IDR \
    --dataset koo \
    --pretrain_path /path/to/model.safetensors \
    --n_trials 30 \
    --output_dir ./hpo_results/lamar \
    --fp16

# Full HPO run (30 trials, CSV data)
python hpo_lamar.py \
    --rbp_name PTBP1_HepG2_IDR \
    --dataset csv \
    --pretrain_path /path/to/model.safetensors \
    --n_trials 30 \
    --output_dir ./hpo_results/lamar \
    --fp16

# Quick test (2 trials, 200 samples)
python hpo_lamar.py \
    --rbp_name QKI_K562_IDR \
    --dataset koo \
    --pretrain_path /path/to/model.safetensors \
    --n_trials 2 \
    --max_train_samples 200
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--rbp_name` | str | *required* | RBP name (e.g., `PTBP1_HepG2_IDR`) |
| `--dataset` | str | *required* | `koo` or `csv` |
| `--pretrain_path` | str | `` | Path to pretrained safetensors file |
| `--tokenizer_path` | str | auto | Path to tokenizer directory |
| `--output_dir` | str | `./hpo_results/lamar` | Root output directory |
| `--n_trials` | int | 30 | Number of Optuna trials |
| `--max_train_samples` | int | None | Max training samples (for quick tests) |
| `--fp16` | flag | False | Enable mixed-precision training |
| `--study_name` | str | auto-generated | Optuna study name (for resuming) |
| `--seed` | int | 42 | Random seed |

### Output Structure

```
hpo_results/lamar/
└── PTBP1_HepG2_IDR_koo/
    ├── optuna_study.db          # Optuna SQLite database (resumable)
    ├── hpo_summary.json         # Best trial summary
    ├── trial_0/
    │   └── result.json          # Trial hyperparams + metrics
    ├── trial_1/
    │   └── result.json
    └── ...
```

---

## Local Test — `test_finetune_lamar.sh`

Runs **5 smoke tests** on small data subsets to verify the pipeline works before submitting SLURM jobs.

| Test | What it does |
|---|---|
| 1 | Fine-tune on Koo data with `finetune_rbp.py` (2 epochs, 1 fold) |
| 2 | Fine-tune on CSV data with `finetune_rbp.py` (2 epochs, 1 fold) |
| 3 | HPO on Koo data (2 trials, 200 samples) |
| 4 | HPO on CSV data (2 trials, 200 samples) |
| 5 | Verify output artefacts exist |

### Usage

```bash
chmod +x test_finetune_lamar.sh
./test_finetune_lamar.sh
```

- Duration: ~5–10 min on GPU, ~20 min on CPU
- Uses QKI_K562_IDR (smallest dataset, ~4k total samples)
- Outputs go to `test_output/` (auto-cleaned on each run)

---

## Environment

**Conda environment:** `lamar_finetune` (fallback: `lamar_fixed`)

```
Python 3.8
torch 2.1.2+cu121
transformers
datasets
safetensors
optuna
scikit-learn
pandas
```

### Setup

```bash
conda activate lamar_finetune
pip install optuna    # if not already installed
```

---

## SLURM Submission

A typical HPO SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=hpo_lamar
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

conda activate lamar_finetune

PRETRAIN=/path/to/Thesis/LAMAR/src/pretrain/saving_model/tapt_256_standard_collator_early_stopping_20/checkpoint-217000/model.safetensors

python hpo_lamar.py \
    --rbp_name PTBP1_HepG2_IDR \
    --dataset koo \
    --pretrain_path $PRETRAIN \
    --n_trials 50 \
    --output_dir /path/to/output \
    --fp16
```
