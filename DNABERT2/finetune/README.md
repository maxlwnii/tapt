# DNABERT2 Fine-tuning & Hyperparameter Optimization

This directory contains scripts for fine-tuning **DNABERT-2-117M** on RBP binding-site classification and running automated hyperparameter optimization with Optuna.

---

## Overview

| Script | Purpose |
|---|---|
| `hpo_dnabert2.py` | Optuna HPO search for DNABERT2 fine-tuning |
| `test_finetune_dnabert2.sh` | Local smoke test (verify everything works before SLURM jobs) |
| `../train.py` | Original DNABERT2 fine-tuning script (upstream, used internally) |

## Model

- **DNABERT-2-117M** (`zhihan1996/DNABERT-2-117M`)
- BPE tokenizer (vocab_size = 4096, no k-mer needed)
- Loaded via `AutoModelForSequenceClassification` with `trust_remote_code=True`

## Datasets

The scripts support **two** input formats:

### 1. CSV data (`--dataset csv`)
- Path: `DNABERT2/data/<RBP>/`
- Files: `train.csv`, `dev.csv`, `test.csv`
- Columns: `sequence, label`
- Sequences: DNA (A/C/G/T), length ~101 nt

### 2. Koo Arrow data (`--dataset koo`)
- Path: `data/finetune_data_koo/<RBP>/`
- Format: HuggingFace `DatasetDict` (Arrow)
- Splits: `train/`, `validation/`, `test/`
- Columns: `seq, label`
- Sequences: RNA (A/C/G/U), length 101 — U→T converted automatically

### Available RBPs

```
GTF2F1_K562_IDR    HNRNPL_K562_IDR    HNRNPM_HepG2_IDR    ILF3_HepG2_IDR
KHSRP_K562_IDR     MATR3_K562_IDR     PTBP1_HepG2_IDR     QKI_K562_IDR
```

---

## HPO Script — `hpo_dnabert2.py`

### Search Space

| Parameter | Range | Scale |
|---|---|---|
| `learning_rate` | 1e-5 – 1e-4 | log-uniform |
| `per_device_train_batch_size` | {4, 8, 16} | categorical |
| `weight_decay` | 0.0 – 0.1 | step 0.01 |
| `warmup_ratio` | 0.0 – 0.15 | step 0.05 |
| `num_train_epochs` | 3 – 10 | integer |
| `gradient_accumulation_steps` | {1, 2} | categorical |
| `model_max_length` | {25, 50, 128} | categorical |

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
# Full HPO run (30 trials, CSV data)
python hpo_dnabert2.py \
    --rbp_name PTBP1_HepG2_IDR \
    --dataset csv \
    --n_trials 30 \
    --output_dir ./hpo_results/dnabert2 \
    --fp16

# Full HPO run (30 trials, Koo data)
python hpo_dnabert2.py \
    --rbp_name PTBP1_HepG2_IDR \
    --dataset koo \
    --n_trials 30 \
    --output_dir ./hpo_results/dnabert2 \
    --fp16

# Quick test (2 trials, 200 samples)
python hpo_dnabert2.py \
    --rbp_name QKI_K562_IDR \
    --dataset csv \
    --n_trials 2 \
    --max_train_samples 200
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--rbp_name` | str | *required* | RBP name (e.g., `PTBP1_HepG2_IDR`) |
| `--dataset` | str | *required* | `csv` or `koo` |
| `--output_dir` | str | `./hpo_results/dnabert2` | Root output directory |
| `--n_trials` | int | 30 | Number of Optuna trials |
| `--max_train_samples` | int | None | Max training samples (for quick tests) |
| `--fp16` | flag | False | Enable mixed-precision training |
| `--study_name` | str | auto-generated | Optuna study name (for resuming) |
| `--seed` | int | 42 | Random seed |

### Output Structure

```
hpo_results/dnabert2/
└── PTBP1_HepG2_IDR_csv/
    ├── optuna_study.db          # Optuna SQLite database (resumable)
    ├── hpo_summary.json         # Best trial summary
    ├── trial_0/
    │   └── result.json          # Trial hyperparams + metrics
    ├── trial_1/
    │   └── result.json
    └── ...
```

---

## Local Test — `test_finetune_dnabert2.sh`

Runs **4 smoke tests** on a small data subset to verify the pipeline works before submitting SLURM jobs.

| Test | What it does |
|---|---|
| 1 | Fine-tune using `train.py` on CSV data (2 epochs, 2 folds) |
| 2 | HPO on CSV data (2 trials, 200 samples) |
| 3 | HPO on Koo data (2 trials, 200 samples) |
| 4 | Verify output artefacts exist |

### Usage

```bash
chmod +x test_finetune_dnabert2.sh
./test_finetune_dnabert2.sh
```

- Duration: ~5–10 min on GPU, ~20 min on CPU
- Uses QKI_K562_IDR (smallest dataset, ~4k total samples)
- Outputs go to `test_output/` (auto-cleaned on each run)

---

## Environment

**Conda environment:** `dnabert2`

```
Python 3.8
torch 2.3.1+cu121
transformers 4.46.3
optuna
scikit-learn
```

### Setup

```bash
conda activate dnabert2
pip install optuna    # if not already installed
```

---

## SLURM Submission

For full-scale runs on the cluster, see `generate_dnabert2_jobs.sh` in the parent directory. A typical HPO SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=hpo_dnabert2
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

conda activate dnabert2

python hpo_dnabert2.py \
    --rbp_name PTBP1_HepG2_IDR \
    --dataset csv \
    --n_trials 50 \
    --output_dir /path/to/output \
    --fp16
```
