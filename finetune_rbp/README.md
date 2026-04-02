# finetune_rbp

Per-RBP supervised fine-tuning workflows for LAMAR and DNABERT2 variants, including full-run submitters, missing-task recovery, and plotting utilities.

## Purpose

This folder handles RBP-level finetuning where each `(variant, rbp)` pair writes one `results.json`.

It supports:
- classic 18-RBP benchmark sets (KOO + IDR),
- expanded all-dataset runs (including diff-cells splits),
- resume/missing-only workflows for long cluster campaigns.

## Key Scripts

- `finetune_rbp.py`
  - Main trainer for one variant on one RBP.
  - Auto-detects dataset root from RBP naming or accepts explicit `--data_root`.
  - Saves outputs to `<output_dir>/<variant>/<rbp_name>/results.json`.

- `finetune_all_rbps.py`
  - Wrapper to discover tasks across one or more roots and invoke `finetune_rbp.py` repeatedly.
  - Supports `--only_missing` and `--completed_run_roots` for robust resume behavior.

- `plot_rbp_metrics.py`
  - Aggregates per-RBP metrics and creates boxplots across variants.

## Submit/Resume Tooling

- `slurm_ft_lamar.sh`
  - Runs one LAMAR variant across the 18 canonical RBPs.

- `slurm_ft_dnabert2.sh`
  - Runs one DNABERT2 variant across the 18 canonical RBPs.

- `slurm_ft_single_rbp.sh`
  - Runs exactly one `(variant, rbp)` task; used by missing-job recovery.

- `submit_all.sh`
  - Submits one job per selected variant (classic set mode).

- `submit_missing_rbps.sh`
  - Submits only missing `(variant, rbp)` pairs by checking `results.json` existence.

- `submit_all_new_datasets.sh`
  - Starts all-dataset campaigns into a new timestamped run root.

- `slurm_ft_all_datasets_lamar.sh`
- `slurm_ft_all_datasets_dnabert2.sh`
  - Cluster wrappers for all-dataset discovered task runs.

## Data Roots

Commonly used roots:
- `/home/fr/fr_fr/fr_ml642/Thesis/data/finetune_data_koo`
- `/home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/data`
- `/home/fr/fr_fr/fr_ml642/Thesis/data/diff_cells_data/splits_csv`

Expected per-task structure:
- `<root>/<task>/train.csv`
- `<root>/<task>/dev.csv`
- `<root>/<task>/test.csv`

## Typical Commands

### Run one variant on one RBP locally

```bash
python finetune_rbp/finetune_rbp.py \
  --variant dnabert2_tapt_v3 \
  --rbp_name GTF2F1_K562_IDR \
  --output_dir /home/fr/fr_fr/fr_ml642/Thesis/finetune_rbp/results/rbp
```

### Run one variant across discovered tasks

```bash
python finetune_rbp/finetune_all_rbps.py \
  --variant lamar_pretrained \
  --data_roots \
    /home/fr/fr_fr/fr_ml642/Thesis/data/finetune_data_koo \
    /home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/data \
    /home/fr/fr_fr/fr_ml642/Thesis/data/diff_cells_data/splits_csv \
  --output_root /home/fr/fr_fr/fr_ml642/Thesis/finetune_rbp/results/rbp_all \
  --only_missing \
  --continue_on_error
```

### Submit missing tasks only

```bash
bash finetune_rbp/submit_missing_rbps.sh --filter dnabert2
```

## Output Layout

Canonical:
- `finetune_rbp/results/rbp/<variant>/<rbp>/results.json`

All-dataset timestamped runs:
- `finetune_rbp/results/<run_tag>/<source>/<variant>/<rbp>/results.json`

## Notes

- Cluster scripts assume `THESIS_ROOT=/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis` unless overridden.
- `submit_missing_rbps.sh` is the safest way to backfill interrupted campaigns.
- Plotting script can read both JSON run trees and selected CSV exports.
