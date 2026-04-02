# finetune_cross

Cross-domain supervised fine-tuning for:
- cross-cell transfer,
- cross-length transfer,
across LAMAR and DNABERT2 variant families.

## Purpose

This folder runs fixed-hyperparameter fine-tuning experiments over precomputed pair lists and stores one `results.json` per pair.

Main entry point:
- `finetune_cross.py`

Supporting utilities include:
- batch SLURM launchers,
- result collection,
- plotting scripts for cross-cell and cross-length summaries.

## Core Script

- `finetune_cross.py`
  - Unified trainer for both model types:
    - LAMAR: random, pretrained, TAPT variants,
    - DNABERT2: pretrained, random, TAPT variants.
  - Uses fixed HPs (no HPO in this script).
  - Reads pair metadata from:
    - `data/cross_cell/valid_pairs.json`
    - `data/cross_length/valid_prefixes.json`
  - Writes per-pair output:
    - `<output_dir>/<pair_name>/results.json`

## Collection and Plotting

- `collect_results.py`
  - Recursively scans result folders and aggregates all `results.json` into a CSV summary.

- `plot_cross_cell_metrics.py`
  - Cross-cell performance visualizations across variants.

- `plot_cross_length_metrics.py`
  - Cross-length performance visualizations across variants.

## Submission Scripts

Main array launchers:
- `slurm_ft_lamar.sh`
- `slurm_ft_dnabert2.sh`

High-level orchestrators:
- `submit_all.sh`
  - Submits full cross-cell + cross-length arrays for baseline variant sets.

- `submit_cross_length_44_all.sh`
  - Submits deterministic subset (44 RBPs) for faster aligned comparisons.

Specialized single-pair scripts/checkpoints:
- `slurm_ft_db2_tapt_v4_28226_cross_cell.sbatch`
- `slurm_ft_db2_tapt_v4_28226_cross_length.sbatch`
- `slurm_ft_db2_tapt_v4_5132_cross_cell.sbatch`
- `slurm_ft_db2_tapt_v4_5132_cross_length.sbatch`
- `slurm_ft_dnabert2_tapt_v3_pair.sh`
- `slurm_ft_lamar_tapt_512_std_pair.sh`

## Typical Commands

### Run one pair locally

```bash
python finetune_cross/finetune_cross.py \
  --model_type dnabert2 \
  --experiment cross_cell \
  --pair_name HNRNPK_train_K562_test_HepG2_fixlen_101 \
  --use_random_init \
  --max_length 128 \
  --output_dir /home/fr/fr_fr/fr_ml642/Thesis/finetune_cross/results/cross_cell/dnabert2_random
```

### Collect all results

```bash
python finetune_cross/collect_results.py \
  --results_dir /home/fr/fr_fr/fr_ml642/Thesis/finetune_cross/results \
  --output /home/fr/fr_fr/fr_ml642/Thesis/finetune_cross/results_summary.csv
```

### Submit all major arrays

```bash
bash finetune_cross/submit_all.sh
```

## Output Layout

Top-level output root:
- `finetune_cross/results/`

Common structure:
- `results/<experiment>/<variant>/<pair_name>/results.json`

Where:
- `<experiment>` is `cross_cell` or `cross_length`
- `<variant>` follows `lamar_*` or `dnabert2_*`

## Operational Notes

- Existing completed runs are skipped by SLURM wrappers when `results.json` is already present.
- Max input length is selected by experiment in submit scripts (typically 128/256 for DNABERT2 in cross settings).
- Environment variable `THESIS_ROOT` is assumed in cluster scripts and should point to the project root.
