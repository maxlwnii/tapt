# linear probe

This directory contains linear probing utilities for embedding-based RBP classification experiments, with emphasis on multi-dataset evaluation and reproducible HPC submission.

## What This Folder Is For

Use this folder to:
- prepare CSV splits for diff-cell FASTA datasets,
- run frozen-embedding linear probe experiments across selected data roots,
- submit model/layer jobs on SLURM.

Primary scope in current scripts:
- fixed-layer probing (layer 6 and last),
- no layer search in the main multi-dataset pipeline,
- support for DNABERT2/LAMAR-compatible model sets through shared logic in `linear_probe_cross_length`.

## Important Files

- `linear_probe_all_datasets.py`
  - Main runner for multi-root linear probing.
  - Discovers tasks from CSV folders containing `train.csv`, `dev.csv`, `test.csv`.
  - Loads helper logic from `../linear_probe_cross_length/linear_probe_cross_length.py`.
  - Produces per-task metrics, summary tables, and optional statistical tests.

- `prepare_diff_cells_splits.py`
  - Converts paired FASTA files (`*.positives.fa`, `*.negatives.fa`) into stratified CSV splits.
  - Writes `train.csv`, `dev.csv`, and `test.csv` per task under the configured output root.

- `run_linear_probe_single.slurm`
  - Single-job SLURM template used by submit scripts (model/layer split style).

- `submit_linear_probe_jobs.sh`
  - Convenience submitter for predefined model and layer combinations.

- `run_prepare_diff_cells_splits.slurm`
  - SLURM wrapper to run split preparation in cluster environment.

## Data Contract

Expected per-task directory format:
- `<root>/<task_name>/train.csv`
- `<root>/<task_name>/dev.csv`
- `<root>/<task_name>/test.csv`

Expected CSV columns:
- sequence column: one of `sequence`, `seq`, `text`, `input`, `x`
- label column: one of `label`, `labels`, `target`, `y`

Labels are assumed binary and coercible to integers.

## Typical Workflows

### 1) Build diff-cells CSV splits

```bash
python "linear probe/prepare_diff_cells_splits.py" \
  --input_root /home/fr/fr_fr/fr_ml642/Thesis/data/diff_cells_data \
  --output_root /home/fr/fr_fr/fr_ml642/Thesis/data/diff_cells_data/splits_csv \
  --seed 42
```

### 2) Run multi-dataset linear probe

```bash
python "linear probe/linear_probe_all_datasets.py" \
  --data_roots \
    /home/fr/fr_fr/fr_ml642/Thesis/data/finetune_data_koo \
    /home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/data \
    /home/fr/fr_fr/fr_ml642/Thesis/data/diff_cells_data/splits_csv \
  --layers 6 last \
  --seed 42 \
  --output_dir "/home/fr/fr_fr/fr_ml642/Thesis/linear probe/results"
```

### 3) Submit batch jobs

```bash
bash "linear probe/submit_linear_probe_jobs.sh"
```

## Outputs

Default outputs are written under:
- `linear probe/results/`

Typical artifacts:
- per-task metric tables (CSV),
- summary tables (mean/std by model and layer),
- statistical test JSON (when enabled by the main pipeline),
- caches under `results/cache/`.

## Reproducibility Notes

- Seeded via `--seed`.
- Sequence normalization converts `U` to `T` where needed.
- Duplicate sequences are removed before training/evaluation.
