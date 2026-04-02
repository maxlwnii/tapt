# LAMAR/evalEmbeddings

Embedding evaluation and downstream probing/CNN benchmarking for LAMAR variants.

## Scope

This folder supports three major evaluation modes:
- frozen-embedding linear probe evaluation,
- CNN downstream evaluation on per-task embeddings,
- plot/summary aggregation across multiple run directories.

It includes both local and SLURM workflows.

## Main Scripts

- `LAMAR_CNN.py`
  - CNN benchmarking for LAMAR variants.
  - Supports per-task layer search or fixed/last-layer operation.
  - Writes incremental results to `LAMAR_CNN_results.csv` in the chosen output directory.

- `linear_probe_lamar.py`
  - Frozen-embedding linear probing across tasks.
  - Supports multiple LAMAR checkpoints and optional pilot layer search.

- `combine_results.py`
  - Merges `per_rbp_metrics.csv` from multiple run roots.
  - Recomputes summary tables, Wilcoxon tests, and comparison plots.

- `plot_compare_lamar_best_layer_auroc.py`
  - Summarizes and visualizes best-layer search JSON logs.

- `OneHot_CNN_Rep.py`
- `OneHot_CNN_clip_data.py`
  - One-hot baseline CNN evaluations for direct model-family comparison.

## Job Scripts

- `run_lamar_cnn.slurm`
- `run_lamar_cnn_diff_cells_last.slurm`
- `submit_all_lamar_cnn.sh`
- `submit_lamar_cnn_diff_cells_last.sh`
  - CNN campaign launchers and submit wrappers.

- `run_linear_probe.sh`
- `run_linear_probe_lamar.slurm`
- `run_linear_probe_embedding_quality.slurm`
  - Linear probe launch paths.

- `run_lamar_clip_eval.sh`
- `run_onehot_clip_eval.sh`
- `run_eclip_256.sh`
  - CLIP/eCLIP-specific evaluation entrypoints.

## Data Expectations

Task folders under data roots should contain:
- `train.csv`
- `dev.csv`
- `test.csv`

CSV schema should provide sequence and binary label columns (with flexible column name detection in main scripts).

## Typical Commands

### Run LAMAR CNN evaluation

```bash
python LAMAR/evalEmbeddings/LAMAR_CNN.py \
  --data_roots \
    /gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/DNABERT2/data \
    /gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/data/finetune_data_koo \
  --output_dir /gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings/results/cnn_all_variants
```

### Run linear probe

```bash
bash LAMAR/evalEmbeddings/run_linear_probe.sh
```

### Submit full CNN variant set

```bash
bash LAMAR/evalEmbeddings/submit_all_lamar_cnn.sh
```

## Outputs

Common output roots:
- `LAMAR/evalEmbeddings/results/`
- `LAMAR/evalEmbeddings/plots/`

Frequent artifacts:
- variant/task metric CSVs,
- aggregated summaries,
- layer-search caches and JSON records,
- PNG figures for per-metric and model-comparison plots.

## Practical Notes

- Cluster scripts assume conda/venv environments already provisioned.
- Some scripts require `PYTHONPATH` to include local `LAMAR` package modules.
- For strict comparability, align data roots and layer mode (`search`, `last`, or fixed layer) across runs.
