# DNABERT2/evalEmbeddings

Embedding evaluation, layer analysis, and CNN/linear-probe benchmarking for DNABERT2 variants and baselines.

## Scope

This folder provides:
- CNN downstream benchmarks on DNABERT2 (and one-hot baseline) embeddings,
- frozen-embedding linear probing,
- layer search utilities,
- plotting and result-combination scripts for publication/reporting.

## Main Scripts

- `DNABERT2_CNN.py`
  - Main CNN benchmark runner.
  - Supports model variants including pretrained, TAPT checkpoints, random init, and one-hot baseline.
  - Supports fixed layer via `--best_layer` to bypass layer search.
  - Writes incremental results to `DNABERT2_CNN_results.csv`.

- `layer_search.py`
  - Shared utility to evaluate transformer layers and select best intermediate layer efficiently.

- `linear_probe_embedding_quality.py`
  - Frozen-embedding linear probe and embedding-quality comparisons.

- `plot_cnn_results_boxplots.py`
  - Produces per-model boxplots from CNN result CSVs.
  - Filters to datasets present across all model variants when configured.

- `plot_per_rbp_boxplots.py`
  - Boxplots from per-RBP linear probe metric tables.

- `combine_results.py`
  - Merges multiple run outputs and recomputes summaries + statistical tests.

## Submission and Run Scripts

- `run_dnabert2_cnn.slurm`
- `run_dnabert2_cnn_diff_cells_last.slurm`
- `run_dnabert2_cnn_onehot_and_random_remaining.slurm`
- `submit_dnabert2_cnn_diff_cells_last.sh`
  - Main CNN campaign launch and targeted rerun utilities.

- `run_linear_probe.sh`
- `run_linear_probe_embedding_quality.slurm`
- `smoke_test_linear_probe.sh`
  - Linear probe run and smoke-test helpers.

- `run_clip_dnabert2.sh`
- `run_embedding_comparison.slurm`
  - Additional benchmark workflows for specific subsets.

## Data Contract

Expected task folder structure under each data root:
- `<root>/<task>/train.csv`
- `<root>/<task>/dev.csv`
- `<root>/<task>/test.csv`

Expected columns:
- sequence-like column (auto-detected from common aliases),
- binary label column.

## Typical Commands

### Run CNN benchmark on selected variants

```bash
python DNABERT2/evalEmbeddings/DNABERT2_CNN.py \
  --data_roots \
    /gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/DNABERT2/data \
    /gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/data/finetune_data_koo \
  --variants dnabert2_pretrained dnabert2_tapt dnabert2_tapt_v3 dnabert2_random one_hot \
  --output_dir /gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/DNABERT2/evalEmbeddings/results/cnn_all_variants
```

### Force last layer (no layer search)

```bash
python DNABERT2/evalEmbeddings/DNABERT2_CNN.py \
  --best_layer 12 \
  --variants dnabert2_random one_hot
```

### Plot comparable model boxplots from CNN outputs

```bash
python DNABERT2/evalEmbeddings/plot_cnn_results_boxplots.py \
  --input_dirs \
    DNABERT2/evalEmbeddings/results/cnn_all_variants \
    DNABERT2/evalEmbeddings/results/cnn_diff_cells_last_layer
```

## Outputs

Default root:
- `DNABERT2/evalEmbeddings/results/`

Common artifacts:
- `DNABERT2_CNN_results.csv`,
- summary CSVs from aggregation scripts,
- per-metric PNG plots,
- cached embeddings/layer-search JSONs under per-run cache folders.

## Notes

- TensorFlow and PyTorch are both used in parts of the pipeline.
- For cluster runs, ensure environment modules and conda envs match script expectations.
- To resume safely, keep output CSVs and per-task cache/results directories intact.
