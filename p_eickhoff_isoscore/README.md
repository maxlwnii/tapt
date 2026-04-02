# p_eickhoff_isoscore

Unsupervised embedding-quality evaluation utilities used in this thesis, centered on IsoScore-family metrics and companion rank/spectrum diagnostics.

## Purpose

This folder is used to evaluate frozen embeddings without supervised classifier training, compare model variants, and export publication-ready figures.

It includes:
- task discovery across RBP datasets,
- embedding extraction and caching,
- isotropy/spectral metrics (IsoScore, RankMe, NESum, StableRank),
- sensitivity analyses and consolidated plots.

## Important Scripts

- `unsupervised_eval.py`
  - Main end-to-end pipeline.
  - Extracts embeddings, caches `.npy`, computes metrics, and writes plots/CSVs.
  - Supports last-layer and optional specific-layer variants configured in model specs.

- `run_unsupervised_eval.sh`
  - Standard launch script for selected model subsets.

- `run_unsupervised_eval_layers.sh`
  - Runs layer-oriented evaluation mode.

- `slurm_unsup_eval_last.sbatch`
- `slurm_unsup_eval_layers.sbatch`
  - SLURM templates for cluster execution of last-layer vs layer-based runs.

- `plot_from_summary.py`
  - Recreates plots from `summary.csv` exports without rerunning embedding extraction.

- `make_plots_from_summary.py`
  - Generates additional heatmap/bar/radar-style figures from summary-level metrics.

- `combine_unsupervised_eval_plots.py`
  - Produces combined comparison figures across two run directories.

## Data Inputs

Expected per-task CSV layout under each data root:
- `<root>/<task_name>/train.csv`
- `<root>/<task_name>/dev.csv`
- `<root>/<task_name>/test.csv`

Expected columns:
- sequence-like column (for nucleotide string),
- binary label column.

## Typical Commands

### Run unsupervised evaluation

```bash
bash p_eickhoff_isoscore/run_unsupervised_eval.sh
```

### Run layer-oriented evaluation

```bash
bash p_eickhoff_isoscore/run_unsupervised_eval_layers.sh
```

### Build plots from summary CSV only

```bash
python p_eickhoff_isoscore/plot_from_summary.py
```

## Outputs

Default outputs are written under:
- `p_eickhoff_isoscore/results/`

Common artifacts:
- `metrics.csv` (per model x task),
- `sensitivity.csv` (length-bin sensitivity metrics),
- `summary.csv` (aggregated means/std where produced),
- `plots/*.png`.

## Environment Notes

- Cluster scripts expect a Python environment with PyTorch, Transformers, and plotting stack installed.
- Some scripts expect `THESIS_ROOT` and `PYTHONPATH` to include the local LAMAR package.
- GPU is recommended for embedding extraction speed.

## Upstream Components

The bundled `IsoScore/` and `I-STAR/` directories contain upstream method implementations and references used by these evaluation scripts.
