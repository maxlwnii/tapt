# Layer Search Quick Reference

## One-Liners

### Find optimal layer for TAPT LAMAR
```bash
cd /home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/evalEmbeddings
python linear_probe_embedding_quality.py --enable_layer_search
```

### Find optimal layer for TAPT LAMAR on specific RBP
```bash
python linear_probe_embedding_quality.py \
    --enable_layer_search \
    --layer_search_pilot_rbp RBFOX2
```

### Use best layer (6) for full evaluation
```bash
python linear_probe_embedding_quality.py \
    --best_layer_override 6 \
    --output_dir results/with_best_layer
```

### Compare models with optimal TAPT LAMAR layer
```bash
python linear_probe_embedding_quality.py \
    --best_layer_override 6 \
    --layer_search_model tapt_lamar \
    --embedding_models one_hot base_dnabert2 tapt_lamar pretrained_lamar
```

### Fast test run with limited data
```bash
python linear_probe_embedding_quality.py \
    --enable_layer_search \
    --max_rbps 1 \
    --max_samples_per_rbp 100 \
    --skip_plots
```

## Output Files

| File | Purpose |
|------|---------|
| `layer_search.json` | Best layer index and per-layer AUROC scores |
| `layer_auroc_curve.png` | Line plot of AUROC across all layers |
| `per_rbp_metrics.csv` | Per-RBP evaluation metrics |
| `summary_metrics.csv` | Summary statistics across RBPs |
| `embedding_health_stats.csv` | Embedding norm and cosine similarity stats |

## Key Numbers

- **Layer count**: Usually 1 (embeddings) + 12 (blocks) = 13 total (DNABERT2-117M)
- **k-fold CV**: 5 splits (default)
- **Linear probe**: LogisticRegression with StandardScaler
- **Metric**: AUROC (Area Under ROC Curve)

## Performance Guide

| Setting | Impact | Recommendation |
|---------|--------|-----------------|
| `--batch_size 64` | Memory, speed | Start here |
| `--batch_size 32` | Lower memory, slower | If OOM |
| `--num_folds 5` | More robust, slower | Keep at 5 |
| `--max_samples_per_rbp 0` | Full data | Recommended |
| `--max_samples_per_rbp 1000` | Faster, less data | Quick tests |

## Typical Workflow

```bash
# 1. Find best layer (takes ~30 min with full data)
python linear_probe_embedding_quality.py \
    --enable_layer_search \
    --layer_search_model tapt_lamar

# 2. Check the result
cat results/linear_probe_embedding_quality/layer_search.json | grep best_layer

# 3. Use that layer for full evaluation (e.g., layer 6)
python linear_probe_embedding_quality.py \
    --best_layer_override 6 \
    --output_dir results/final_evaluation
```

## Caching Directory Structure

```
cache/
└── layer_search/
    └── model_<hash>/
        ├── layer_00/
        │   ├── RBFOX2.npy
        │   ├── RBM15.npy
        │   └── ...
        ├── layer_01/
        │   ├── RBFOX2.npy
        │   └── ...
        └── ...
```

Each `.npy` file is shape `(N, hidden_dim)` where N is sample count.

## Example Results

Expected output format:

```
=== Layer Search Complete ===
  Model   : /home/fr/fr_fr/fr_ml642/Thesis/LAMAR/src/pretrain/saving_model/tapt_lamar/checkpoint-98000
  Pilot   : RBFOX2
  Layers  : 13 total
  Best    : Layer 6 → AUROC = 0.8471
  Results : /home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/evalEmbeddings/results/linear_probe_embedding_quality/layer_search.json
  Plot    : /home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/evalEmbeddings/results/linear_probe_embedding_quality/layer_auroc_curve.png

=== Summary (mean ± std across RBPs) ===
Model                AUROC         F1        AUPRC
base_dnabert2        0.7650       0.6234    0.5123
tapt_lamar           0.8471       0.7563    0.6789
pretrained_lamar     0.8234       0.7234    0.6456
```

## Interpretation

- **Best layer**: Often middle layers (5-7 for 13-layer models) capture optimal features
- **Near-best layers**: Often within 1-2% of best AUROC
- **Embedding layer (0)**: Usually underperforms
- **Final layers (11-12)**: Often underperform (overly specific to pretraining task)

## Further Customization

### Run on custom data root
```bash
python linear_probe_embedding_quality.py \
    --data_roots /path/to/custom/data /another/path \
    --enable_layer_search
```

### Save to custom output directory
```bash
python linear_probe_embedding_quality.py \
    --output_dir /custom/output/path \
    --cache_dir /custom/cache/path \
    --enable_layer_search
```

### Different random seed (for reproducibility)
```bash
python linear_probe_embedding_quality.py \
    --seed 123 \
    --enable_layer_search
```
