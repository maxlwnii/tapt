# Intermediate Layer Search for DNA Transformer Embeddings

## Overview

The layer search functionality enables you to identify the optimal intermediate hidden layer of a transformer model for generating embeddings used in downstream RNA-binding protein (RBP) prediction tasks. This is particularly useful for optimizing models like TAPT LAMAR, where different layers may encode different levels of biological information.

## Files

- **layer_search.py**: Core module implementing the layer search algorithm
- **linear_probe_embedding_quality.py**: Main evaluation script with integrated layer search support

## Usage

### Basic Usage: Run Layer Search on TAPT LAMAR

```bash
python linear_probe_embedding_quality.py \
    --enable_layer_search \
    --layer_search_model tapt_lamar
```

This will:
1. Load the first RBP task alphabetically as a pilot
2. Extract embeddings from all hidden layers of the TAPT LAMAR model
3. Linear probe each layer with 5-fold cross-validation
4. Identify the layer with the best AUROC
5. Save results to the output directory

### Advanced Usage: Specify Pilot RBP

```bash
python linear_probe_embedding_quality.py \
    --enable_layer_search \
    --layer_search_pilot_rbp RBFOX2 \
    --layer_search_model tapt_lamar
```

This runs layer search on the RBFOX2 RBP task specifically.

### Use a Pre-determined Layer (Skip Search)

If you've already run layer search and know the best layer index, use:

```bash
python linear_probe_embedding_quality.py \
    --best_layer_override 6 \
    --layer_search_model tapt_lamar
```

This skips the expensive layer search and directly uses layer 6 for all embeddings.

### Run on Different Models

Layer search can be run on any of the available models:

```bash
# Search for best layer in base DNABERT2
python linear_probe_embedding_quality.py \
    --enable_layer_search \
    --layer_search_model base_dnabert2

# Search for best layer in pretrained LAMAR
python linear_probe_embedding_quality.py \
    --enable_layer_search \
    --layer_search_model pretrained_lamar
```

## How Layer Search Works

### Step 1: Load Pilot Data
- Selects one representative RBP task (default: first alphabetically)
- Merges train.csv, dev.csv, test.csv
- Optionally subsamples to `--max_samples_per_rbp`

### Step 2: Discover Layer Count
- Runs a dummy forward pass with `output_hidden_states=True`
- Counts total hidden states (1 embedding layer + N transformer blocks)

### Step 3: Extract All Layers (Single Pass)
- Loads the transformer model once
- Extracts embeddings from ALL layers in a single forward pass (efficient!)
- Mean-pools over non-padding and non-special tokens
- Caches results to avoid recomputation

### Step 4: Linear Probe Each Layer
- For each layer:
  - Runs 5-fold stratified cross-validation
  - Fits StandardScaler → LogisticRegression pipeline per fold
  - Computes mean AUROC across folds

### Step 5: Report Results
- Prints ranked table of layers by AUROC
- Saves results to `output_dir/layer_search.json`
- Saves AUROC-vs-layer line plot to `output_dir/layer_auroc_curve.png`

## Output

After layer search completes, you'll see:

```
=== Layer Search Complete ===
  Model   : /path/to/model
  Pilot   : RBFOX2
  Layers  : 13 total
  Best    : Layer 6 → AUROC = 0.8471
  Results : /output/layer_search.json
  Plot    : /output/layer_auroc_curve.png
```

### layer_search.json Format

```json
{
  "best_layer": 6,
  "best_auroc": 0.8471,
  "pilot_rbp": "RBFOX2",
  "model": "/path/to/tapt_lamar/checkpoint-98000",
  "layer_aurocs": {
    "0": 0.71,
    "1": 0.79,
    "2": 0.82,
    ...
    "6": 0.8471,
    ...
    "12": 0.79
  }
}
```

## Key Features

### Efficiency
- **Single forward pass**: Extracts all layers simultaneously, not N separate passes
- **Smart caching**: Embeddings are cached per model/layer/RBP combination
- **Lazy loading**: Skips recomputation if cache exists

### Robustness
- **Tokenizer fallback**: If local checkpoint lacks tokenizer, falls back to HF pre-trained
- **Degenerate layer detection**: Skips layers with zero-norm embeddings
- **Memory management**: Deletes model and clears CUDA cache after extraction

### Flexibility
- **Per-model search**: Run on base_dnabert2, tapt_lamar, or pretrained_lamar
- **Per-RBP pilots**: Choose any RBP or use first alphabetically
- **Override capability**: Skip search and use pre-determined layer

## Integration with Main Pipeline

After running layer search, the identified best layer is automatically used:

1. Layer search identifies best layer (e.g., layer 6)
2. Remaining RBPs are evaluated using layer 6 for that model
3. All other models (one_hot, base_dnabert2, etc.) continue to use their defaults

## Performance Considerations

### Time
- Layer search on a pilot RBP: ~5-30 minutes (depending on model size and dataset)
- Cache hit (subsequent runs): ~1 minute
- Main evaluation with best layer: Same as original, but using optimal layer

### Memory
- GPU: ~8-12 GB for extracting all layers simultaneously (vs N × 1-2 GB sequentially)
- Storage: ~100 MB-1 GB per model/RBP combination for cached embeddings

## Command-Line Arguments

**Layer search specific:**
- `--enable_layer_search`: Enable layer search (default: False)
- `--layer_search_pilot_rbp <name>`: RBP for layer search (default: first alphabetically)
- `--layer_search_model <name>`: Model to search (default: tapt_lamar)
- `--best_layer_override <int>`: Skip search, use this layer directly

**Existing arguments (also relevant):**
- `--max_length 512`: Max sequence length for tokenization
- `--batch_size 64`: Batch size for forward passes
- `--num_folds 5`: K-fold cross-validation folds
- `--max_samples_per_rbp 0`: Cap samples per RBP (0 = no cap)
- `--seed 42`: Random seed for reproducibility

## Example Workflow

```bash
# Step 1: Run layer search on TAPT LAMAR with RBFOX2 pilot
python linear_probe_embedding_quality.py \
    --enable_layer_search \
    --layer_search_pilot_rbp RBFOX2 \
    --layer_search_model tapt_lamar \
    --output_dir results/layer_search_run

# Output: layer_search.json shows best_layer = 6

# Step 2: Evaluate all RBPs with the best layer
python linear_probe_embedding_quality.py \
    --best_layer_override 6 \
    --layer_search_model tapt_lamar \
    --output_dir results/main_evaluation

# Step 3: Compare with other models
python linear_probe_embedding_quality.py \
    --best_layer_override 6 \
    --layer_search_model tapt_lamar \
    --embedding_models base_dnabert2 tapt_lamar pretrained_lamar \
    --output_dir results/cross_model_comparison
```

## Troubleshooting

### Cache Issues
If you encounter stale cache:
```bash
rm -rf /path/to/cache_dir/layer_search
```
Then re-run layer search.

### Model Not Found
If you see `[WARN] Skipping tapt_lamar: local path not found`:
- Check that `--tapt_lamar_model` path is correct and accessible

### Out of Memory
If GPU OOM occurs during layer extraction:
- Reduce `--batch_size` (e.g., 32 instead of 64)
- Reduce `--max_length` if appropriate for your sequences
- Or use CPU (slower but lower memory): `--device cpu`

## Citation

If you use this layer search functionality, please cite the original DNABERT2 and relevant architecture papers.

## References

- DNABERT2: [https://github.com/zhihan1996/DNABERT2](https://github.com/zhihan1996/DNABERT2)
- Transformers library: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
