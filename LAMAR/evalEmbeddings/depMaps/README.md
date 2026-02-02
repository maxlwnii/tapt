# LAMAR Dependency Maps - Compute and Visualize

Compute and visualize dependency maps for LAMAR finetuned models. Similar to the dependencies_DNALM project but for RBP binding prediction models.

## Overview

Dependency maps show which token positions are important for model predictions by measuring how predictions change when positions are perturbed.

### Analysis Types

1. **In-Silico Mutagenesis**: Single-position mutations showing effect of each base at each position
2. **Gradient-Based Saliency**: Position importance based on gradient magnitudes during backpropagation

## Usage

### Basic Usage

```bash
python compute_dep_maps.py \
    --model_path /path/to/finetuned/model \
    --sequence "ACGTACGTACGT..."
```

### With Sequence File (FASTA)

```bash
python compute_dep_maps.py \
    --model_path /path/to/finetuned/model \
    --seq_file sequences.fasta
```

### Specific Analysis Type

```bash
# Mutagenesis only
python compute_dep_maps.py \
    --model_path /path/to/model \
    --sequence "ACGTACGT..." \
    --analysis_type mutagenesis

# Saliency only
python compute_dep_maps.py \
    --model_path /path/to/model \
    --sequence "ACGTACGT..." \
    --analysis_type saliency

# Both (default)
python compute_dep_maps.py \
    --model_path /path/to/model \
    --sequence "ACGTACGT..." \
    --analysis_type both
```

### Custom Output Directory

```bash
python compute_dep_maps.py \
    --model_path /path/to/model \
    --sequence "ACGTACGT..." \
    --output_dir /custom/output/path
```

## Examples

### Analyze a finetuned RBP model

```bash
python compute_dep_maps.py \
    --model_path /home/fr/fr_fr/fr_ml642/Thesis/LAMAR/models/finetuned_10epochs/Pretrained/HNRNPM_HepG2_IDR \
    --sequence "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" \
    --analysis_type both
```

### Analyze from FASTA file

```bash
python compute_dep_maps.py \
    --model_path /home/fr/fr_fr/fr_ml642/Thesis/LAMAR/models/finetuned_10epochs/Pretrained/HNRNPM_HepG2_IDR \
    --seq_file my_sequence.fasta
```

## Output Files

Each analysis generates:

1. **mutagenesis_effects.csv** - Effect size for each base mutation at each position
2. **mutagenesis_heatmap.png** - Visualization of mutagenesis effects
3. **saliency_scores.csv** - Saliency score for each position
4. **saliency_plot.png** - Line plot of saliency scores
5. **analysis_summary.txt** - Summary of the analysis

All outputs are saved to a timestamped subdirectory.

## Requirements

- torch
- transformers
- datasets
- numpy
- pandas
- matplotlib
- seaborn

## Notes

- Sequences are automatically converted to uppercase and U→T
- GPU is used if available (much faster for gradient computation)
- Longer sequences will take longer to analyze
- Typical runtime: ~5-30 seconds depending on sequence length and analysis type
