# DNA Language Models for RBP Binding Prediction

This repository contains implementations and evaluation pipelines for DNA/RNA language models applied to RNA-Binding Protein (RBP) binding site prediction tasks.

## Overview

This project compares three approaches for predicting RBP binding sites:
- **DNABERT2**: Transformer-based DNA language model with BPE tokenization
- **LAMAR**: ESM-2 based nucleotide language model with rotary position embeddings
- **Baseline**: One-hot encoding with CNN classifiers

## Repository Structure

```
.
├── data/                          # Data processing and analysis scripts
│   ├── eval_clip_data/           # Dataset analysis tools
│   ├── DATA_STRUCTURE.md         # Data documentation
│   └── *.py                      # Analysis scripts
│
├── DNBERT2/                       # DNABERT2 implementation
│   ├── train.py                  # Training script
│   ├── convert_fasta_to_csv.py   # Data preprocessing
│   ├── visualisation.py          # Model interpretation
│   ├── plot_eval_scatter.py      # Results visualization
│   ├── generate_dnabert2_jobs.sh # SLURM job generator
│   ├── environment.yml           # Conda environment
│   └── DNABERT2_STRUCTURE.md     # Detailed documentation
│
└── LAMAR/                         # LAMAR implementation
    ├── LAMAR/                    # Core model code
    │   └── modeling_nucESM2.py   # ESM-2 architecture
    ├── evalEmbeddings/           # Evaluation framework
    │   ├── LAMAR_CNN_clip_data.py
    │   ├── OneHot_CNN_clip_data.py
    │   ├── depMaps/              # Attribution analysis
    │   └── isoscore/             # Embedding quality metrics
    ├── finetune_scripts/         # Fine-tuning pipelines
    │   └── finetune_rbp.py
    ├── preprocess/               # Data preprocessing
    │   └── preprocess.py
    ├── visualisation/            # Model interpretation
    ├── environment_finetune_fixed.yml
    └── LAMAR_STRUCTURE.md        # Detailed documentation
```

## Features

### DNABERT2
- Fine-tuning with LoRA support
- K-fold cross-validation
- Early stopping
- Random initialization baselines
- Multiple interpretation methods:
  - In-silico mutagenesis
  - Gradient-based saliency
  - Integrated gradients
  - Sliding window analysis

### LAMAR
- ESM-2 based architecture with rotary embeddings
- Embedding extraction + CNN evaluation
- Full fine-tuning with encoder freezing option
- Task-Adaptive Pre-Training (TAPT) support
- IsoScore embedding quality metrics
- Dependency map analysis

### Data Analysis
- GC/AU content analysis
- K-mer frequency profiling
- Sequence composition analysis
- Dataset quality control

## Installation

### DNABERT2 Environment
```bash
conda env create -f DNBERT2/environment.yml
conda activate dnabert2
```

### LAMAR Environment
```bash
conda env create -f LAMAR/environment_finetune_fixed.yml
conda activate lamar
```

## Usage

### DNABERT2 Training
```bash
cd DNBERT2
python train.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --data_path ./data/RBP_NAME \
    --model_max_length 512 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 5 \
    --learning_rate 3e-5 \
    --output_dir ./output/RBP_NAME \
    --metric_for_best_model eval_auprc
```

### LAMAR Evaluation
```bash
cd LAMAR/evalEmbeddings
python LAMAR_CNN_clip_data.py
```

### LAMAR Fine-tuning
```bash
cd LAMAR/finetune_scripts
python finetune_rbp.py \
    --rbp_name RBP_NAME \
    --data_path /path/to/data \
    --output_dir ./models/RBP_NAME \
    --pretrain_path /path/to/checkpoint.safetensors \
    --epochs 10 \
    --batch_size 4 \
    --lr 3e-5 \
    --freeze_encoder \
    --cv_folds 5
```

## Data Format

### Input Data
- **FASTA files**: RNA sequences with U→T conversion
- **CSV files**: `sequence,label` format for DNABERT2
- **HDF5 files**: Pre-processed binary format
- **BED files**: Genomic coordinates for peak regions

### Expected Directory Structure
```
data/
├── clip_training_data_uhl/     # Raw FASTA files
│   ├── {RBP}.positives.fa
│   └── {RBP}.negatives.fa
├── eclip_koo/                  # HDF5 datasets
└── finetune_data_koo/          # Tokenized HuggingFace datasets
```

## Model Performance

### Metrics
- AUC-ROC: Area under receiver operating characteristic curve
- AUC-PRC: Area under precision-recall curve (primary metric)
- Accuracy, F1, Precision, Recall
- Matthews Correlation Coefficient

## RBPs Evaluated

Eight RNA-binding proteins across K562 and HepG2 cell lines:
- GTF2F1 (K562)
- HNRNPL (K562)
- HNRNPM (HepG2)
- ILF3 (HepG2)
- KHSRP (K562)
- MATR3 (K562)
- PTBP1 (HepG2)
- QKI (K562)

## Documentation

Detailed documentation for each component:
- **[data/DATA_STRUCTURE.md](data/DATA_STRUCTURE.md)**: Complete data organization guide
- **[DNBERT2/DNABERT2_STRUCTURE.md](DNBERT2/DNABERT2_STRUCTURE.md)**: DNABERT2 implementation details
- **[LAMAR/LAMAR_STRUCTURE.md](LAMAR/LAMAR_STRUCTURE.md)**: LAMAR architecture and usage

## Requirements

### Core Dependencies
- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- TensorFlow 2.12+ (for CNN baselines)
- scikit-learn
- biopython
- pandas, numpy
- matplotlib, seaborn

### Optional
- CUDA 11.8+ for GPU acceleration
- Flash Attention for efficient training
- SLURM for cluster job submission

## Contact

maximilianl.lewinfr@gmail.com

## Acknowledgments

- DNABERT2 model from zhihan1996/DNABERT-2-117M
- ESM-2 architecture adapted for nucleotide sequences
- CLIP-seq data processing based on Koo et al.
