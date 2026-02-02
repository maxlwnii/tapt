# Data Directory Structure Documentation

## Overview
The `data/` directory contains RNA-binding protein (RBP) CLIP-seq datasets for training and evaluating DNA language models. The data is organized into multiple subdirectories for different processing stages and use cases.

---

## Directory Organization

### 1. **clip_training_data_uhl/**
Raw FASTA files for RNA-binding protein (RBP) analysis with paired positive and negative sequences.

**Structure:**
```
{RBP}_{CELL_LINE}_IDR.positives.fa
{RBP}_{CELL_LINE}_IDR.negatives.fa
```

**RBPs included:**
- GTF2F1 (K562)
- HNRNPL (K562)
- HNRNPM (HepG2)
- ILF3 (HepG2)
- KHSRP (K562)
- MATR3 (K562)
- PTBP1 (HepG2)
- QKI (K562)

**File format:**
```fasta
>sequence_header
AUGCAUGCAUGCAUGC...
```

**Key characteristics:**
- Sequences contain RNA nucleotides (AUCG)
- Positive samples: regions where RBP binds
- Negative samples: background/non-binding regions
- Used for binary classification tasks

---

### 2. **eclip_koo/**
Pre-processed HDF5 datasets from Koo et al. for direct model evaluation.

**Files:**
- `{RBP}_K562_200.h5` (10 different RBPs)

**Format:** HDF5 with standardized 200bp sequences

**RBPs:**
- HNRNPK, PTBP1, PUM2, QKI, RBFOX2
- SF3B4, SRSF1, TARDBP, TIA1, U2AF1

**Usage:** Benchmark dataset for comparing model performance

---

### 3. **eval_clip_data/**
Analysis scripts and results for characterizing CLIP dataset properties.

#### **Scripts:**

**analyze_clip.py** - GC/AU content analysis
```python
def compute_gc_au(sequences):
    """Compute GC and AU percentages for sequences"""
    total_bases = 0
    gc_count = 0
    au_count = 0
    for seq in sequences:
        counts = Counter(seq)
        total_bases += len(seq)
        gc_count += counts.get('G', 0) + counts.get('C', 0)
        au_count += counts.get('A', 0) + counts.get('U', 0)
    
    gc_percent = (gc_count / total_bases) * 100
    au_percent = (au_count / total_bases) * 100
    return gc_percent, au_percent
```

**Key outputs:**
- GC/AU composition differences between positives and negatives
- Helps identify sequence bias in datasets
- Results saved in `clip_analysis_differences.csv`

**kmer_analysis.py** - K-mer frequency profiling
```python
def count_kmers_in_sequence(seq, k):
    """Count k-mer occurrences in sequence"""
    counts = Counter()
    n = len(seq)
    for i in range(n - k + 1):
        kmer = seq[i:i + k]
        if all(ch in 'ACGT' for ch in kmer):
            counts[kmer] += 1
    return counts
```

**Features:**
- Computes dinucleotide and trinucleotide frequencies
- Generates frequency distributions
- Outputs: `{RBP}_kmer_freqs.txt`

**analyze_cg_content.py** - CG-specific content analysis

#### **plots/** subdirectory
Contains analysis outputs:
- K-mer frequency tables for all RBPs
- Comparative statistics between positive/negative sets

---

### 4. **finetune_data_koo/**
Tokenized and preprocessed datasets ready for HuggingFace Transformers training.

**Structure per RBP:**
```
{RBP}/
├── dataset_dict.json      # Dataset metadata
├── train/                 # Training split
│   ├── data-*.arrow      # Memory-mapped data
│   ├── dataset_info.json
│   └── state.json
├── validation/            # Validation split
└── test/                  # Test split
```

**Format:** HuggingFace Arrow format for efficient loading

**Key features:**
- Pre-tokenized sequences
- Train/validation/test splits
- Memory-efficient arrow format
- Compatible with Transformers Trainer API

**Usage example:**
```python
from datasets import load_from_disk
dataset = load_from_disk("finetune_data_koo/GTF2F1_K562_IDR")
train_data = dataset['train']
```

---

### 5. **rel_data/**
Reference genomic data and overlap analysis between CLIP experiments.

**Files:**

**hg38.fa** - Human reference genome (GRCh38)
- Complete human genome sequence
- Used for extracting genomic coordinates

**merged_clip_regions.ext150.hg38.bed** - CLIP peak regions
- Merged peaks from multiple CLIP experiments
- Extended by 150bp on each side
- BED format: chr, start, end, metadata

**combined_sorted_idr.bed** - High-confidence IDR peaks
- IDR (Irreproducible Discovery Rate) thresholded peaks
- Sorted by chromosome and position
- Represents most reproducible binding sites

**common_variants.bed** - Genetic variants
- Common SNPs/variants in peak regions
- Used for variant effect analysis

**overlap.bed** - Peak overlap coordinates
```bed
chr1    1000    2000    peak_count:5
chr2    5000    6000    peak_count:3
```

**overlap.fa** - Sequences from overlapping peaks
- FASTA format
- Sequences extracted from overlap.bed coordinates
- Used for cross-RBP binding analysis

**Usage:**
```bash
# Extract sequences from BED file
bedtools getfasta -fi hg38.fa -bed overlap.bed -fo overlap.fa

# Find overlaps between datasets
bedtools intersect -a file1.bed -b file2.bed > overlap.bed
```

---

## Data Processing Workflow

### 1. Raw Data → Training Format

```
clip_training_data_uhl/*.fa 
    ↓ (parse FASTA)
finetune_data_koo/{RBP}/ 
    ↓ (tokenize + split)
HuggingFace Arrow format
```

### 2. Sequence Analysis Pipeline

```
FASTA files
    ↓
analyze_clip.py → GC/AU content
    ↓
kmer_analysis.py → K-mer frequencies
    ↓
plots/ (visualization)
```

### 3. Genomic Context Integration

```
BED coordinates + hg38.fa
    ↓ (bedtools)
overlap.fa sequences
    ↓
Analysis with rel_data/
```

---

## Key Statistics

**Dataset sizes (typical):**
- Positives per RBP: ~3,000-10,000 sequences
- Negatives per RBP: ~3,000-10,000 sequences
- Sequence length: 101-200bp
- Train/Val/Test split: 70/15/15

**File formats:**
- `.fa/.fasta`: Raw sequences
- `.h5`: HDF5 binary format
- `.arrow`: HuggingFace memory-mapped format
- `.bed`: Genomic coordinates
- `.csv`: Analysis results

---

## Usage Examples

### Loading FASTA data
```python
from Bio import SeqIO

def load_sequences(fasta_path):
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper().replace("U", "T")
        sequences.append(seq)
    return sequences

positives = load_sequences("clip_training_data_uhl/GTF2F1_K562_IDR.positives.fa")
```

### Loading HuggingFace datasets
```python
from datasets import load_from_disk

# Load pre-processed dataset
dataset = load_from_disk("finetune_data_koo/HNRNPL_K562_IDR")
print(f"Train size: {len(dataset['train'])}")
print(f"Sample: {dataset['train'][0]}")
```

### Loading HDF5 data
```python
import h5py

with h5py.File("eclip_koo/QKI_K562_200.h5", 'r') as f:
    X_train = f['X_train'][:]
    y_train = f['y_train'][:]
```

---

## Quality Control

**Dataset validation checks:**
1. Sequence length consistency
2. No ambiguous nucleotides (N's)
3. Balanced positive/negative ratios
4. No duplicate sequences
5. RNA → DNA conversion (U → T)

**Analysis outputs help identify:**
- Sequence composition bias
- K-mer enrichment in positives vs negatives
- GC content skew
- Dataset quality issues

---

## References

- IDR: Irreproducible Discovery Rate for peak calling
- CLIP-seq: Cross-Linking and ImmunoPrecipitation sequencing
- RBP: RNA-Binding Protein
- BED: Browser Extensible Data format
- Arrow: Apache Arrow columnar format
