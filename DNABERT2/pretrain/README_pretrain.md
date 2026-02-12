# DNABERT2 — Task-Adaptive Continued Pretraining on eCLIP Data

Continued pretraining (TAPT) of [DNABERT-2-117M](https://huggingface.co/zhihan1996/DNABERT-2-117M) on eCLIP genomic data with **region-specific masking**.

## Overview

Standard MLM masks a uniform 15 % of all tokens. This implementation masks eCLIP binding regions (high-confidence protein-RNA interaction sites) at a **higher rate** (20–25 %) and flanking regions at a **lower rate**, while guaranteeing that the **total masking percentage is always exactly 15 %** per sequence.

### How the Adaptive Masking Works

For each sequence in the batch:

1. Identify tokens overlapping eCLIP peaks (`eclip_token_starts` / `eclip_token_ends`).
2. Sample a desired eCLIP masking probability $p_e \sim \mathcal{U}(0.20,\ 0.25)$.
3. Solve for the flanking probability $p_f$ such that:

$$p_e \cdot N_{\text{eclip}} + p_f \cdot N_{\text{flank}} = 0.15 \cdot N_{\text{total}}$$

4. Clamp $p_f \geq 0.05$ (minimum flanking rate). If the clamped $p_f$ would break the invariant, re-derive $p_e$.
5. Apply standard 80/10/10 replacement strategy (MASK / random / keep).

### BPE Token Mapping

DNABERT2 uses Byte-Pair Encoding (BPE), so each token may represent multiple nucleotides. The tokenization function converts nucleotide-level `peak_start`/`peak_end` positions to **token-level** positions using the tokenizer's offset mapping.

## Directory Structure

```
DNABERT2/pretrain/
├── pretrain_dnabert2.py           # Main training script + data collator
├── pretrain_dnabert2_full.slurm   # SLURM batch job (A100, full dataset)
├── test_pretrain_local.sh         # Local smoke test (small GPU / CPU)
├── README_pretrain.md             # This file
├── models/                        # Output checkpoints
├── logs/                          # SLURM job logs
└── scripts/                       # Existing utility scripts
```

## Prerequisites

### Environment

Use the existing `lamar_fixed` conda env or create a compatible one:

```bash
# Existing environment (recommended)
conda activate lamar_fixed

# Or install the key packages:
pip install torch transformers datasets accelerate tensorboard
```

### Data

Preprocessed JSON files from `preprocess/`:

| File | Samples | Description |
|------|---------|-------------|
| `preprocessed_data_metadata_train.json` | ~657 K | Training set (512-length sequences) |
| `preprocessed_data_metadata_val.json` | ~13 K | Validation set |

Each sample has the format:
```json
{
  "sequence": "AAGCTTGCAA...",
  "seq_id": "chr7:8212822-8218663_window_3584_4096",
  "seq_len": 512,
  "eclip_regions": [
    {"peak_start": 194, "peak_end": 365, "genomic_start": 11773665, "genomic_end": 11773836}
  ]
}
```

## Quick Start

### 1. Run the Local Test (recommended first step)

```bash
cd /path/to/Thesis/DNABERT2/pretrain
chmod +x test_pretrain_local.sh
bash test_pretrain_local.sh
```

This runs three checks:
- **Test 1**: Adaptive masking — 20 training steps with eCLIP-aware masking
- **Test 2**: Standard masking — 20 training steps with uniform 15 % masking
- **Test 3**: Statistical verification that masking rates match targets

### 2. Submit the Full Training Job

```bash
sbatch pretrain_dnabert2_full.slurm
```

Monitor with:
```bash
squeue -u $USER
tail -f logs/dnabert2_tapt_*.out

# TensorBoard (from login node or local machine)
tensorboard --logdir models/dnabert2_tapt_eclip/logs
```

## Training Configuration (Full Job)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | `zhihan1996/DNABERT-2-117M` | 117 M parameters |
| Sequence length | 512 tokens | |
| Batch size | 32 × 2 accum = 64 effective | |
| Learning rate | 3e-5 | Cosine schedule with 5 % warmup |
| Precision | FP16 | |
| Early stopping | Patience 5 | Based on validation loss |
| Checkpoints | Every 2000 steps | Keep best 3 |
| Masking | Adaptive: 20–25 % eCLIP, dynamic flanking | Total = 15 % |

## Key Arguments

```bash
python pretrain_dnabert2.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --train_file /path/to/train.json \
    --validation_file /path/to/val.json \
    --output_dir ./models/my_run \
    --use_adaptive_masking \        # Enable eCLIP-aware masking
    --target_mlm_prob 0.15 \        # Total masking target (invariant)
    --eclip_mlm_lo 0.20 \           # eCLIP region lower bound
    --eclip_mlm_hi 0.25 \           # eCLIP region upper bound
    --min_flanking_prob 0.05 \      # Floor for flanking regions
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-5 \
    --fp16 \
    --gradient_checkpointing \      # Saves ~40% VRAM
    --early_stopping_patience 5 \
    --max_train_samples 1000        # For quick testing
```

Without `--use_adaptive_masking`, the script uses standard uniform 15 % MLM masking.

## Bugs Fixed in the Original LAMAR Collator

The provided `DataCollatorForLanguageModelingCustom` from LAMAR had several issues:

1. **No 15 % total masking guarantee** — eCLIP and flanking probabilities were set independently, causing the total masking rate to vary with peak coverage.
2. **Single random draw per batch** — `np.random.uniform()` was called once for the entire batch; all samples shared the same flanking rate.
3. **No BPE offset mapping** — LAMAR uses single-nucleotide tokens, so `peak_start`/`peak_end` map 1:1 to token indices. DNABERT2 uses BPE, requiring explicit nucleotide-to-token conversion.
4. **Padding tokens not excluded** — The original code built a `special_tokens_mask` but did not additionally mask padding positions, which could lead to masking `[PAD]` tokens.

All four issues are fixed in `EclipAdaptiveMLMCollator`.

## Output

After training, the output directory contains:

```
models/dnabert2_tapt_eclip/
├── config.json              # Model config
├── model.safetensors        # Trained weights
├── tokenizer.json           # Tokenizer
├── tokenizer_config.json
├── train_results.json       # Training metrics
├── eval_results.json        # Final evaluation metrics
├── logs/                    # TensorBoard logs
├── checkpoint-*/            # Intermediate checkpoints
└── trainer_state.json       # Trainer state for resuming
```

## Resuming Training

```bash
python pretrain_dnabert2.py \
    --resume_from_checkpoint models/dnabert2_tapt_eclip/checkpoint-4000 \
    ... (same args as original run)
```
