"""
Task-Adaptive Continued Pretraining (TAPT) of DNABERT2 on eCLIP Data.

Implements region-specific masking where eCLIP binding regions receive higher
masking probability while guaranteeing an overall 15% masking rate per sequence
(matching standard MLM). Uses DNABERT2's BPE tokenizer with proper
nucleotide-to-token position mapping.

Data format (JSON array):
  [
    {
      "sequence": "AAGCTTGCAA...",
      "seq_id": "chr7:...",
      "seq_len": 512,
      "eclip_regions": [
        {"peak_start": 194, "peak_end": 365, "genomic_start": ..., "genomic_end": ...}
      ]
    },
    ...
  ]

Usage:
  python pretrain_dnabert2.py \
    --train_file /path/to/train.json \
    --validation_file /path/to/val.json \
    --output_dir ./models/dnabert2_tapt \
    --use_adaptive_masking \
    --fp16

Author: Maximilian Lewin
"""

import os
import sys
import json
import math
import logging
import argparse
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.models.bert.configuration_bert import BertConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TARGET_MLM_PROBABILITY = 0.15


# ═════════════════════════════════════════════════════════════════════
#  Custom Data Collators
# ═════════════════════════════════════════════════════════════════════

@dataclass
class EclipAdaptiveMLMCollator(DataCollatorForLanguageModeling):
    """
    MLM data collator with region-adaptive masking for eCLIP data.

    CRITICAL INVARIANT:
        The overall masking rate per sequence is always ``target_mlm_prob``
        (default 15 %).  eCLIP binding regions get a *higher* masking
        probability, and flanking/background regions get a *lower*
        probability, dynamically adjusted so that:

            p_eclip * N_eclip + p_flank * N_flank  =  target * N_total

    Masking strategy (standard BERT-style):
        80 % of masked tokens → replaced with [MASK]
        10 % of masked tokens → replaced with a random token
        10 % of masked tokens → kept unchanged

    The collator expects each example to carry two flat integer lists:
        eclip_token_starts  – start indices (token-level) of eCLIP peaks
        eclip_token_ends    – end   indices (token-level, exclusive)

    These lists are produced by the tokenisation function (see
    ``create_tokenize_fn``), which maps nucleotide positions from the
    preprocessed JSON to BPE token positions.

    Args:
        tokenizer:          HuggingFace tokenizer (must have mask_token).
        target_mlm_prob:    Target fraction of tokens to mask (default 0.15).
        eclip_mlm_range:    (lo, hi) for sampling the eCLIP masking prob.
        min_flanking_prob:  Floor for flanking masking probability.
    """

    target_mlm_prob: float = TARGET_MLM_PROBABILITY
    eclip_mlm_range: Tuple[float, float] = (0.20, 0.25)
    min_flanking_prob: float = 0.05

    # ── validation ────────────────────────────────────────────────
    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "Tokenizer is missing a mask_token, required for MLM."
            )
        if not self.mlm:
            raise ValueError("EclipAdaptiveMLMCollator requires mlm=True.")

    # ── public entry point ────────────────────────────────────────
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        1. Pop token-level eCLIP peak boundaries from each example.
        2. Pad the remaining fields (input_ids, attention_mask).
        3. Apply region-adaptive masking to produce labels.
        """
        # Extract BEFORE tokenizer.pad() – these are variable-length
        eclip_starts = [ex.pop("eclip_token_starts", []) for ex in examples]
        eclip_ends   = [ex.pop("eclip_token_ends",   []) for ex in examples]

        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch["input_ids"], batch["labels"] = self._mask_tokens(
            batch["input_ids"], eclip_starts, eclip_ends,
        )
        return batch

    # ── core masking logic ────────────────────────────────────────
    def _mask_tokens(
        self,
        inputs: torch.Tensor,
        eclip_starts_batch: List[List[int]],
        eclip_ends_batch:   List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a per-token probability matrix that guarantees *target_mlm_prob*
        total masking, then apply standard 80/10/10 replacement.
        """
        labels = inputs.clone()
        bs, seq_len = inputs.shape

        # ── special-token mask (includes padding) ─────────────────
        special_mask = torch.tensor(
            [
                self.tokenizer.get_special_tokens_mask(
                    ids, already_has_special_tokens=True
                )
                for ids in labels.tolist()
            ],
            dtype=torch.bool,
        )
        if self.tokenizer.pad_token_id is not None:
            special_mask = special_mask | inputs.eq(self.tokenizer.pad_token_id)

        # ── per-token probability matrix ──────────────────────────
        prob = torch.zeros(bs, seq_len)

        for i in range(bs):
            valid = ~special_mask[i]              # maskable positions
            n_total = valid.sum().item()
            if n_total == 0:
                continue

            # Boolean mask marking tokens in eCLIP peaks
            eclip = torch.zeros(seq_len, dtype=torch.bool)
            starts = eclip_starts_batch[i] if i < len(eclip_starts_batch) else []
            ends   = eclip_ends_batch[i]   if i < len(eclip_ends_batch)   else []
            for s, e in zip(starts, ends):
                s, e = max(0, int(s)), min(seq_len, int(e))
                if s < e:
                    eclip[s:e] = True

            eclip_valid = eclip & valid
            flank_valid = (~eclip) & valid
            n_eclip = eclip_valid.sum().item()
            n_flank = flank_valid.sum().item()

            # ── dynamic probability calculation ───────────────────
            # Solve:  p_e * n_eclip + p_f * n_flank = target * n_total
            if n_eclip == 0 or n_flank == 0:
                # No differentiation possible → uniform target
                p_e = self.target_mlm_prob
                p_f = self.target_mlm_prob
            else:
                p_e = np.random.uniform(*self.eclip_mlm_range)
                p_f = (self.target_mlm_prob * n_total - p_e * n_eclip) / n_flank

                # Clamp flanking prob to a reasonable floor
                if p_f < self.min_flanking_prob:
                    p_f = self.min_flanking_prob
                    # Re-derive eclip prob to maintain the 15 % invariant
                    p_e = (self.target_mlm_prob * n_total
                           - p_f * n_flank) / n_eclip

                p_e = float(np.clip(p_e, 0.0, 1.0))
                p_f = float(np.clip(p_f, 0.0, 1.0))

            prob[i, eclip_valid] = p_e
            prob[i, flank_valid] = p_f

        # Zero out special / padding tokens
        prob.masked_fill_(special_mask, 0.0)

        # ── sample masked positions ───────────────────────────────
        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100              # loss only on masked tokens

        # 80 % → [MASK]
        replace_mask = (
            torch.bernoulli(torch.full_like(prob, 0.8)).bool() & masked
        )
        inputs[replace_mask] = self.tokenizer.mask_token_id

        # 10 % → random token (half of the remaining 20 %)
        random_mask = (
            torch.bernoulli(torch.full_like(prob, 0.5)).bool()
            & masked
            & ~replace_mask
        )
        random_ids = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[random_mask] = random_ids[random_mask]

        # 10 % → keep original (no-op)
        return inputs, labels


class StandardMLMCollatorClean(DataCollatorForLanguageModeling):
    """
    Thin wrapper around the standard HuggingFace MLM collator that strips out
    extra eCLIP fields before padding (prevents tokenizer.pad() from choking
    on variable-length lists).
    """

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        for ex in examples:
            ex.pop("eclip_token_starts", None)
            ex.pop("eclip_token_ends", None)
        return super().__call__(examples)


# ═════════════════════════════════════════════════════════════════════
#  Tokenisation helpers — BPE-aware nucleotide → token mapping
# ═════════════════════════════════════════════════════════════════════

def _nuc_to_token_pos(
    offsets: List[Tuple[int, int]],
    nuc_start: int,
    nuc_end: int,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Map a nucleotide range ``[nuc_start, nuc_end)`` to token indices using the
    offset_mapping produced by a fast tokenizer.

    Special / padding tokens have offset ``(0, 0)`` and are skipped.

    Returns:
        (tok_start, tok_end) — inclusive start, exclusive end.
        Either may be ``None`` if no overlap was found.
    """
    tok_start, tok_end = None, None
    for idx, (cs, ce) in enumerate(offsets):
        if cs == ce:                        # special or padding token
            continue
        if tok_start is None and ce > nuc_start:
            tok_start = idx
        if cs < nuc_end:
            tok_end = idx + 1
    return tok_start, tok_end


def _manual_offsets(
    tokenizer,
    input_ids: List[int],
) -> List[Tuple[int, int]]:
    """
    Fallback for tokenizers that do not support ``return_offsets_mapping``.
    Decode each token to estimate its character span.
    """
    specials = {
        tokenizer.cls_token,
        tokenizer.sep_token,
        tokenizer.pad_token,
        tokenizer.unk_token,
        None,
    }
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    offsets: List[Tuple[int, int]] = []
    pos = 0
    for tok in tokens:
        if tok in specials:
            offsets.append((0, 0))
        else:
            decoded = tokenizer.convert_tokens_to_string([tok]).replace(" ", "")
            length = max(len(decoded), 1)
            offsets.append((pos, pos + length))
            pos += length
    return offsets


def _get_regions_for_example(raw_regions, idx: int) -> List[Dict[str, int]]:
    """
    Robustly extract eclip regions for one example regardless of whether
    ``datasets`` stored them as list-of-dicts (raw JSON) or as a columnar
    dict-of-lists (Arrow backend).
    """
    if raw_regions is None:
        return []

    # Case A: list of lists of dicts (raw JSON / JSONL)
    if isinstance(raw_regions, list):
        entry = raw_regions[idx] if idx < len(raw_regions) else []
        if isinstance(entry, list):
            return entry
        if isinstance(entry, dict):
            return [entry]
        return []

    # Case B: columnar dict-of-lists (Arrow)
    if isinstance(raw_regions, dict):
        starts = raw_regions.get("peak_start", [])
        ends   = raw_regions.get("peak_end",   [])
        if idx < len(starts):
            return [
                {"peak_start": s, "peak_end": e}
                for s, e in zip(starts[idx], ends[idx])
            ]
        return []

    return []


def create_tokenize_fn(tokenizer, max_length: int):
    """
    Return a **batched** tokenisation function suitable for ``datasets.map()``.

    What it does:
        1. Tokenize DNA sequences with padding / truncation.
        2. Convert nucleotide-level eCLIP peak positions to BPE token-level
           positions using the tokenizer's offset mapping.
        3. Store the token positions as two flat integer lists
           (``eclip_token_starts``, ``eclip_token_ends``) so that the
           downstream data collator can use them directly.
    """
    # Probe whether the tokenizer supports return_offsets_mapping
    try:
        probe = tokenizer("ATCG", return_offsets_mapping=True)
        use_offsets = "offset_mapping" in probe
    except Exception:
        use_offsets = False
    logger.info(f"Tokenizer supports fast offset_mapping: {use_offsets}")

    def _fn(examples):
        seqs = examples["sequence"]
        tok_kwargs = dict(
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        if use_offsets:
            tok_kwargs["return_offsets_mapping"] = True

        tokenized = tokenizer(seqs, **tok_kwargs)

        # Obtain character-to-token offset mapping
        if use_offsets:
            all_offsets = tokenized.pop("offset_mapping")
        else:
            all_offsets = [
                _manual_offsets(tokenizer, ids) for ids in tokenized["input_ids"]
            ]

        # Convert eclip_regions (nucleotide coords) → token coords
        raw_regions = examples.get("eclip_regions", None)
        batch_size = len(seqs)
        all_starts: List[List[int]] = []
        all_ends:   List[List[int]] = []

        for i in range(batch_size):
            tok_starts: List[int] = []
            tok_ends:   List[int] = []
            regions = _get_regions_for_example(raw_regions, i)
            for reg in regions:
                ns = reg["peak_start"]
                ne = reg["peak_end"]
                ts, te = _nuc_to_token_pos(all_offsets[i], ns, ne)
                if ts is not None and te is not None and ts < te:
                    tok_starts.append(ts)
                    tok_ends.append(te)
            all_starts.append(tok_starts)
            all_ends.append(tok_ends)

        tokenized["eclip_token_starts"] = all_starts
        tokenized["eclip_token_ends"]   = all_ends
        return tokenized

    return _fn


# ═════════════════════════════════════════════════════════════════════
#  Argument parsing
# ═════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="DNABERT2 Task-Adaptive Continued Pretraining on eCLIP data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ─────────────────────────────────────────────────────
    g = p.add_argument_group("Model")
    g.add_argument(
        "--model_name_or_path", type=str, default="zhihan1996/DNABERT-2-117M",
        help="HuggingFace model name or local checkpoint path",
    )
    g.add_argument(
        "--max_seq_length", type=int, default=512,
        help="Maximum token sequence length (including special tokens)",
    )

    # ── Data ──────────────────────────────────────────────────────
    g = p.add_argument_group("Data")
    g.add_argument("--train_file", type=str, required=True,
                   help="Path to training JSON file")
    g.add_argument("--validation_file", type=str, default=None,
                   help="Path to validation JSON file")
    g.add_argument("--max_train_samples", type=int, default=None,
                   help="Limit training samples (for testing)")
    g.add_argument("--max_eval_samples", type=int, default=None,
                   help="Limit evaluation samples (for testing)")
    g.add_argument("--preprocessing_num_workers", type=int, default=4,
                   help="Workers for dataset tokenisation")

    # ── Masking ───────────────────────────────────────────────────
    g = p.add_argument_group("Masking")
    g.add_argument("--use_adaptive_masking", action="store_true",
                   help="Use eCLIP-adaptive masking (default: standard 15%% MLM)")
    g.add_argument("--target_mlm_prob", type=float, default=0.15,
                   help="Target overall masking probability")
    g.add_argument("--eclip_mlm_lo", type=float, default=0.20,
                   help="Lower bound of eCLIP region masking probability")
    g.add_argument("--eclip_mlm_hi", type=float, default=0.25,
                   help="Upper bound of eCLIP region masking probability")
    g.add_argument("--min_flanking_prob", type=float, default=0.05,
                   help="Minimum flanking region masking probability")

    # ── Training ──────────────────────────────────────────────────
    g = p.add_argument_group("Training")
    g.add_argument("--output_dir", type=str, default="./models/dnabert2_tapt",
                   help="Directory for checkpoints and final model")
    g.add_argument("--num_train_epochs", type=int, default=10)
    g.add_argument("--max_steps", type=int, default=-1,
                   help="Max training steps (overrides epochs when > 0)")
    g.add_argument("--per_device_train_batch_size", type=int, default=16)
    g.add_argument("--per_device_eval_batch_size", type=int, default=32)
    g.add_argument("--gradient_accumulation_steps", type=int, default=16)
    g.add_argument("--learning_rate", type=float, default=5e-5)
    g.add_argument("--weight_decay", type=float, default=0.01)
    g.add_argument("--warmup_ratio", type=float, default=0.06)
    g.add_argument("--lr_scheduler_type", type=str, default="cosine")
    g.add_argument("--max_grad_norm", type=float, default=5.0)
    g.add_argument("--optim", type=str, default="adamw_torch")

    # ── Early stopping ────────────────────────────────────────────
    g = p.add_argument_group("Early stopping")
    g.add_argument("--early_stopping_patience", type=int, default=5,
                   help="Evaluation rounds without improvement before stopping")

    # ── Checkpointing / logging ───────────────────────────────────
    g = p.add_argument_group("Checkpointing")
    g.add_argument("--eval_strategy", type=str, default="epoch",
                   choices=["epoch", "steps", "no"],
                   help="Evaluation strategy: 'epoch' (recommended) or 'steps'")
    g.add_argument("--save_steps",    type=int, default=1000,
                   help="Save checkpoint every N steps (only used when eval_strategy=steps)")
    g.add_argument("--eval_steps",    type=int, default=1000,
                   help="Evaluate every N steps (only used when eval_strategy=steps)")
    g.add_argument("--logging_steps", type=int, default=50)
    g.add_argument("--save_total_limit", type=int, default=5)

    # ── Hardware ──────────────────────────────────────────────────
    g = p.add_argument_group("Hardware")
    g.add_argument("--fp16", action="store_true")
    g.add_argument("--bf16", action="store_true")
    g.add_argument("--gradient_checkpointing", action="store_true",
                   help="Trade compute for memory (slower but lower VRAM)")
    g.add_argument("--dataloader_num_workers", type=int, default=4)
    g.add_argument("--dataloader_pin_memory", action="store_true")

    # ── Misc ──────────────────────────────────────────────────────
    g = p.add_argument_group("Misc")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--report_to", type=str, default="tensorboard",
                   choices=["tensorboard", "wandb", "none"])
    g.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to checkpoint directory to resume from")

    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)

    # ── directories ───────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    #  Tokenizer
    # ══════════════════════════════════════════════════════════════
    logger.info(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        model_max_length=args.max_seq_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("pad_token not set → using eos_token as pad_token")

    logger.info(
        f"  vocab_size={len(tokenizer)}  "
        f"pad_id={tokenizer.pad_token_id}  "
        f"mask_id={tokenizer.mask_token_id}  "
        f"max_length={tokenizer.model_max_length}"
    )

    # ══════════════════════════════════════════════════════════════
    #  Model
    # ══════════════════════════════════════════════════════════════
    logger.info(f"Loading model: {args.model_name_or_path}")

    # Load config via AutoConfig so trust_remote_code actually takes effect.
    # (BertConfig.from_pretrained silently ignores trust_remote_code.)
    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    logger.info(f"  Config loaded: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")

    # Load model with explicit config
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        trust_remote_code=True,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"  Parameters: {n_params:.1f} M")

    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            logger.info("  Gradient checkpointing: enabled (standard API)")
        except (ValueError, Exception) as e:
            logger.warning(
                f"Standard gradient_checkpointing_enable() failed ({e}). "
                "Applying manual gradient checkpointing via torch.utils.checkpoint."
            )
            # DNABERT2 / MosaicBERT: manually set flag on encoder layers
            gc_set = False
            for module in model.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = True
                    gc_set = True
            if not gc_set:
                # Wrap each encoder layer's forward with checkpoint
                import torch.utils.checkpoint as ckpt
                encoder = getattr(model, "bert", model)
                encoder = getattr(encoder, "encoder", encoder)
                if hasattr(encoder, "layer"):
                    for layer in encoder.layer:
                        orig_forward = layer.forward
                        def _make_ckpt_forward(fwd):
                            def _ckpt_forward(*args, **kwargs):
                                def create_custom_forward(module):
                                    def custom_forward(*inputs):
                                        return module(*inputs)
                                    return custom_forward
                                return ckpt.checkpoint(create_custom_forward(fwd.__self__), *args, use_reentrant=False)
                            return _ckpt_forward
                        layer.forward = _make_ckpt_forward(orig_forward)
                    logger.info(f"  Gradient checkpointing: enabled (manual, {len(encoder.layer)} layers)")
                else:
                    logger.warning(
                        "Could not enable gradient checkpointing for this model. "
                        "Continuing without it (higher memory usage)."
                    )

    # ══════════════════════════════════════════════════════════════
    #  Data
    # ══════════════════════════════════════════════════════════════
    tokenize_fn = create_tokenize_fn(tokenizer, args.max_seq_length)

    # ── DDP guard: only rank 0 does data loading & tokenization ──
    # In DDP, all ranks racing to load/map/cache the same dataset
    # causes cache corruption (.incomplete arrow files).  Let rank 0
    # build the cache; other ranks wait, then load from cache.
    import glob, shutil
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    # Initialize process group early if running under torchrun/DDP
    if is_distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    if local_rank == 0:
        # Clean up any incomplete/corrupted HF dataset cache entries
        for p in glob.glob(os.path.join(os.path.expanduser("~"), ".cache/huggingface/datasets/**/*.incomplete"), recursive=True):
            incomplete_dir = p if os.path.isdir(p) else os.path.dirname(p)
            logger.warning(f"Removing incomplete cache entry: {incomplete_dir}")
            shutil.rmtree(incomplete_dir, ignore_errors=True)

    # Barrier: if DDP, wait for rank 0 to finish cache cleanup
    if is_distributed:
        torch.distributed.barrier()

    # ── training set ──────────────────────────────────────────────
    logger.info(f"Loading training data: {args.train_file}")
    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
        logger.info(f"  Truncated to {len(train_ds)} training samples")

    remove_cols = list(train_ds.column_names)
    # Use num_proc only on rank 0; other ranks load from cache
    map_num_proc = args.preprocessing_num_workers if local_rank == 0 else 1
    train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        num_proc=map_num_proc,
        remove_columns=remove_cols,
        desc="Tokenizing train",
    )
    logger.info(f"  Train dataset: {len(train_ds)} examples")

    # Barrier: ensure all ranks have the tokenized dataset
    if is_distributed:
        torch.distributed.barrier()

    # ── validation set ────────────────────────────────────────────
    val_ds = None
    if args.validation_file:
        logger.info(f"Loading validation data: {args.validation_file}")
        val_ds = load_dataset("json", data_files=args.validation_file, split="train")
        if args.max_eval_samples is not None:
            val_ds = val_ds.select(range(min(args.max_eval_samples, len(val_ds))))
            logger.info(f"  Truncated to {len(val_ds)} eval samples")

        val_remove = list(val_ds.column_names)
        val_ds = val_ds.map(
            tokenize_fn,
            batched=True,
            num_proc=map_num_proc,
            remove_columns=val_remove,
            desc="Tokenizing val",
        )
        logger.info(f"  Validation dataset: {len(val_ds)} examples")

    # Final barrier after all data loading
    if is_distributed:
        torch.distributed.barrier()

    # ── quick sanity check ────────────────────────────────────────
    sample = train_ds[0]
    logger.info(
        f"  Sample columns: {list(sample.keys())}  |  "
        f"input_ids len: {len(sample['input_ids'])}  |  "
        f"eclip peaks: {len(sample.get('eclip_token_starts', []))}"
    )

    # ══════════════════════════════════════════════════════════════
    #  Data collator
    # ══════════════════════════════════════════════════════════════
    if args.use_adaptive_masking:
        logger.info(
            f"Using EclipAdaptiveMLMCollator  "
            f"(target={args.target_mlm_prob}, "
            f"eclip=[{args.eclip_mlm_lo}, {args.eclip_mlm_hi}], "
            f"flank_min={args.min_flanking_prob})"
        )
        collator = EclipAdaptiveMLMCollator(
            tokenizer=tokenizer,
            mlm=True,
            target_mlm_prob=args.target_mlm_prob,
            eclip_mlm_range=(args.eclip_mlm_lo, args.eclip_mlm_hi),
            min_flanking_prob=args.min_flanking_prob,
        )
    else:
        logger.info(
            f"Using standard MLM collator (uniform {args.target_mlm_prob*100:.0f}%)"
        )
        collator = StandardMLMCollatorClean(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=args.target_mlm_prob,
        )

    # ══════════════════════════════════════════════════════════════
    #  Training arguments
    # ══════════════════════════════════════════════════════════════
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        # Schedule
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,

        # Batch / accumulation
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_drop_last=True,

        # Optimiser
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,

        # Precision
        fp16=args.fp16,
        bf16=args.bf16,

        # Evaluation & checkpointing strategy
        # epoch-based is recommended for continued pretraining so that
        # early_stopping_patience is measured in epochs, not arbitrary steps.
        eval_strategy=args.eval_strategy if val_ds else "no",
        eval_steps=args.eval_steps if args.eval_strategy == "steps" and val_ds else None,
        save_strategy=args.eval_strategy if val_ds else "steps",
        save_steps=args.save_steps if args.eval_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="eval_loss" if val_ds else None,
        greater_is_better=False if val_ds else None,
        save_safetensors=False,  # BERT shared weights don't work with safetensors

        # Logging
        logging_dir=log_dir,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        disable_tqdm=False,

        # DataLoader
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,

        # Misc
        seed=args.seed,
    )

    # ══════════════════════════════════════════════════════════════
    #  Callbacks
    # ══════════════════════════════════════════════════════════════
    callbacks = []
    if val_ds:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
            )
        )
        logger.info(
            f"Early stopping enabled (patience={args.early_stopping_patience})"
        )

    # ══════════════════════════════════════════════════════════════
    #  Trainer
    # ══════════════════════════════════════════════════════════════
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # ══════════════════════════════════════════════════════════════
    #  Train
    # ══════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("  Starting training")
    logger.info("=" * 60)
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ══════════════════════════════════════════════════════════════
    #  Save & evaluate
    # ══════════════════════════════════════════════════════════════
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if val_ds:
        logger.info("Running final evaluation …")
        eval_metrics = trainer.evaluate()
        perplexity = math.exp(min(eval_metrics["eval_loss"], 20))
        eval_metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        logger.info(f"  Final eval loss: {eval_metrics['eval_loss']:.4f}")
        logger.info(f"  Final perplexity: {perplexity:.2f}")

    logger.info(f"Model saved to {args.output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
