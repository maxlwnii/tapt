"""
DNABERT2 continued pre-training (MLM) — v3 (simple).

Masking behaviour
-----------------
  * 15 % of non-special tokens are chosen uniformly at random.
  * ALL selected tokens are replaced with [MASK]  (no 80/10/10 split).
  * Loss is computed only on masked positions.

Input JSON format (array of objects; only "sequence" is required):
  [{"sequence": "ACGT...", "seq_id": "...", ...}, ...]

Usage (single GPU):
    python pretrain_dnabert2_v3.py \
        --train_file  /path/to/train.json \
        --val_file    /path/to/val.json \
        --output_dir  ./models/dnabert2_tapt_v3 \
        --fp16

Usage (multi-GPU / torchrun):
    torchrun --nproc_per_node=2 pretrain_dnabert2_v3.py ...

Author: Maximilian Lewin
"""

import argparse
import inspect
import logging
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.models.bert.configuration_bert import BertConfig

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


_DNABERT2_CUSTOM_CODE_FILES = (
    "configuration_bert.py",
    "bert_layers.py",
    "bert_padding.py",
    "flash_attn_triton.py",
)


def _get_model_code_dir(model) -> Optional[Path]:
    try:
        mod_file = inspect.getfile(model.__class__)
    except Exception:
        return None
    return Path(mod_file).resolve().parent


def _copy_custom_code_files(src_dir: Optional[Path], dst_dir: Path) -> None:
    if src_dir is None:
        return
    if not src_dir.exists():
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []
    for fname in _DNABERT2_CUSTOM_CODE_FILES:
        src = src_dir / fname
        dst = dst_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(fname)
        else:
            missing.append(fname)

    if copied:
        log.info(f"  copied custom code files to {dst_dir}: {copied}")
    if missing:
        log.warning(f"  missing custom code files in source {src_dir}: {missing}")


class SaveCustomCodeCallback(TrainerCallback):
    def __init__(self, src_dir: Optional[Path], output_dir: Path):
        self.src_dir = src_dir
        self.output_dir = output_dir

    def on_save(self, args, state, control, **kwargs):
        if state.global_step is None:
            return control
        ckpt_dir = self.output_dir / f"checkpoint-{state.global_step}"
        _copy_custom_code_files(self.src_dir, ckpt_dir)
        return control


# ──────────────────────────────────────────────────────────────────────────────
#  Data collator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SimpleMaskCollator(DataCollatorForLanguageModeling):
    """
    MLM collator that replaces exactly ``mlm_probability`` of non-special
    tokens with [MASK].  No random-token replacement, no keep-original.

    All masked positions have their original token id in ``labels``; all
    other positions have -100 (ignored by the loss).
    """

    mlm_probability: float = 0.15

    def __post_init__(self):
        super().__post_init__()  # validates mlm_probability range, sets self.generator
        if not self.mlm:
            raise ValueError("SimpleMaskCollator requires mlm=True.")
        if self.tokenizer.mask_token is None:
            raise ValueError("Tokenizer has no mask_token.")

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        batch["input_ids"], batch["labels"] = self._mask(batch["input_ids"])
        return batch

    def _mask(self, input_ids: torch.Tensor):
        labels = input_ids.clone()

        # Build special-token / padding mask  (True = do NOT mask)
        special = torch.tensor(
            [
                self.tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True)
                for row in input_ids.tolist()
            ],
            dtype=torch.bool,
        )
        if self.tokenizer.pad_token_id is not None:
            special |= input_ids.eq(self.tokenizer.pad_token_id)

        # Sample bernoulli mask over eligible positions only
        prob = torch.full(input_ids.shape, self.mlm_probability)
        prob.masked_fill_(special, 0.0)
        masked = torch.bernoulli(prob).bool()

        labels[~masked] = -100                          # only compute loss on masked
        input_ids[masked] = self.tokenizer.mask_token_id
        return input_ids, labels


# ──────────────────────────────────────────────────────────────────────────────
#  Tokenisation
# ──────────────────────────────────────────────────────────────────────────────

def make_tokenize_fn(tokenizer, max_length: int):
    """Return a batched tokenisation function for datasets.map()."""
    def _fn(examples):
        return tokenizer(
            examples["sequence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    return _fn


# ──────────────────────────────────────────────────────────────────────────────
#  Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="DNABERT2 continued pre-training — simple 15% MLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument("--model",        default="zhihan1996/DNABERT-2-117M",
                   help="HuggingFace model ID or local checkpoint path")
    p.add_argument("--max_length",   type=int, default=512)

    # Data
    p.add_argument("--train_file",   required=True)
    p.add_argument("--val_file",     default=None)
    p.add_argument("--max_train",    type=int, default=None,
                   help="Limit training samples (for quick tests)")
    p.add_argument("--max_val",      type=int, default=None)
    p.add_argument("--num_workers",  type=int, default=4,
                   help="Dataset map workers")

    # Masking
    p.add_argument("--mlm_prob",     type=float, default=0.15)

    # Training
    p.add_argument("--output_dir",   default="./models/dnabert2_tapt_v3")
    p.add_argument("--epochs",       type=int, default=10)
    p.add_argument("--batch_size",   type=int, default=16,
                   help="Per-device train batch size")
    p.add_argument("--eval_batch",   type=int, default=32)
    p.add_argument("--accum",        type=int, default=16,
                   help="Gradient accumulation steps")
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm",type=float, default=1.0)

    # Eval / checkpointing
    p.add_argument("--patience",     type=int, default=5,
                   help="Early stopping patience (eval rounds)")
    p.add_argument("--save_total",   type=int, default=3)
    p.add_argument("--logging_steps",type=int, default=50)

    # Hardware
    p.add_argument("--fp16",         action="store_true")
    p.add_argument("--bf16",         action="store_true")
    p.add_argument("--grad_ckpt",    action="store_true",
                   help="Enable gradient checkpointing")
    p.add_argument("--pin_memory",   action="store_true")

    # Misc
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--report_to",    default="tensorboard",
                   choices=["tensorboard", "wandb", "none"])
    p.add_argument("--resume",       default=None,
                   help="Checkpoint path to resume from")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── tokenizer ─────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, model_max_length=args.max_length
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info(
        f"  vocab={len(tokenizer)}  "
        f"pad={tokenizer.pad_token_id}  mask={tokenizer.mask_token_id}"
    )

    # ── model ─────────────────────────────────────────────────────────────
    log.info(f"Loading model: {args.model}")
    config = BertConfig.from_pretrained(args.model, trust_remote_code=True)
    model  = AutoModelForMaskedLM.from_pretrained(
        args.model, config=config, trust_remote_code=True
    )
    log.info(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    model_code_dir = _get_model_code_dir(model)

    if args.grad_ckpt:
        try:
            model.gradient_checkpointing_enable()
            log.info("  gradient checkpointing: on")
        except ValueError as exc:
            log.warning(
                f"  gradient checkpointing not supported by this model ({exc}); "
                "continuing without it."
            )

    # ── datasets ──────────────────────────────────────────────────────────
    tokenize_fn = make_tokenize_fn(tokenizer, args.max_length)

    log.info(f"Loading train data: {args.train_file}")
    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    if args.max_train:
        train_ds = train_ds.select(range(min(args.max_train, len(train_ds))))

    remove_cols = list(train_ds.column_names)
    train_ds = train_ds.map(
        tokenize_fn, batched=True,
        num_proc=args.num_workers,
        remove_columns=remove_cols,
        desc="Tokenising train",
    )
    log.info(f"  train: {len(train_ds):,} examples")

    val_ds = None
    if args.val_file:
        log.info(f"Loading val data: {args.val_file}")
        val_ds = load_dataset("json", data_files=args.val_file, split="train")
        if args.max_val:
            val_ds = val_ds.select(range(min(args.max_val, len(val_ds))))

        val_remove = list(val_ds.column_names)
        val_ds = val_ds.map(
            tokenize_fn, batched=True,
            num_proc=args.num_workers,
            remove_columns=val_remove,
            desc="Tokenising val",
        )
        log.info(f"  val:   {len(val_ds):,} examples")

    # quick sanity check
    sample = train_ds[0]
    log.info(f"  columns: {list(sample.keys())}  ids_len={len(sample['input_ids'])}")

    # ── collator ──────────────────────────────────────────────────────────
    collator = SimpleMaskCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob,
    )
    log.info(f"Collator: SimpleMaskCollator  mlm_prob={args.mlm_prob}")

    # ── training arguments ────────────────────────────────────────────────
    have_val = val_ds is not None
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.accum,
        dataloader_drop_last=True,

        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,

        fp16=args.fp16,
        bf16=args.bf16,

        eval_strategy="epoch" if have_val else "no",
        save_strategy="epoch" if have_val else "steps",
        save_steps=1000,   # only effective when save_strategy='steps' (no val file)
        save_total_limit=args.save_total,
        load_best_model_at_end=have_val,
        metric_for_best_model="eval_loss" if have_val else None,
        greater_is_better=False if have_val else None,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        report_to=args.report_to,

        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=args.pin_memory,
        seed=args.seed,
    )

    # ── callbacks ─────────────────────────────────────────────────────────
    callbacks = []
    if have_val:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
        log.info(f"Early stopping: patience={args.patience}")
    callbacks.append(SaveCustomCodeCallback(model_code_dir, Path(args.output_dir)))

    # ── trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # ── train ─────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Starting training")
    log.info("=" * 60)
    result = trainer.train(resume_from_checkpoint=args.resume)

    # ── save ──────────────────────────────────────────────────────────────
    trainer.save_model(args.output_dir)
    state = load_file(f"{args.output_dir}/model.safetensors")
    assert not any(k.startswith("bert.") for k in state), \
        "Keys still have 'bert.' prefix — safetensors save failed"
    print("Key sample:", list(state.keys())[:3])
    tokenizer.save_pretrained(args.output_dir)
    _copy_custom_code_files(model_code_dir, Path(args.output_dir))
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()

    if have_val:
        eval_metrics = trainer.evaluate()
        ppl = math.exp(min(eval_metrics["eval_loss"], 20))
        eval_metrics["perplexity"] = ppl
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        log.info(f"Final eval loss: {eval_metrics['eval_loss']:.4f}  ppl: {ppl:.2f}")

    log.info(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
