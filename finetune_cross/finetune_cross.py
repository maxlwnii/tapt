"""
Fixed-hyperparameter finetuning for cross-cell and cross-length experiments.

Supports:
  LAMAR variants:   random | pretrained | tapt_512 | tapt_1024
  DNABERT2 variants: pretrained | random

Hyperparameters are fixed (no HPO). Each run trains on one RBP/pair,
evaluates on validation and test splits, and saves results.json.

Usage examples:
  # LAMAR pretrained, cross-cell
  python finetune_cross.py \\
      --model_type lamar \\
      --experiment cross_cell \\
      --pair_name HNRNPK_train_K562_test_HepG2_fixlen_101 \\
      --pretrain_path $THESIS_ROOT/LAMAR/weights \\
      --max_length 1024 \\
      --output_dir ./results/cross_cell/lamar_pretrained

  # DNABERT2 random, cross-length
  python finetune_cross.py \\
      --model_type dnabert2 \\
      --experiment cross_length \\
      --pair_name HNRNPK_K562_ENCSR268ETU \\
      --use_random_init --max_length 128 \\
      --output_dir ./results/cross_length/dnabert2_random

Author: Maximilian Lewin
"""

import argparse
import csv
import inspect
import json
import logging
import os
import shutil
import sys

import numpy as np
import sklearn.metrics
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, Subset

import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Conditional imports — only needed for LAMAR
try:
    from transformers import EsmConfig, EsmForSequenceClassification
    from safetensors.torch import load_file
    from datasets import load_dataset as load_ds
except ImportError:
    pass  # Only needed when --model_type lamar

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Version-aware eval_strategy detection
#    transformers ≥4.46 renamed evaluation_strategy → eval_strategy
# ═══════════════════════════════════════════════════════════════
_sig = inspect.signature(TrainingArguments.__init__)
_EVAL_KEY = (
    "eval_strategy"
    if "eval_strategy" in _sig.parameters
    else "evaluation_strategy"
)


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════
THESIS_ROOT = os.environ.get(
    "THESIS_ROOT",
    "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis",
)

LAMAR_TOKENIZER = os.path.join(
    THESIS_ROOT, "LAMAR", "src", "pretrain", "saving_model",
    "tapt_lamar", "checkpoint-100000",
)
DNABERT2_MODEL = "zhihan1996/DNABERT-2-117M"


# ═══════════════════════════════════════════════════════════════
#  Fixed Hyperparameters
# ═══════════════════════════════════════════════════════════════
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_EPOCHS = 10
GRAD_ACCUM = 1
PATIENCE = 3
N_FOLDS = 5


# ═══════════════════════════════════════════════════════════════
#  Common Helpers
# ═══════════════════════════════════════════════════════════════

def init_weights(module):
    """Kaiming-style random initialization (WOLF baseline)."""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)


# ═══════════════════════════════════════════════════════════════
#  CV helpers
# ═══════════════════════════════════════════════════════════════

def _aggregate_cv_metrics(fold_metrics: list) -> dict:
    """Mean ± std across folds for every key that starts with 'eval_'."""
    agg = {}
    all_keys = {k for m in fold_metrics for k in m}
    for k in sorted(all_keys):
        vals = [m[k] for m in fold_metrics if k in m]
        if vals and isinstance(vals[0], (int, float)):
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"]  = float(np.std(vals))
    return agg


# ═══════════════════════════════════════════════════════════════
#  LAMAR Helpers
# ═══════════════════════════════════════════════════════════════

def load_encoder_weights(model, weights_path):
    """Load pretrained LAMAR encoder weights (safetensors file)."""
    state_dict = load_file(weights_path)
    encoder_weights = {}
    for k, v in state_dict.items():
        if "lm_head" in k or "classifier" in k:
            continue
        if k.startswith("esm."):
            encoder_weights[k] = v
        else:
            encoder_weights[f"esm.{k}"] = v
    missing, unexpected = model.load_state_dict(encoder_weights, strict=False)
    logger.info(
        f"Loaded {len(encoder_weights)} encoder tensors, "
        f"{len(missing)} missing, {len(unexpected)} unexpected"
    )
    return model


def compute_metrics_lamar(p):
    """Compute metrics for LAMAR (binary classification)."""
    predictions, labels = p
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    probs = torch.nn.functional.softmax(
        torch.tensor(predictions), dim=-1
    ).numpy()
    pred_labels = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        labels, pred_labels, average="binary", zero_division=0
    )
    acc = sklearn.metrics.accuracy_score(labels, pred_labels)
    mcc = sklearn.metrics.matthews_corrcoef(labels, pred_labels)
    try:
        auc = sklearn.metrics.roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc = 0.5
    try:
        auprc = sklearn.metrics.average_precision_score(labels, probs[:, 1])
    except Exception:
        auprc = 0.5

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
        "auprc": float(auprc),
        "matthews_correlation": float(mcc),
    }


# ═══════════════════════════════════════════════════════════════
#  DNABERT2 Helpers
# ═══════════════════════════════════════════════════════════════

class SupervisedDataset(Dataset):
    """Load CSV (sequence, label) → tokenized dataset for DNABERT2."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        super().__init__()
        with open(data_path) as f:
            data = list(csv.reader(f))[1:]  # skip header
        self.texts = [d[0] for d in data]
        self.labels = [int(d[1]) for d in data]
        self.num_labels = len(set(self.labels))

        output = tokenizer(
            self.texts,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class DataCollatorForSupervisedDataset:
    """Pad input_ids in a batch, create attention_mask."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.tensor(labels, dtype=torch.long)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def compute_metrics_dnabert2(eval_pred):
    """Compute metrics for DNABERT2 (binary classification)."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)

    predictions = np.argmax(logits.numpy(), axis=-1)
    probabilities = torch.softmax(logits, dim=-1).numpy()
    labels_np = labels if isinstance(labels, np.ndarray) else labels.numpy()

    valid = labels_np != -100
    preds = predictions[valid]
    labs = labels_np[valid]
    probs = probabilities[valid]

    metrics = {
        "accuracy": float(sklearn.metrics.accuracy_score(labs, preds)),
        "f1": float(sklearn.metrics.f1_score(
            labs, preds, average="macro", zero_division=0
        )),
        "matthews_correlation": float(
            sklearn.metrics.matthews_corrcoef(labs, preds)
        ),
        "precision": float(sklearn.metrics.precision_score(
            labs, preds, average="macro", zero_division=0
        )),
        "recall": float(sklearn.metrics.recall_score(
            labs, preds, average="macro", zero_division=0
        )),
    }
    if probs.shape[1] == 2:
        try:
            metrics["auc"] = float(
                sklearn.metrics.roc_auc_score(labs, probs[:, 1])
            )
        except ValueError:
            metrics["auc"] = 0.5
        try:
            metrics["auprc"] = float(
                sklearn.metrics.average_precision_score(labs, probs[:, 1])
            )
        except ValueError:
            metrics["auprc"] = 0.5
    return metrics


# ═══════════════════════════════════════════════════════════════
#  Run LAMAR
# ═══════════════════════════════════════════════════════════════

def _build_lamar_model(tokenizer, pretrain_path: str):
    """Fresh LAMAR model, optionally loading pretrained encoder weights."""
    config = EsmConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        token_dropout=False,
        positional_embedding_type="rotary",
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        num_labels=2,
    )
    model = EsmForSequenceClassification(config)
    model.apply(init_weights)
    if pretrain_path and os.path.exists(pretrain_path):
        load_encoder_weights(model, pretrain_path)
    return model


def run_lamar(args):
    tokenizer_path = args.tokenizer_path or LAMAR_TOKENIZER
    logger.info(f"Loading LAMAR tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    logger.info(f"  vocab_size={len(tokenizer)}")

    # ── Load data ─────────────────────────────────────────────
    pair_dir = os.path.join(THESIS_ROOT, "data", args.experiment, args.pair_name)
    if not os.path.isdir(pair_dir):
        logger.error(f"Data directory not found: {pair_dir}")
        sys.exit(1)

    # Load all three splits as separate datasets
    def _load_split(fname):
        return load_ds("csv", data_files={"data": os.path.join(pair_dir, fname)})["data"]

    raw_train = _load_split("train.csv")
    raw_val   = _load_split("dev.csv")
    raw_test  = _load_split("test.csv")

    # Detect sequence column
    seq_col = "sequence" if "sequence" in raw_train.column_names else "seq"

    def preprocess(examples):
        seqs = [s.replace("U", "T").replace("u", "t") for s in examples[seq_col]]
        return tokenizer(seqs, truncation=True, padding="max_length", max_length=args.max_length)

    enc_train = raw_train.map(preprocess, batched=True, remove_columns=[seq_col])
    enc_val   = raw_val.map(preprocess, batched=True, remove_columns=[seq_col])
    enc_test  = raw_test.map(preprocess, batched=True, remove_columns=[seq_col])

    # Pool train + val for cross-validation
    from datasets import concatenate_datasets
    enc_trainval = concatenate_datasets([enc_train, enc_val])

    if args.max_train_samples and args.max_train_samples < len(enc_trainval):
        enc_trainval = enc_trainval.shuffle(seed=args.seed).select(range(args.max_train_samples))

    labels_trainval = np.array(enc_trainval["label"])
    logger.info(
        f"  trainval={len(enc_trainval)}, test={len(enc_test)}  "
        f"(CV with {N_FOLDS} folds)"
    )

    # ── Output directory ──────────────────────────────────────
    output_dir = os.path.join(args.output_dir, args.pair_name)
    os.makedirs(output_dir, exist_ok=True)

    def _ta(fold_dir):
        return TrainingArguments(**{
            "output_dir": fold_dir,
            "num_train_epochs": NUM_EPOCHS,
            "per_device_train_batch_size": BATCH_SIZE,
            "per_device_eval_batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "adam_beta1": 0.9,
            "adam_beta2": 0.98,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            _EVAL_KEY: "epoch",
            "save_strategy": "epoch",
            "logging_steps": 50,
            "save_total_limit": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "auc",
            "greater_is_better": True,
            "report_to": "none",
            "seed": args.seed,
            "dataloader_drop_last": True,
        })

    # ── 5-fold cross-validation on trainval ───────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.seed)
    fold_metrics_list = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(enc_trainval)), labels_trainval)):
        logger.info(f"  CV fold {fold_idx + 1}/{N_FOLDS}  "
                    f"(train={len(tr_idx)}, val={len(va_idx)})")
        fold_train = enc_trainval.select(tr_idx.tolist())
        fold_val   = enc_trainval.select(va_idx.tolist())

        model = _build_lamar_model(tokenizer, args.pretrain_path)
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        trainer = Trainer(
            model=model,
            args=_ta(fold_dir),
            train_dataset=fold_train,
            eval_dataset=fold_val,
            compute_metrics=compute_metrics_lamar,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
        )
        trainer.train()
        fold_metrics_list.append(trainer.evaluate(eval_dataset=fold_val))
        shutil.rmtree(fold_dir, ignore_errors=True)  # clean checkpoints

    cv_metrics = _aggregate_cv_metrics(fold_metrics_list)
    logger.info(
        f"CV {N_FOLDS}-fold auc: "
        f"{cv_metrics.get('eval_auc_mean', 0):.4f} "
        f"± {cv_metrics.get('eval_auc_std', 0):.4f}"
    )

    # ── Final model: train on full trainval, evaluate on test ─
    logger.info("  Training final model on full trainval …")
    final_model = _build_lamar_model(tokenizer, args.pretrain_path)
    final_dir = os.path.join(output_dir, "final")
    trainer_final = Trainer(
        model=final_model,
        args=_ta(final_dir),
        train_dataset=enc_trainval,
        eval_dataset=enc_test,
        compute_metrics=compute_metrics_lamar,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )
    trainer_final.train()
    test_metrics = trainer_final.evaluate(eval_dataset=enc_test)
    shutil.rmtree(final_dir, ignore_errors=True)

    logger.info(
        f"test_auc={test_metrics.get('eval_auc', 0):.4f}"
    )

    # ── Determine variant name ────────────────────────────────
    variant = "random"
    if args.pretrain_path:
        if "checkpoint-98000" in args.pretrain_path:
            variant = "tapt_512"
        elif "checkpoint-134000" in args.pretrain_path:
            variant = "tapt_1024"
        elif "weights" in args.pretrain_path:
            variant = "pretrained"
        else:
            variant = "pretrained_custom"

    # ── Save results ──────────────────────────────────────────
    results = {
        "experiment": args.experiment,
        "model_type": "lamar",
        "variant": variant,
        "pair_name": args.pair_name,
        "pretrain_path": args.pretrain_path or "",
        "max_length": args.max_length,
        "n_folds": N_FOLDS,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "per_device_train_batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "num_train_epochs": NUM_EPOCHS,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "early_stopping_patience": PATIENCE,
        },
        "cv_metrics": cv_metrics,
        "cv_fold_metrics": fold_metrics_list,
        "test_metrics": {k: v for k, v in test_metrics.items()},
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return results


# ═══════════════════════════════════════════════════════════════
#  Run DNABERT2
# ═══════════════════════════════════════════════════════════════

def run_dnabert2(args):
    from transformers.models.bert.configuration_bert import BertConfig

    dnabert2_model_path = args.dnabert2_model_path or DNABERT2_MODEL

    tokenizer = AutoTokenizer.from_pretrained(
        dnabert2_model_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # ── Load data ─────────────────────────────────────────────
    pair_dir = os.path.join(THESIS_ROOT, "data", args.experiment, args.pair_name)
    if not os.path.isdir(pair_dir):
        logger.error(f"Data directory not found: {pair_dir}")
        sys.exit(1)

    train_ds = SupervisedDataset(
        os.path.join(pair_dir, "train.csv"), tokenizer, args.max_length
    )
    val_ds = SupervisedDataset(
        os.path.join(pair_dir, "dev.csv"), tokenizer, args.max_length
    )
    test_ds = SupervisedDataset(
        os.path.join(pair_dir, "test.csv"), tokenizer, args.max_length
    )

    # Pool train + val for cross-validation
    trainval_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])
    # Collect labels for stratified splitting
    tv_labels = (
        [train_ds.labels[i] for i in range(len(train_ds))]
        + [val_ds.labels[i] for i in range(len(val_ds))]
    )
    tv_labels = np.array(tv_labels)

    if args.max_train_samples and args.max_train_samples < len(trainval_ds):
        keep = min(args.max_train_samples, len(trainval_ds))
        trainval_ds = Subset(trainval_ds, list(range(keep)))
        tv_labels = tv_labels[:keep]

    logger.info(
        f"  trainval={len(trainval_ds)}, test={len(test_ds)}  "
        f"(CV with {N_FOLDS} folds)"
    )

    config = BertConfig.from_pretrained(dnabert2_model_path, num_labels=2)
    collator = DataCollatorForSupervisedDataset(tokenizer)

    def _build_db2_model():
        m = AutoModelForSequenceClassification.from_pretrained(
            dnabert2_model_path, config=config, trust_remote_code=True, low_cpu_mem_usage=False
        )
        if args.use_random_init:
            m.apply(init_weights)
        return m

    # ── Output directory ──────────────────────────────────────
    output_dir = os.path.join(args.output_dir, args.pair_name)
    os.makedirs(output_dir, exist_ok=True)

    def _ta(fold_dir):
        return TrainingArguments(**{
            "output_dir": fold_dir,
            "num_train_epochs": NUM_EPOCHS,
            "per_device_train_batch_size": BATCH_SIZE,
            "per_device_eval_batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "optim": "adamw_torch",
            "max_grad_norm": 1.0,
            _EVAL_KEY: "epoch",
            "save_strategy": "epoch",
            "logging_steps": 50,
            "save_total_limit": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "auc",
            "greater_is_better": True,
            "report_to": "none",
            "seed": args.seed,
            "dataloader_pin_memory": True,
        })

    # ── 5-fold cross-validation on trainval ───────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.seed)
    fold_metrics_list = []
    all_indices = np.arange(len(trainval_ds))

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(all_indices, tv_labels)):
        logger.info(f"  CV fold {fold_idx + 1}/{N_FOLDS}  "
                    f"(train={len(tr_idx)}, val={len(va_idx)})")
        fold_train = Subset(trainval_ds, tr_idx.tolist())
        fold_val   = Subset(trainval_ds, va_idx.tolist())

        model = _build_db2_model()
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        trainer = Trainer(
            model=model,
            args=_ta(fold_dir),
            train_dataset=fold_train,
            eval_dataset=fold_val,
            data_collator=collator,
            compute_metrics=compute_metrics_dnabert2,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
            tokenizer=tokenizer,
        )
        trainer.train()
        fold_metrics_list.append(trainer.evaluate(eval_dataset=fold_val))
        shutil.rmtree(fold_dir, ignore_errors=True)  # clean checkpoints

    cv_metrics = _aggregate_cv_metrics(fold_metrics_list)
    logger.info(
        f"CV {N_FOLDS}-fold auc: "
        f"{cv_metrics.get('eval_auc_mean', 0):.4f} "
        f"± {cv_metrics.get('eval_auc_std', 0):.4f}"
    )

    # ── Final model: train on full trainval, evaluate on test ─
    logger.info("  Training final model on full trainval …")
    final_model = _build_db2_model()
    final_dir = os.path.join(output_dir, "final")
    trainer_final = Trainer(
        model=final_model,
        args=_ta(final_dir),
        train_dataset=trainval_ds,
        eval_dataset=test_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_dnabert2,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
        tokenizer=tokenizer,
    )
    trainer_final.train()
    test_metrics = trainer_final.evaluate(eval_dataset=test_ds)
    shutil.rmtree(final_dir, ignore_errors=True)

    logger.info(f"test_auc={test_metrics.get('eval_auc', 0):.4f}")

    # ── Save results ──────────────────────────────────────────
    if args.use_random_init:
        variant = "random"
    elif "dnabert2_tapt_v3" in dnabert2_model_path:
        variant = "tapt_v3"
    elif dnabert2_model_path != DNABERT2_MODEL:
        variant = "pretrained_custom"
    else:
        variant = "pretrained"

    results = {
        "experiment": args.experiment,
        "model_type": "dnabert2",
        "variant": variant,
        "pair_name": args.pair_name,
        "max_length": args.max_length,
        "n_folds": N_FOLDS,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "per_device_train_batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "num_train_epochs": NUM_EPOCHS,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "early_stopping_patience": PATIENCE,
        },
        "cv_metrics": cv_metrics,
        "cv_fold_metrics": fold_metrics_list,
        "test_metrics": {k: v for k, v in test_metrics.items()},
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return results


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Fixed-hyperparameter finetuning for cross-cell / cross-length",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model_type", required=True, choices=["lamar", "dnabert2"],
        help="Model family",
    )
    p.add_argument(
        "--experiment", required=True, choices=["cross_cell", "cross_length"],
        help="Experiment type",
    )
    p.add_argument(
        "--pair_name", required=True,
        help="RBP/pair directory name (e.g. HNRNPK_train_K562_test_HepG2_fixlen_101)",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Base output dir; results go to <output_dir>/<pair_name>/",
    )
    # LAMAR-specific
    p.add_argument(
        "--pretrain_path", default="",
        help="Path to LAMAR pretrained weights (safetensors). Omit for random init.",
    )
    p.add_argument(
        "--tokenizer_path", default="",
        help="Path to LAMAR tokenizer (default: checkpoint-100000)",
    )
    # DNABERT2-specific
    p.add_argument(
        "--use_random_init", action="store_true",
        help="For DNABERT2: apply random init instead of pretrained weights",
    )
    p.add_argument(
        "--dnabert2_model_path", default=DNABERT2_MODEL,
        help="DNABERT2 checkpoint path or HF ID (for model_type=dnabert2)",
    )
    # Common
    p.add_argument("--max_length", type=int, default=1024, help="Max tokenizer length")
    p.add_argument(
        "--max_train_samples", type=int, default=None,
        help="Limit training samples (for testing)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    transformers.set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 60)
    logger.info(f"Model: {args.model_type} | Experiment: {args.experiment}")
    logger.info(f"Pair:  {args.pair_name}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Hyperparameters: lr={LEARNING_RATE}, bs={BATCH_SIZE}, "
                f"wd={WEIGHT_DECAY}, warmup={WARMUP_RATIO}, "
                f"epochs={NUM_EPOCHS}, patience={PATIENCE}")
    logger.info("=" * 60)

    if args.model_type == "lamar":
        run_lamar(args)
    else:
        run_dnabert2(args)


if __name__ == "__main__":
    main()
