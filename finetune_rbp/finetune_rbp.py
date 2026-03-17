"""
finetune_rbp.py
===============
Fixed-hyperparameter fine-tuning for ALL model variants on per-RBP datasets.

Data sources (auto-detected from rbp_name, or pass --data_root explicitly):
  data/finetune_data_koo/  – 10 RBPs (names ending in _200)
  DNABERT2/data/           –  8 RBPs (IDR-based, names ending in _IDR)

Supported variants:
  LAMAR:    lamar_pretrained, lamar_tapt_1024, lamar_tapt_standard_1gpu,
            lamar_tapt_custom_1gpu, lamar_tapt_512, lamar_tapt_512_std,
            lamar_random
  DNABERT2: dnabert2_pretrained, dnabert2_tapt, dnabert2_tapt_v3,
            dnabert2_random

Training protocol:
  - 5-fold CV on train+dev, final model trained on full train+dev, tested on test
  - Same hyperparameters for LAMAR and DNABERT2 (no HPO)
  - Results saved to <output_dir>/<variant>/<rbp_name>/results.json

Usage:
  python finetune_rbp.py --variant lamar_pretrained --rbp_name HNRNPK_K562_200
  python finetune_rbp.py --variant dnabert2_tapt_v3 --rbp_name GTF2F1_K562_IDR
  python finetune_rbp.py --variant lamar_tapt_512  --rbp_name QKI_K562_200 \\
         --output_dir ./results/rbp
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

# LAMAR-specific (optional at import time; errors surface at variant load time)
try:
    from transformers import EsmConfig, EsmForSequenceClassification
    from safetensors.torch import load_file as _load_safetensors
    from datasets import load_dataset as load_ds
    from datasets import concatenate_datasets
except ImportError:
    pass

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Version-aware eval_strategy key
# ═══════════════════════════════════════════════════════════════
_sig = inspect.signature(TrainingArguments.__init__)
_EVAL_KEY = (
    "eval_strategy"
    if "eval_strategy" in _sig.parameters
    else "evaluation_strategy"
)


# ═══════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════
THESIS_ROOT = os.environ.get(
    "THESIS_ROOT",
    "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis",
)
_BASE     = THESIS_ROOT
_DB2_BASE = os.path.join(_BASE, "DNABERT2")

DATA_ROOT_KOO = os.path.join(_BASE, "data", "finetune_data_koo")
DATA_ROOT_CSV = os.path.join(_DB2_BASE, "data")


# ═══════════════════════════════════════════════════════════════
#  Model specs  (mirrors linear_probe_cross_cell.py MODEL_DEFAULTS)
# ═══════════════════════════════════════════════════════════════
MODEL_SPECS: dict[str, dict] = {
    # ── LAMAR ──────────────────────────────────────────────────
    "lamar_pretrained": {
        "type":         "lamar",
        "weights_path": f"{_BASE}/LAMAR/weights",
        "tokenizer":    (f"{_BASE}/LAMAR/src/pretrain/saving_model"
                         "/tapt_1024_standard_collator/checkpoint-134000"),
        "max_length":   1024,
    },
    "lamar_tapt_1024": {
        "type":         "lamar",
        "weights_path": (f"{_BASE}/LAMAR/src/pretrain/saving_model"
                         "/tapt_1024_standard_collator/checkpoint-134000"),
        "tokenizer":    (f"{_BASE}/LAMAR/src/pretrain/saving_model"
                         "/tapt_1024_standard_collator/checkpoint-134000"),
        "max_length":   1024,
    },
    "lamar_tapt_standard_1gpu": {
        "type":         "lamar",
        "weights_path": (f"{_BASE}/LAMAR/src/pretrain/saving_model"
                         "/tapt_1024_standard_collator_1gpu/checkpoint-232000"),
        "tokenizer":    (f"{_BASE}/LAMAR/src/pretrain/saving_model"
                         "/tapt_1024_standard_collator/checkpoint-134000"),
        "max_length":   1024,
    },
    "lamar_tapt_custom_1gpu": {
        "type":         "lamar",
        "weights_path": (f"{_BASE}/LAMAR/src/pretrain/saving_model"
                         "/tapt_1024_custom_collator_1gpu/checkpoint-232000"),
        "tokenizer":    (f"{_BASE}/LAMAR/src/pretrain/saving_model"
                         "/tapt_1024_standard_collator/checkpoint-134000"),
        "max_length":   1024,
    },
    "lamar_tapt_512": {
        "type":         "lamar",
        "weights_path": (f"{_BASE}/LAMAR/src/pretrain/saving_model"
                         "/tapt_lamar/checkpoint-98000"),
        "tokenizer":    (f"{_BASE}/LAMAR/src/pretrain/saving_model"
                         "/tapt_1024_standard_collator/checkpoint-134000"),
        "max_length":   512,
    },
    "lamar_tapt_512_std": {
        "type":         "lamar",
        "weights_path": (f"{_BASE}/LAMAR/src/pretrain/saving_model"
                         "/tapt_512_standard_collator_1gpu/checkpoint-265000"),
        "tokenizer":    (f"{_BASE}/LAMAR/src/pretrain/saving_model"
                         "/tapt_1024_standard_collator/checkpoint-134000"),
        "max_length":   512,
    },
    "lamar_random": {
        "type":         "lamar",
        "weights_path": "",          # empty → random init
        "tokenizer":    (f"{_BASE}/LAMAR/src/pretrain/saving_model"
                         "/tapt_1024_standard_collator/checkpoint-134000"),
        "max_length":   1024,
    },
    # ── DNABERT-2 ──────────────────────────────────────────────
    "dnabert2_pretrained": {
        "type":         "dnabert2",
        "weights_path": "zhihan1996/DNABERT-2-117M",
        "tokenizer":    "zhihan1996/DNABERT-2-117M",
        "max_length":   512,
    },
    "dnabert2_tapt": {
        "type":         "dnabert2",
        "weights_path": (f"{_DB2_BASE}/pretrain/models"
                         "/dnabert2_standard_mlm/checkpoint-25652"),
        "tokenizer":    "zhihan1996/DNABERT-2-117M",
        "max_length":   512,
    },
    "dnabert2_tapt_v3": {
        "type":         "dnabert2",
        "weights_path": (f"{_DB2_BASE}/pretrain/models"
                         "/dnabert2_tapt_v3/checkpoint-2566"),
        "tokenizer":    "zhihan1996/DNABERT-2-117M",
        "max_length":   512,
    },
    "dnabert2_random": {
        "type":         "dnabert2",
        "weights_path": "",          # empty → random init
        "tokenizer":    "zhihan1996/DNABERT-2-117M",
        "max_length":   512,
    },
}

ALL_VARIANTS = list(MODEL_SPECS.keys())


# ═══════════════════════════════════════════════════════════════
#  Fixed Hyperparameters  (same as finetune_cross.py)
# ═══════════════════════════════════════════════════════════════
LEARNING_RATE = 5e-5
BATCH_SIZE    = 16
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
NUM_EPOCHS    = 10
GRAD_ACCUM    = 1
PATIENCE      = 3
N_FOLDS       = 5


# ═══════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════

def init_weights(module):
    """Kaiming-style random init (WOLF baseline)."""
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


def _aggregate_cv_metrics(fold_metrics: list) -> dict:
    """Mean ± std across folds for every eval_ metric."""
    agg: dict = {}
    all_keys = {k for m in fold_metrics for k in m}
    for k in sorted(all_keys):
        vals = [m[k] for m in fold_metrics if k in m]
        if vals and isinstance(vals[0], (int, float)):
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"]  = float(np.std(vals))
    return agg


def _find_data_root(rbp_name: str) -> str:
    """Auto-detect data root from rbp_name pattern."""
    if rbp_name.endswith("_200"):
        root = DATA_ROOT_KOO
    else:
        root = DATA_ROOT_CSV
    rbp_dir = os.path.join(root, rbp_name)
    if not os.path.isdir(rbp_dir):
        raise FileNotFoundError(
            f"RBP directory not found: {rbp_dir}\n"
            f"Checked koo root: {DATA_ROOT_KOO}\n"
            f"Checked csv root: {DATA_ROOT_CSV}"
        )
    return root


# ═══════════════════════════════════════════════════════════════
#  Weight loading  (directories + single files)
# ═══════════════════════════════════════════════════════════════

def _load_weights_from_path(path: str) -> dict:
    """Load safetensors or pytorch_model.bin from file or directory."""
    from pathlib import Path
    p = Path(path)
    if p.is_dir():
        st = p / "model.safetensors"
        if st.exists():
            return _load_safetensors(str(st))
        pb = p / "pytorch_model.bin"
        if pb.exists():
            return torch.load(str(pb), map_location="cpu")
        raise FileNotFoundError(f"No weights file found in directory: {path}")
    return _load_safetensors(str(p))


def load_lamar_encoder_weights(model, weights_path: str):
    """Load LAMAR encoder weights into an EsmForSequenceClassification model."""
    state_dict = _load_weights_from_path(weights_path)
    encoder_weights: dict = {}
    for k, v in state_dict.items():
        if "lm_head" in k or "classifier" in k:
            continue
        if k.startswith("esm."):
            encoder_weights[k] = v
        elif k.startswith("lm_head"):
            continue  # skip
        else:
            encoder_weights[f"esm.{k}"] = v
    missing, unexpected = model.load_state_dict(encoder_weights, strict=False)
    non_trivial_missing = [k for k in missing if "lm_head" not in k and "classifier" not in k]
    if non_trivial_missing:
        logger.warning(f"  LAMAR: {len(non_trivial_missing)} non-trivial missing keys: "
                       f"{non_trivial_missing[:5]}")
    logger.info(f"  Loaded {len(encoder_weights)} encoder tensors "
                f"({len(missing)} missing, {len(unexpected)} unexpected)")
    return model


# ═══════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════

def _compute_metrics(eval_pred) -> dict:
    """Unified metric computation for binary classification (both model types)."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    if not isinstance(logits, np.ndarray):
        logits = np.array(logits)
    labels_np = np.array(labels)

    valid = labels_np != -100
    labs   = labels_np[valid]
    preds  = np.argmax(logits[valid], axis=-1)
    probs  = torch.softmax(torch.tensor(logits[valid], dtype=torch.float32), dim=-1).numpy()

    metrics: dict = {
        "accuracy":              float(sklearn.metrics.accuracy_score(labs, preds)),
        "f1":                    float(sklearn.metrics.f1_score(labs, preds, average="binary", zero_division=0)),
        "precision":             float(sklearn.metrics.precision_score(labs, preds, average="binary", zero_division=0)),
        "recall":                float(sklearn.metrics.recall_score(labs, preds, average="binary", zero_division=0)),
        "matthews_correlation":  float(sklearn.metrics.matthews_corrcoef(labs, preds)),
    }
    if probs.shape[1] == 2:
        try:
            metrics["auc"] = float(sklearn.metrics.roc_auc_score(labs, probs[:, 1]))
        except ValueError:
            metrics["auc"] = 0.5
        try:
            metrics["auprc"] = float(sklearn.metrics.average_precision_score(labs, probs[:, 1]))
        except ValueError:
            metrics["auprc"] = 0.5
    return metrics


# ═══════════════════════════════════════════════════════════════
#  DNABERT-2 Dataset
# ═══════════════════════════════════════════════════════════════

class SupervisedDataset(Dataset):
    """CSV (sequence, label) → tokenised dataset."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        super().__init__()
        with open(data_path) as f:
            data = list(csv.reader(f))[1:]  # skip header
        header_row = None
        with open(data_path) as f:
            reader = csv.reader(f)
            header_row = next(reader)
        seq_idx   = 0 if header_row[0].lower() in ("sequence", "seq", "text") else 0
        label_idx = 1 if header_row[1].lower() in ("label", "labels", "target", "y") else 1
        # Handle varying column orders gracefully
        col_lower = [c.lower() for c in header_row]
        for cand in ("sequence", "seq", "text", "input"):
            if cand in col_lower:
                seq_idx = col_lower.index(cand); break
        for cand in ("label", "labels", "target", "y"):
            if cand in col_lower:
                label_idx = col_lower.index(cand); break

        self.texts  = [row[seq_idx].replace("U", "T").replace("u", "t") for row in data]
        self.labels = [int(row[label_idx]) for row in data]

        output = tokenizer(
            self.texts,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        self.input_ids      = output["input_ids"]
        self.attention_mask = output["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class DataCollatorForSupervisedDataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [inst["input_ids"] for inst in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.tensor([inst["labels"] for inst in instances], dtype=torch.long)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# ═══════════════════════════════════════════════════════════════
#  Training-argument factory
# ═══════════════════════════════════════════════════════════════

def _make_training_args(output_dir: str, seed: int, extra: dict | None = None) -> TrainingArguments:
    base: dict = {
        "output_dir":                    output_dir,
        "num_train_epochs":              NUM_EPOCHS,
        "per_device_train_batch_size":   BATCH_SIZE,
        "per_device_eval_batch_size":    BATCH_SIZE,
        "gradient_accumulation_steps":   GRAD_ACCUM,
        "learning_rate":                 LEARNING_RATE,
        "weight_decay":                  WEIGHT_DECAY,
        "warmup_ratio":                  WARMUP_RATIO,
        "adam_beta1":                    0.9,
        "adam_beta2":                    0.98,
        "adam_epsilon":                  1e-8,
        "max_grad_norm":                 1.0,
        _EVAL_KEY:                       "epoch",
        "save_strategy":                 "epoch",
        "logging_steps":                 50,
        "save_total_limit":              1,
        "load_best_model_at_end":        True,
        "metric_for_best_model":         "auc",
        "greater_is_better":             True,
        "report_to":                     "none",
        "seed":                          seed,
        "dataloader_drop_last":          False,
    }
    if extra:
        base.update(extra)
    return TrainingArguments(**base)


# ═══════════════════════════════════════════════════════════════
#  Run LAMAR
# ═══════════════════════════════════════════════════════════════

def run_lamar(spec: dict, variant: str, rbp_dir: str, output_dir: str, args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer"])
    logger.info(f"  LAMAR tokenizer: vocab_size={len(tokenizer)}")

    max_length = spec["max_length"]

    # ── Load data ─────────────────────────────────────────────
    def _load(fname):
        return load_ds("csv", data_files={"data": os.path.join(rbp_dir, fname)})["data"]

    raw_train = _load("train.csv")
    raw_dev   = _load("dev.csv")
    raw_test  = _load("test.csv")

    seq_col = "sequence" if "sequence" in raw_train.column_names else "seq"

    def _preprocess(examples):
        seqs = [s.replace("U", "T").replace("u", "t") for s in examples[seq_col]]
        return tokenizer(seqs, truncation=True, padding="max_length", max_length=max_length)

    enc_train = raw_train.map(_preprocess, batched=True, remove_columns=[seq_col])
    enc_dev   = raw_dev.map(_preprocess,   batched=True, remove_columns=[seq_col])
    enc_test  = raw_test.map(_preprocess,  batched=True, remove_columns=[seq_col])

    enc_trainval = concatenate_datasets([enc_train, enc_dev])
    labels_trainval = np.array(enc_trainval["label"])

    logger.info(f"  trainval={len(enc_trainval)}, test={len(enc_test)}")

    # ── Builder ───────────────────────────────────────────────
    def _build():
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
        if spec["weights_path"] and os.path.exists(spec["weights_path"]):
            load_lamar_encoder_weights(model, spec["weights_path"])
        return model

    # ── 5-fold CV ─────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.seed)
    fold_metrics_list = []

    for fold_idx, (tr_idx, va_idx) in enumerate(
        skf.split(np.zeros(len(enc_trainval)), labels_trainval)
    ):
        logger.info(f"    fold {fold_idx+1}/{N_FOLDS}  (train={len(tr_idx)}, val={len(va_idx)})")
        fold_train = enc_trainval.select(tr_idx.tolist())
        fold_val   = enc_trainval.select(va_idx.tolist())
        model      = _build()
        fold_dir   = os.path.join(output_dir, f"fold_{fold_idx}")
        trainer    = Trainer(
            model=model,
            args=_make_training_args(fold_dir, args.seed),
            train_dataset=fold_train,
            eval_dataset=fold_val,
            compute_metrics=_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
        )
        trainer.train()
        fold_metrics_list.append(trainer.evaluate(eval_dataset=fold_val))
        shutil.rmtree(fold_dir, ignore_errors=True)

    cv_metrics = _aggregate_cv_metrics(fold_metrics_list)
    logger.info(
        f"  CV auc: {cv_metrics.get('eval_auc_mean', 0):.4f} "
        f"± {cv_metrics.get('eval_auc_std', 0):.4f}"
    )

    # ── Final model on full trainval ──────────────────────────
    logger.info("  Training final model on full trainval …")
    final_model = _build()
    final_dir   = os.path.join(output_dir, "final")
    trainer_final = Trainer(
        model=final_model,
        args=_make_training_args(final_dir, args.seed),
        train_dataset=enc_trainval,
        eval_dataset=enc_test,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )
    trainer_final.train()
    test_metrics = trainer_final.evaluate(eval_dataset=enc_test)
    shutil.rmtree(final_dir, ignore_errors=True)
    logger.info(f"  test_auc={test_metrics.get('eval_auc', 0):.4f}")

    return _build_results(variant, "lamar", args, spec, cv_metrics, fold_metrics_list, test_metrics)


# ═══════════════════════════════════════════════════════════════
#  Run DNABERT-2
# ═══════════════════════════════════════════════════════════════

def run_dnabert2(spec: dict, variant: str, rbp_dir: str, output_dir: str, args) -> dict:
    from transformers.models.bert.configuration_bert import BertConfig

    weights_path = spec["weights_path"]
    tokenizer_path = spec["tokenizer"]
    max_length = spec["max_length"]
    is_random  = not bool(weights_path)
    # For a random model we still need a HF base to initialise architecture
    model_path = weights_path if weights_path else "zhihan1996/DNABERT-2-117M"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # ── Load datasets ─────────────────────────────────────────
    train_ds  = SupervisedDataset(os.path.join(rbp_dir, "train.csv"), tokenizer, max_length)
    dev_ds    = SupervisedDataset(os.path.join(rbp_dir, "dev.csv"),   tokenizer, max_length)
    test_ds   = SupervisedDataset(os.path.join(rbp_dir, "test.csv"),  tokenizer, max_length)
    collator  = DataCollatorForSupervisedDataset(tokenizer)

    trainval_ds  = torch.utils.data.ConcatDataset([train_ds, dev_ds])
    tv_labels    = np.array(train_ds.labels + dev_ds.labels)

    logger.info(f"  trainval={len(trainval_ds)}, test={len(test_ds)}")

    # ── Builder ───────────────────────────────────────────────
    config = BertConfig.from_pretrained(
        "zhihan1996/DNABERT-2-117M", num_labels=2, trust_remote_code=True
    )

    def _build():
        m = AutoModelForSequenceClassification.from_pretrained(
            model_path, config=config,
            trust_remote_code=True, low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
        )
        if is_random:
            m.apply(init_weights)
        return m

    # ── 5-fold CV ─────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.seed)
    fold_metrics_list = []
    all_indices = np.arange(len(trainval_ds))

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(all_indices, tv_labels)):
        logger.info(f"    fold {fold_idx+1}/{N_FOLDS}  (train={len(tr_idx)}, val={len(va_idx)})")
        model    = _build()
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        trainer  = Trainer(
            model=model,
            args=_make_training_args(
                fold_dir, args.seed, extra={"optim": "adamw_torch", "dataloader_pin_memory": True}
            ),
            train_dataset=Subset(trainval_ds, tr_idx.tolist()),
            eval_dataset=Subset(trainval_ds, va_idx.tolist()),
            data_collator=collator,
            compute_metrics=_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
            tokenizer=tokenizer,
        )
        trainer.train()
        fold_metrics_list.append(trainer.evaluate(eval_dataset=Subset(trainval_ds, va_idx.tolist())))
        shutil.rmtree(fold_dir, ignore_errors=True)

    cv_metrics = _aggregate_cv_metrics(fold_metrics_list)
    logger.info(
        f"  CV auc: {cv_metrics.get('eval_auc_mean', 0):.4f} "
        f"± {cv_metrics.get('eval_auc_std', 0):.4f}"
    )

    # ── Final model on full trainval ──────────────────────────
    logger.info("  Training final model on full trainval …")
    final_model = _build()
    final_dir   = os.path.join(output_dir, "final")
    trainer_final = Trainer(
        model=final_model,
        args=_make_training_args(
            final_dir, args.seed, extra={"optim": "adamw_torch", "dataloader_pin_memory": True}
        ),
        train_dataset=trainval_ds,
        eval_dataset=test_ds,
        data_collator=collator,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
        tokenizer=tokenizer,
    )
    trainer_final.train()
    test_metrics = trainer_final.evaluate(eval_dataset=test_ds)
    shutil.rmtree(final_dir, ignore_errors=True)
    logger.info(f"  test_auc={test_metrics.get('eval_auc', 0):.4f}")

    return _build_results(variant, "dnabert2", args, spec, cv_metrics, fold_metrics_list, test_metrics)


# ═══════════════════════════════════════════════════════════════
#  Results builder
# ═══════════════════════════════════════════════════════════════

def _build_results(
    variant: str,
    model_type: str,
    args,
    spec: dict,
    cv_metrics: dict,
    fold_metrics_list: list,
    test_metrics: dict,
) -> dict:
    return {
        "experiment":   "rbp",
        "model_type":   model_type,
        "variant":      variant,
        "rbp_name":     args.rbp_name,
        "weights_path": spec["weights_path"],
        "max_length":   spec["max_length"],
        "n_folds":      N_FOLDS,
        "hyperparameters": {
            "learning_rate":                LEARNING_RATE,
            "per_device_train_batch_size":  BATCH_SIZE,
            "weight_decay":                 WEIGHT_DECAY,
            "warmup_ratio":                 WARMUP_RATIO,
            "num_train_epochs":             NUM_EPOCHS,
            "gradient_accumulation_steps":  GRAD_ACCUM,
            "early_stopping_patience":      PATIENCE,
        },
        "cv_metrics":      cv_metrics,
        "cv_fold_metrics": fold_metrics_list,
        "test_metrics":    {k: v for k, v in test_metrics.items()},
    }


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Fixed-hyperparameter RBP fine-tuning for all DNA-LM variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--variant", required=True, choices=ALL_VARIANTS,
        help="Model variant name (e.g. lamar_pretrained, dnabert2_tapt_v3)",
    )
    p.add_argument(
        "--rbp_name", required=True,
        help="RBP directory name (e.g. HNRNPK_K562_200 or GTF2F1_K562_IDR)",
    )
    p.add_argument(
        "--data_root", default="",
        help="Explicit path to the folder containing <rbp_name>/. "
             "Auto-detected from rbp_name suffix if omitted.",
    )
    p.add_argument(
        "--output_dir", default="./results/rbp",
        help="Base output directory; results go to <output_dir>/<variant>/<rbp_name>/",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    transformers.set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    spec = MODEL_SPECS[args.variant]

    # Resolve data root
    if args.data_root:
        data_root = args.data_root
        rbp_dir   = os.path.join(data_root, args.rbp_name)
    else:
        data_root = _find_data_root(args.rbp_name)
        rbp_dir   = os.path.join(data_root, args.rbp_name)

    if not os.path.isdir(rbp_dir):
        logger.error(f"RBP directory not found: {rbp_dir}")
        sys.exit(1)

    output_dir = os.path.join(args.output_dir, args.variant, args.rbp_name)
    results_path = os.path.join(output_dir, "results.json")

    if os.path.exists(results_path):
        logger.info(f"[SKIP] results.json already exists at: {results_path}")
        sys.exit(0)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 65)
    logger.info(f"  Variant    : {args.variant}")
    logger.info(f"  RBP        : {args.rbp_name}")
    logger.info(f"  Data root  : {data_root}")
    logger.info(f"  Output dir : {output_dir}")
    logger.info(f"  Model type : {spec['type']}  |  max_length={spec['max_length']}")
    logger.info(f"  Weights    : {spec['weights_path'] or '(random init)'}")
    logger.info(f"  LR={LEARNING_RATE}  BS={BATCH_SIZE}  Epochs={NUM_EPOCHS}  "
                f"Patience={PATIENCE}  N_folds={N_FOLDS}")
    logger.info("=" * 65)

    if spec["type"] == "lamar":
        results = run_lamar(spec, args.variant, rbp_dir, output_dir, args)
    else:
        results = run_dnabert2(spec, args.variant, rbp_dir, output_dir, args)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"[saved] {results_path}")


if __name__ == "__main__":
    main()
