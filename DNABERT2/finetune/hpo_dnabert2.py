"""
Hyperparameter Optimization for DNABERT2 Fine-tuning with Optuna.

Runs Optuna trials to search over learning rate, batch size, weight decay,
warmup ratio, epochs, gradient accumulation, and max sequence length.

Supports both:
  - CSV data   (DNABERT2/data/<RBP>/         with train.csv, dev.csv, test.csv)
  - Koo data   (data/finetune_data_koo/<RBP>/  with train.csv, dev.csv, test.csv)
  These are DIFFERENT datasets with DIFFERENT RBPs.

Usage:
  python hpo_dnabert2.py \
      --rbp_name PTBP1_K562_200 \
      --dataset koo \
      --n_trials 30 \
      --output_dir ./hpo_results/dnabert2

Author: Maximilian Lewin
"""

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import optuna
import sklearn.metrics
import torch
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.models.bert.configuration_bert import BertConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════

THESIS_ROOT = os.environ.get(
    "THESIS_ROOT",
    "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis",
)

DATASET_PATHS = {
    "csv": os.path.join(THESIS_ROOT, "DNABERT2", "data"),          # CSV IDR data
    "koo": os.path.join(THESIS_ROOT, "data", "finetune_data_koo"),  # eclip_koo data
}

RBPS_CSV = [
    "GTF2F1_K562_IDR",
    "HNRNPL_K562_IDR",
    "HNRNPM_HepG2_IDR",
    "ILF3_HepG2_IDR",
    "KHSRP_K562_IDR",
    "MATR3_K562_IDR",
    "PTBP1_HepG2_IDR",
    "QKI_K562_IDR",
]

RBPS_KOO = [
    "HNRNPK_K562_200",
    "PTBP1_K562_200",
    "PUM2_K562_200",
    "QKI_K562_200",
    "RBFOX2_K562_200",
    "SF3B4_K562_200",
    "SRSF1_K562_200",
    "TARDBP_K562_200",
    "TIA1_K562_200",
    "U2AF1_K562_200",
]

RBPS = sorted(set(RBPS_CSV + RBPS_KOO))  # union for argparse choices

MODEL_NAME = "zhihan1996/DNABERT-2-117M"


def init_weights(module):
    """WOLF random initialization (matches train.py)."""
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


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning from CSV."""

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


# Koo data is now CSV (eclip_koo), same format as csv dataset.
# No Arrow conversion needed — load CSVs directly.


# ═══════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════

def compute_metrics(eval_pred):
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
        "f1": float(sklearn.metrics.f1_score(labs, preds, average="macro", zero_division=0)),
        "matthews_correlation": float(sklearn.metrics.matthews_corrcoef(labs, preds)),
        "precision": float(sklearn.metrics.precision_score(labs, preds, average="macro", zero_division=0)),
        "recall": float(sklearn.metrics.recall_score(labs, preds, average="macro", zero_division=0)),
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
#  Collator
# ═══════════════════════════════════════════════════════════════

class DataCollatorForSupervisedDataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.tensor(labels, dtype=torch.long)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# ═══════════════════════════════════════════════════════════════
#  Objective
# ═══════════════════════════════════════════════════════════════

def make_objective(args):
    """Build an Optuna objective closure over the args namespace."""

    def objective(trial: optuna.Trial) -> float:
        # ── Sample hyperparameters ────────────────────────────
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
        batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16])
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15, step=0.05)
        num_epochs = trial.suggest_int("num_train_epochs", 3, 10)
        grad_accum = trial.suggest_categorical("gradient_accumulation_steps", [1, 2])
        max_length = trial.suggest_categorical("model_max_length", [25, 50, 128])

        trial_dir = os.path.join(args.output_dir, args.rbp_name, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)

        # ── Tokenizer ────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            model_max_length=max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

        # ── Data ──────────────────────────────────────────────
        if args.dataset == "csv":
            data_dir = os.path.join(DATASET_PATHS["csv"], args.rbp_name)
            train_ds = SupervisedDataset(os.path.join(data_dir, "train.csv"), tokenizer, max_length)
            val_ds = SupervisedDataset(os.path.join(data_dir, "dev.csv"), tokenizer, max_length)
            test_ds = SupervisedDataset(os.path.join(data_dir, "test.csv"), tokenizer, max_length)
        else:
            data_dir = os.path.join(DATASET_PATHS["koo"], args.rbp_name)
            train_ds = SupervisedDataset(os.path.join(data_dir, "train.csv"), tokenizer, max_length)
            val_ds = SupervisedDataset(os.path.join(data_dir, "dev.csv"), tokenizer, max_length)
            test_ds = SupervisedDataset(os.path.join(data_dir, "test.csv"), tokenizer, max_length)

        # ── Subsample for speed if requested ──────────────────
        if args.max_train_samples:
            n = min(args.max_train_samples, len(train_ds))
            indices = list(range(n))
            train_ds = torch.utils.data.Subset(train_ds, indices)
            # Subset doesn't propagate num_labels, stash it
            train_ds.num_labels = 2

        num_labels = getattr(train_ds, "num_labels", 2)

        # ── Model ─────────────────────────────────────────────
        config = BertConfig.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            config=config,
            trust_remote_code=True,
        )
        if args.use_random_init:
            model.apply(init_weights)
            logger.info("Applied random initialization (WOLF)")

        # ── Training args ─────────────────────────────────────
        training_args = TrainingArguments(
            output_dir=trial_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            optim="adamw_torch",
            fp16=args.fp16,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            greater_is_better=True,
            report_to="none",
            seed=42,
            dataloader_pin_memory=True,
        )

        collator = DataCollatorForSupervisedDataset(tokenizer)

        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            tokenizer=tokenizer,
        )

        # ── Train ─────────────────────────────────────────────
        trainer.train()

        # ── Evaluate on val (objective) ───────────────────────
        val_metrics = trainer.evaluate(eval_dataset=val_ds)
        auc = val_metrics.get("eval_auc", 0.5)

        # ── Also evaluate on test (informational) ─────────────
        test_metrics = trainer.evaluate(eval_dataset=test_ds)

        # Save trial results
        result = {
            "trial": trial.number,
            "params": trial.params,
            "val_auc": auc,
            "val_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in val_metrics.items()},
            "test_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in test_metrics.items()},
        }
        with open(os.path.join(trial_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Trial {trial.number}: val_auc={auc:.4f}, test_auc={test_metrics.get('eval_auc', 0):.4f}")
        return auc

    return objective


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="HPO for DNABERT2 fine-tuning with Optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rbp_name", type=str, required=True, choices=RBPS,
                   help="RBP to fine-tune on")
    p.add_argument("--dataset", type=str, required=True, choices=["csv", "koo"],
                   help="Dataset to use: 'csv' (DNABERT2/data) or 'koo' (finetune_data_koo)")
    p.add_argument("--output_dir", type=str, default="./hpo_results/dnabert2",
                   help="Base output directory for HPO results")
    p.add_argument("--n_trials", type=int, default=30,
                   help="Number of Optuna trials")
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Limit training samples per trial (for testing)")
    p.add_argument("--fp16", action="store_true",
                   help="Use mixed precision")
    p.add_argument("--use_random_init", action="store_true",
                   help="Use random weight initialization instead of pretrained")
    p.add_argument("--study_name", type=str, default=None,
                   help="Optuna study name (default: dnabert2_<rbp>_<dataset>)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    transformers.set_seed(args.seed)

    init_tag = "random" if args.use_random_init else "pretrained"
    study_name = args.study_name or f"dnabert2_{init_tag}_{args.rbp_name}_{args.dataset}"
    output_dir = os.path.join(args.output_dir, f"{args.rbp_name}_{args.dataset}")
    args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Optuna study — maximize validation AUC
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        storage=f"sqlite:///{os.path.join(output_dir, 'optuna.db')}",
        load_if_exists=True,
    )

    study.optimize(make_objective(args), n_trials=args.n_trials)

    # ── Report ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("HPO Complete!")
    logger.info(f"  Best trial : {study.best_trial.number}")
    logger.info(f"  Best AUC   : {study.best_value:.4f}")
    logger.info(f"  Best params:")
    for k, v in study.best_params.items():
        logger.info(f"    {k}: {v}")
    logger.info("=" * 60)

    # Save summary
    summary = {
        "study_name": study_name,
        "rbp_name": args.rbp_name,
        "dataset": args.dataset,
        "n_trials": args.n_trials,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    with open(os.path.join(output_dir, "hpo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {os.path.join(output_dir, 'hpo_summary.json')}")


if __name__ == "__main__":
    main()
