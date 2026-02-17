"""
Hyperparameter Optimization for LAMAR Fine-tuning with Optuna.

Runs Optuna trials to search over learning rate, batch size, weight decay,
warmup ratio, epochs, gradient accumulation, freeze_encoder, and warmup_epochs.

Supports both:
  - Koo data   (data/finetune_data_koo/<RBP>/  with train.csv, dev.csv, test.csv)
  - CSV data   (data/finetune_data_koo/<RBP>/  with train.csv, dev.csv, test.csv)
  Both currently point to the same eclip_koo-derived CSV data.

Usage:
  python hpo_lamar.py \
      --rbp_name PTBP1_K562_200 \
      --dataset koo \
      --pretrain_path /path/to/model.safetensors \
      --n_trials 30 \
      --output_dir ./hpo_results/lamar

Author: Maximilian Lewin
"""

import argparse
import json
import logging
import os
import sys
import traceback

import numpy as np
import optuna
import pandas as pd
import torch
from datasets import DatasetDict
from safetensors.torch import load_file
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    EsmConfig,
    EsmForSequenceClassification,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

THESIS_ROOT = os.environ.get(
    "THESIS_ROOT",
    "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis",
)

DATASET_PATHS = {
    "koo": os.path.join(THESIS_ROOT, "data", "finetune_data_koo"),
    "csv": os.path.join(THESIS_ROOT, "data", "finetune_data_koo"),
}

RBPS = [
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

DEFAULT_TOKENIZER = os.path.join(
    THESIS_ROOT, "pretrain", "saving_model", "tapt_lamar", "checkpoint-100000"
)


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def init_weights(module):
    """Random initialization matching Wolf / LAMAR convention."""
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


def load_encoder_weights(model, weights_path):
    """Load ONLY encoder weights from safetensors, leaving classifier random."""
    state_dict = load_file(weights_path)
    encoder_weights = {}
    for k, v in state_dict.items():
        if "lm_head" in k or "classifier" in k:
            continue
        if k.startswith("esm."):
            encoder_weights[k] = v
        else:
            encoder_weights["esm." + k] = v
    missing, unexpected = model.load_state_dict(encoder_weights, strict=False)
    logger.info(f"Loaded {len(encoder_weights)} encoder tensors, {len(missing)} missing, {len(unexpected)} unexpected")
    return model


def freeze_encoder(model, freeze=True):
    """Freeze/unfreeze encoder layers, keep classifier trainable."""
    for name, param in model.named_parameters():
        if name.startswith("esm."):
            param.requires_grad = not freeze
        else:
            param.requires_grad = True


def compute_metrics(p):
    predictions, labels = p
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()
    pred_labels = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_labels, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, pred_labels)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc = 0.5
    try:
        auprc = average_precision_score(labels, probs[:, 1])
    except Exception:
        auprc = 0.5
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
        "auprc": float(auprc),
    }


def load_dataset_for_lamar(dataset_type: str, rbp_name: str):
    """Load dataset as a DatasetDict with train/validation/test splits."""
    if dataset_type == "koo":
        # Koo data is now CSV (eclip_koo-derived)
        data_dir = os.path.join(DATASET_PATHS["koo"], rbp_name)
        from datasets import load_dataset as load_ds
        csvs = {
            "train": os.path.join(data_dir, "train.csv"),
            "validation": os.path.join(data_dir, "dev.csv"),
            "test": os.path.join(data_dir, "test.csv"),
        }
        ds = load_ds("csv", data_files=csvs)
        return ds
    else:
        # CSV data — load with datasets library
        data_dir = os.path.join(DATASET_PATHS["csv"], rbp_name)
        from datasets import load_dataset as load_ds
        csvs = {
            "train": os.path.join(data_dir, "train.csv"),
            "validation": os.path.join(data_dir, "dev.csv"),
            "test": os.path.join(data_dir, "test.csv"),
        }
        ds = load_ds("csv", data_files=csvs)
        return ds


# ═══════════════════════════════════════════════════════════════
#  Objective
# ═══════════════════════════════════════════════════════════════

def make_objective(args, tokenizer, dataset):
    """Build an Optuna objective closure."""

    # Determine sequence column
    sample_split = dataset["train"]
    if "seq" in sample_split.column_names:
        seq_col = "seq"
    elif "sequence" in sample_split.column_names:
        seq_col = "sequence"
    else:
        raise ValueError(f"No 'seq' or 'sequence' column. Found: {sample_split.column_names}")

    def preprocess_function(examples):
        seqs = [s.replace("U", "T").replace("u", "t") for s in examples[seq_col]]
        return tokenizer(seqs, truncation=True, padding="max_length", max_length=101)

    # Tokenize once (shared across trials since tokenizer is fixed)
    remove_cols = [seq_col]
    encoded = dataset.map(preprocess_function, batched=True, remove_columns=remove_cols)

    def objective(trial: optuna.Trial) -> float:
        # ── Sample hyperparameters ────────────────────────────
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
        batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16])
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15, step=0.05)
        num_epochs = trial.suggest_int("num_train_epochs", 3, 10)
        grad_accum = trial.suggest_categorical("gradient_accumulation_steps", [1, 2])
        do_freeze = trial.suggest_categorical("freeze_encoder", [True, False])
        warmup_epochs = trial.suggest_int("warmup_epochs", 0, 2) if not do_freeze else 0

        trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)

        # ── Subsample if requested ────────────────────────────
        train_ds = encoded["train"]
        val_ds = encoded["validation"]
        test_ds = encoded["test"]

        if args.max_train_samples and args.max_train_samples < len(train_ds):
            train_ds = train_ds.shuffle(seed=42).select(range(args.max_train_samples))

        # ── Model ─────────────────────────────────────────────
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

        if args.pretrain_path and os.path.exists(args.pretrain_path):
            load_encoder_weights(model, args.pretrain_path)

        if do_freeze:
            freeze_encoder(model, freeze=True)

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
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
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
            dataloader_drop_last=True,
        )

        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

        # ── Warmup phase (classifier only) ────────────────────
        if warmup_epochs > 0 and not do_freeze:
            freeze_encoder(model, freeze=True)
            warmup_args = TrainingArguments(
                output_dir=os.path.join(trial_dir, "warmup"),
                num_train_epochs=warmup_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=16,
                learning_rate=lr,
                weight_decay=weight_decay,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                load_best_model_at_end=False,
                logging_steps=50,
                report_to="none",
                seed=42,
            )
            warmup_trainer = Trainer(
                model=model,
                args=warmup_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=compute_metrics,
            )
            warmup_trainer.train()
            freeze_encoder(model, freeze=False)

        # ── Main training ─────────────────────────────────────
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        trainer.train()

        # ── Evaluate ──────────────────────────────────────────
        val_metrics = trainer.evaluate(eval_dataset=val_ds)
        auc = val_metrics.get("eval_auc", 0.5)

        test_metrics = trainer.evaluate(eval_dataset=test_ds)

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
        description="HPO for LAMAR fine-tuning with Optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rbp_name", type=str, required=True, choices=RBPS)
    p.add_argument("--dataset", type=str, required=True, choices=["koo", "csv"],
                   help="Dataset: 'koo' (eclip_koo CSV) or 'csv' (same data, both point to finetune_data_koo)")
    p.add_argument("--pretrain_path", type=str, default="",
                   help="Path to pretrained encoder weights (safetensors)")
    p.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER,
                   help="Path to LAMAR tokenizer")
    p.add_argument("--output_dir", type=str, default="./hpo_results/lamar",
                   help="Base output directory")
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Limit training samples per trial (for testing)")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--study_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    study_name = args.study_name or f"lamar_{args.rbp_name}_{args.dataset}"
    output_dir = os.path.join(args.output_dir, f"{args.rbp_name}_{args.dataset}")
    args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    logger.info(f"  vocab_size={len(tokenizer)}")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset} / {args.rbp_name}")
    dataset = load_dataset_for_lamar(args.dataset, args.rbp_name)
    logger.info(f"  splits: { {k: len(v) for k, v in dataset.items()} }")

    # Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=f"sqlite:///{os.path.join(output_dir, 'optuna.db')}",
        load_if_exists=True,
    )

    study.optimize(make_objective(args, tokenizer, dataset), n_trials=args.n_trials)

    # Report
    logger.info("=" * 60)
    logger.info("HPO Complete!")
    logger.info(f"  Best trial : {study.best_trial.number}")
    logger.info(f"  Best AUC   : {study.best_value:.4f}")
    for k, v in study.best_params.items():
        logger.info(f"    {k}: {v}")
    logger.info("=" * 60)

    summary = {
        "study_name": study_name,
        "rbp_name": args.rbp_name,
        "dataset": args.dataset,
        "pretrain_path": args.pretrain_path,
        "n_trials": args.n_trials,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
    }
    with open(os.path.join(output_dir, "hpo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {os.path.join(output_dir, 'hpo_summary.json')}")


if __name__ == "__main__":
    main()
