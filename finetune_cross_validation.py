#!/usr/bin/env python3
"""
Unified finetuning script for DNABERT2 and LAMAR models with 5-fold cross-validation.

This script handles:
  - Cross-cell and cross-length datasets
  - Multiple model variants (random, pretrained, checkpoint)
  - 5-fold cross-validation with proper train/val/test splits
  - Automatic k-mer processing
  - Metrics calculation and result saving
  - Clean model weight saving per fold

Usage:
  python finetune_cross_validation.py \
    --model_name dnabert2 \
    --variant random \
    --dataset_path /path/to/dataset \
    --output_dir /path/to/output \
    --fold 1

Author: Maximilian Lewin
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import sklearn.metrics
import torch
import transformers
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    model_name: str = "dnabert2"  # dnabert2 or lamar
    variant: str = "random"  # random, pretrained, checkpoint (for LAMAR: tapt)
    model_path: str = ""
    tokenizer_path: str = ""
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "query,value"


@dataclass
class DataConfig:
    """Configuration for data loading."""
    dataset_path: str = ""
    kmer: int = -1  # -1 means no k-mer
    fold: int = 1  # which fold (1-5)
    cv_folds: int = 5


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 3
    model_max_length: int = 512
    seed: int = 42
    fp16: bool = False


def init_weights(module):
    """Random initialization for weights."""
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


def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logger.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        logger.warning(f"Generating {k}-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logger.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)

    return kmer


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        kmer: int = -1,
    ):
        super(SupervisedDataset, self).__init__()

        # Load data from disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]

        if len(data[0]) == 2:
            # Single sequence classification
            logger.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # Sequence-pair classification
            logger.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        if kmer != -1:
            logger.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        return dict(input_ids=input_ids, labels=torch.tensor(labels))


def calculate_metrics_with_sklearn(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray = None,
) -> Dict[str, float]:
    """Calculate metrics using sklearn."""
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    
    metrics = {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }
    
    if probabilities is not None:
        valid_probabilities = probabilities[valid_mask]
        if valid_probabilities.shape[1] == 2:
            metrics["auc"] = sklearn.metrics.roc_auc_score(
                valid_labels, valid_probabilities[:, 1]
            )
        else:
            metrics["auc"] = sklearn.metrics.roc_auc_score(
                valid_labels, valid_probabilities, multi_class="ovr", zero_division=0
            )

    return metrics


def compute_metrics(eval_pred):
    """Compute metrics for trainer."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    predictions = np.argmax(logits.numpy(), axis=-1)
    probabilities = torch.softmax(logits, dim=-1).numpy()
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    return calculate_metrics_with_sklearn(predictions, labels, probabilities)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Save model state dict to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def load_datasets_with_cv(
    data_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    fold: int,
    cv_folds: int = 5,
    kmer: int = -1,
) -> Tuple[SupervisedDataset, SupervisedDataset, SupervisedDataset]:
    """Load datasets with cross-validation split."""
    
    # Load all data
    all_texts = []
    all_labels = []
    
    for split in ["train", "dev", "test"]:
        csv_path = os.path.join(data_path, f"{split}.csv")
        with open(csv_path, "r") as f:
            data = list(csv.reader(f))[1:]
        
        if len(data[0]) == 2:
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        else:
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        
        all_texts.extend(texts)
        all_labels.extend(labels)
    
    # Stratified k-fold split
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    folds = list(skf.split(all_texts, all_labels))
    
    test_fold = fold - 1
    val_fold = fold % cv_folds
    train_folds = [i for i in range(cv_folds) if i not in [test_fold, val_fold]]
    
    # Collect indices
    train_indices = []
    for f in train_folds:
        train_indices.extend(list(folds[f][0]) + list(folds[f][1]))
    
    val_indices = np.concatenate([folds[val_fold][0], folds[val_fold][1]])
    test_indices = np.concatenate([folds[test_fold][0], folds[test_fold][1]])
    
    # Create data lists
    train_data = [(all_texts[i], all_labels[i]) for i in train_indices]
    val_data = [(all_texts[i], all_labels[i]) for i in val_indices]
    test_data = [(all_texts[i], all_labels[i]) for i in test_indices]
    
    # Create temporary CSV files
    train_csv = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    val_csv = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    test_csv = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    
    for csv_file, data in [(train_csv, train_data), (val_csv, val_data), (test_csv, test_data)]:
        writer = csv.writer(csv_file)
        writer.writerow(["sequence", "label"])
        for seq, label in data:
            if isinstance(seq, list):
                writer.writerow(seq + [label])
            else:
                writer.writerow([seq, label])
        csv_file.close()
    
    # Load datasets
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=train_csv.name, kmer=kmer
    )
    val_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=val_csv.name, kmer=kmer
    )
    test_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=test_csv.name, kmer=kmer
    )
    
    # Clean up temp files
    os.unlink(train_csv.name)
    os.unlink(val_csv.name)
    os.unlink(test_csv.name)
    
    return train_dataset, val_dataset, test_dataset


def finetune(
    model_config: ModelConfig,
    data_config: DataConfig,
    training_config: TrainingConfig,
    output_dir: str,
) -> Dict[str, Any]:
    """Main finetuning function."""
    
    # Set seed
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.warning(f"Loading tokenizer from {model_config.tokenizer_path}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_config.tokenizer_path,
        model_max_length=training_config.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    
    if "InstaDeep" in model_config.tokenizer_path:
        tokenizer.eos_token = tokenizer.pad_token
    
    logger.warning("Loading datasets with cross-validation...")
    train_dataset, val_dataset, test_dataset = load_datasets_with_cv(
        data_config.dataset_path,
        tokenizer,
        data_config.fold,
        data_config.cv_folds,
        data_config.kmer,
    )
    
    logger.warning(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    # Load model
    logger.warning(f"Loading model from {model_config.model_path}...")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_config.model_path,
        num_labels=train_dataset.num_labels,
        trust_remote_code=True,
    )
    
    # Apply random initialization if variant is random
    if model_config.variant == "random":
        logger.warning("Applying random initialization...")
        model.apply(init_weights)
    
    # Configure LoRA if requested
    if model_config.use_lora and LoraConfig is not None:
        logger.warning("Configuring LoRA...")
        lora_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=list(model_config.lora_target_modules.split(",")),
            lora_dropout=model_config.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Calculate steps
    steps_per_epoch = math.ceil(
        len(train_dataset)
        / (training_config.batch_size * training_config.gradient_accumulation_steps)
    )
    eval_save_steps = max(1, steps_per_epoch // 2)
    
    # Training arguments
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        learning_rate=training_config.learning_rate,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        num_train_epochs=training_config.epochs,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        warmup_steps=0,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=eval_save_steps,
        eval_steps=eval_save_steps,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        logging_steps=eval_save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        fp16=training_config.fp16,
        seed=training_config.seed,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # Create trainer
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_config.early_stopping_patience
            )
        ],
    )
    
    logger.warning("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    logger.warning("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    
    # Save results
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Save model weights
    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    safe_save_model_for_hf_trainer(trainer, model_dir)
    
    logger.warning(f"Test results: {test_results}")
    logger.warning(f"Results saved to {results_dir}")
    
    return test_results


def main():
    parser = argparse.ArgumentParser(
        description="Finetune DNABERT2/LAMAR with cross-validation"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["dnabert2", "lamar"],
        help="Model to finetune",
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Model variant (random, pretrained, checkpoint/tapt)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for efficient finetuning",
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (e.g., HNRNPK_train_K562_test_HepG2_fixlen_101)",
    )
    parser.add_argument(
        "--kmer",
        type=int,
        default=-1,
        help="K-mer size (-1 for no k-mer)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Cross-validation fold (1-5)",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision",
    )
    
    args = parser.parse_args()
    
    # Validate fold number
    if args.fold < 1 or args.fold > 5:
        print("ERROR: --fold must be between 1 and 5")
        sys.exit(1)
    
    # Create configs
    model_config = ModelConfig(
        model_name=args.model_name,
        variant=args.variant,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        use_lora=args.use_lora,
    )
    
    data_config = DataConfig(
        dataset_path=args.dataset_path,
        kmer=args.kmer,
        fold=args.fold,
        cv_folds=5,
    )
    
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        fp16=args.fp16,
    )
    
    # Run finetuning
    finetune(model_config, data_config, training_config, args.output_dir)


if __name__ == "__main__":
    main()
