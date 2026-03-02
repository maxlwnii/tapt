import os
import csv
import copy
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union
from transformers.models.bert.configuration_bert import BertConfig
import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedKFold

from transformers import EarlyStoppingCallback


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    use_random_init: bool = field(default=False, metadata={"help": "whether to use random initialization"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
    fold: int = field(default=-1, metadata={"help": "Fold number for cross-validation. -1 means no CV."})
    cv_folds: int = field(default=5, metadata={"help": "Number of CV folds."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    do_eval: bool = field(default=True, metadata={"help": "Enable evaluation."})
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    eval_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_accuracy", metadata={"help": "Metric to use for early stopping."})
    greater_is_better: bool = field(default=True, metadata={"help": "Whether the metric is better when higher."})
    early_stopping_patience: int = field(default=3, metadata={"help": "Number of evaluations to wait before early stopping."})
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def init_weights(module):
    """WOLF random initialization."""
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


"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

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
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray, probabilities: np.ndarray = None):
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
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
        # For binary classification, use probabilities for positive class
        if valid_probabilities.shape[1] == 2:
            pos_prob = valid_probabilities[:, 1]
            metrics["auc"] = sklearn.metrics.roc_auc_score(valid_labels, pos_prob)
            metrics["auprc"] = sklearn.metrics.average_precision_score(valid_labels, pos_prob)
        else:
            # For multi-class, compute macro AUC
            metrics["auc"] = sklearn.metrics.roc_auc_score(valid_labels, valid_probabilities, multi_class="ovr", average="macro")
            metrics["auprc"] = sklearn.metrics.average_precision_score(valid_labels, valid_probabilities, average="macro")
    return metrics

"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    # Ensure logits is a torch tensor
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    predictions = np.argmax(logits.numpy(), axis=-1)
    # Get probabilities for AUC
    probabilities = torch.softmax(logits, dim=-1).numpy()
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    return calculate_metric_with_sklearn(predictions, labels, probabilities)


def train():

    print(sys.argv)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    if data_args.fold != -1:
        # Load all data for CV
        all_texts = []
        all_labels = []
        for split in ['train', 'dev', 'test']:
            with open(os.path.join(data_args.data_path, f"{split}.csv"), "r") as f:
                data = list(csv.reader(f))[1:]
                if len(data[0]) == 2:
                    texts = [d[0] for d in data]
                    labels = [int(d[1]) for d in data]
                else:
                    raise ValueError("Data format not supported for CV")
                all_texts.extend(texts)
                all_labels.extend(labels)
        
        # Split into folds
        skf = StratifiedKFold(n_splits=data_args.cv_folds, shuffle=True, random_state=42)
        folds = list(skf.split(all_texts, all_labels))
        
        # Determine folds for train, val, test
        test_fold = data_args.fold - 1
        val_fold = data_args.fold % data_args.cv_folds
        train_folds = [i for i in range(data_args.cv_folds) if i not in [test_fold, val_fold]]
        
        # Collect indices
        train_indices = []
        for f in train_folds:
            train_indices.extend(folds[f][0])
            train_indices.extend(folds[f][1])
        val_indices = np.concatenate([folds[val_fold][0], folds[val_fold][1]])
        test_indices = np.concatenate([folds[test_fold][0], folds[test_fold][1]])
        
        # Create data lists
        train_data = [(all_texts[i], all_labels[i]) for i in train_indices]
        val_data = [(all_texts[i], all_labels[i]) for i in val_indices]
        test_data = [(all_texts[i], all_labels[i]) for i in test_indices]
        
        # Create temporary CSV files
        import tempfile
        train_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        val_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        test_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        for csv_file, data in [(train_csv, train_data), (val_csv, val_data), (test_csv, test_data)]:
            csv_file.write("sequence,label\n")
            for seq, label in data:
                csv_file.write(f"{seq},{label}\n")
            csv_file.close()
        
        # Load datasets
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=train_csv.name, kmer=data_args.kmer)
        val_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=val_csv.name, kmer=data_args.kmer)
        test_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=test_csv.name, kmer=data_args.kmer)
        
        # Clean up temp files
        os.unlink(train_csv.name)
        os.unlink(val_csv.name)
        os.unlink(test_csv.name)
    else:
        # Original loading
        train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                          data_path=os.path.join(data_args.data_path, "train.csv"), 
                                          kmer=data_args.kmer)
        val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                         data_path=os.path.join(data_args.data_path, "dev.csv"), 
                                         kmer=data_args.kmer)
        test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                         data_path=os.path.join(data_args.data_path, "test.csv"), 
                                         kmer=data_args.kmer)
    
    # Update run_name and output_dir for CV
    if data_args.fold != -1:
        training_args.run_name = f"{training_args.run_name}_fold{data_args.fold}"
        training_args.output_dir = f"{training_args.output_dir}_fold{data_args.fold}"
    
    config = BertConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=train_dataset.num_labels,
        cache_dir=training_args.cache_dir,
    )

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        config=config
    )

    # Apply random initialization if requested
    if model_args.use_random_init:
        model.apply(init_weights)

    # configure LoRA
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator,
                                   callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)])
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)


if __name__ == "__main__":    
    train()