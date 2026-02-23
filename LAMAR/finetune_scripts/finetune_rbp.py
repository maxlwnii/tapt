import argparse
import os
import torch
import numpy as np
import pandas as pd
import json
from datasets import load_from_disk, concatenate_datasets
from safetensors.torch import load_file
from transformers import (
    EsmConfig,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    EarlyStoppingCallback,
    EsmForSequenceClassification
)
import sys
# Ensure project root is on PYTHONPATH so `import LAMAR...` works when running from the finetune_scripts folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Compute project root path for resolving relative resources
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
inner_lamar = os.path.join(project_root, 'LAMAR')
#from LAMAR.sequence_classification_patch import EsmForSequenceClassification
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, average_precision_score

TOKENIZER_PATH = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/src/pretrain/saving_model/tapt_lamar/checkpoint-100000/"

def compute_metrics(p):
    predictions, labels = p
    # If model returns tuple (logits, ...)
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # If single-output regression (num_labels == 1) - detect by labels dtype or shape
    try:
        # flatten
        preds = np.squeeze(np.array(predictions))
        labs = np.squeeze(np.array(labels))
    except Exception:
        preds = predictions
        labs = labels

    results = {}
    # Regression branch when labels look continuous or model outputs single value
    if preds.ndim == 1 or (isinstance(preds, np.ndarray) and (preds.shape[-1] == 1)):
        # Ensure 1D arrays
        preds_1d = preds.reshape(-1)
        labs_1d = labs.reshape(-1)
        mse = float(np.mean((preds_1d - labs_1d) ** 2))
        df = pd.DataFrame({'pred': preds_1d, 'label': labs_1d})
        try:
            corr_coef_pearson = float(df.corr(method='pearson').iloc[0, 1])
        except Exception:
            corr_coef_pearson = float('nan')
        try:
            corr_coef_spearman = float(df.corr(method='spearman').iloc[0, 1])
        except Exception:
            corr_coef_spearman = float('nan')

        results.update({
            'mse': mse,
            'corr_coef_pearson': corr_coef_pearson,
            'corr_coef_spearman': corr_coef_spearman,
        })
    else:
        # Classification branch
        probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()
        pred_labels = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_labels, average='binary', zero_division=0)
        acc = accuracy_score(labels, pred_labels)
        try:
            auc = roc_auc_score(labels, probs[:, 1])
        except Exception:
            auc = 0.5
        try:
            auprc = average_precision_score(labels, probs[:, 1])
        except Exception:
            auprc = 0.5

        results.update({
            'accuracy': float(acc),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'auc': float(auc),
            'auprc': float(auprc),
        })

    return results

# Wolf initialization
def init_weights(module):
    """Custom random initialization for LAMAR model."""
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
    print(f"\n{'='*60}")
    print(f"Loading encoder weights from: {weights_path}")
    print(f"{'='*60}")
    
    state_dict = load_file(weights_path)
    
    # Filter to keep ONLY encoder weights (esm.*), exclude lm_head and classifier
    encoder_weights = {}
    for k, v in state_dict.items():
        # Skip language model head and classifier
        if 'lm_head' in k or 'classifier' in k:
            continue
            
        # Ensure proper esm. prefix for encoder
        if k.startswith("esm."):
            encoder_weights[k] = v
        else:
            # Add esm. prefix if missing
            encoder_weights["esm." + k] = v
    
    print(f"Found {len(encoder_weights)} encoder weight tensors")
    
    # Load encoder weights only (strict=False for classifier mismatch)
    missing_keys, unexpected_keys = model.load_state_dict(encoder_weights, strict=False)
    
    # Verify loading
    encoder_loaded = [k for k in missing_keys if k.startswith('esm.')]
    classifier_missing = [k for k in missing_keys if 'classifier' in k]
    
    print(f"\n✓ Encoder weights loaded: {len(encoder_weights)} tensors")
    print(f"✓ Classifier randomly initialized: {len(classifier_missing)} tensors")
    
    if encoder_loaded:
        print(f"⚠ WARNING: {len(encoder_loaded)} encoder weights missing!")
        print(f"  Sample missing: {encoder_loaded[:3]}")
    
    if unexpected_keys:
        print(f"⚠ WARNING: {len(unexpected_keys)} unexpected keys")
        print(f"  Sample unexpected: {unexpected_keys[:3]}")
    
    print(f"{'='*60}\n")
    
    return model


def freeze_encoder(model, freeze=True):
    """Freeze encoder layers, keep classifier trainable."""
    for name, param in model.named_parameters():
        if name.startswith('esm.'):
            param.requires_grad = not freeze
        else:  # classifier layers
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--rbp_name", type=str, required=True)
        parser.add_argument("--data_path", type=str, required=True)
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--pretrain_path", type=str, default="")
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--lr", type=float, default=3e-5)
        parser.add_argument("--freeze_encoder", action="store_true", 
                            help="Freeze encoder, train only classifier (recommended for small datasets)")
        parser.add_argument("--warmup_epochs", type=int, default=0,
                            help="Number of epochs to train only classifier before unfreezing encoder")
        parser.add_argument("--early_stopping_patience", type=int, default=None,
                            help="Patience for early stopping (None to disable)")
        parser.add_argument("--subsample_pos", type=int, default=None,
                            help="Number of positive samples to subsample (for limited data)")
        parser.add_argument("--subsample_neg", type=int, default=None,
                            help="Number of negative samples to subsample (for limited data)")
        parser.add_argument("--cv_folds", type=int, default=5,
                            help="Number of cross-validation folds (1 for no CV)")
        parser.add_argument("--nlabels", type=int, default=2, help="Number of labels (set 1 for regression)")
        parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio for scheduler")
        parser.add_argument("--grad_clipping_norm", type=float, default=1.0, help="Max norm for gradient clipping")
        parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
        parser.add_argument("--save_epochs", type=int, default=10, help="Save every N epochs/steps")
        parser.add_argument("--logging_steps", type=int, default=100, help="Logging interval in steps")
        parser.add_argument("--fp16", action='store_true', help="Enable fp16 training")
        parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH, help="Tokenizer path override")
        parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
        
        args = parser.parse_args()
        
        print(f"\n{'='*60}", flush=True)
        print(f"Finetuning Configuration", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"RBP: {args.rbp_name}", flush=True)
        print(f"Data: {args.data_path}", flush=True)
        print(f"Output: {args.output_dir}", flush=True)
        print(f"Pretrain: {args.pretrain_path if args.pretrain_path else 'None (Random Init)'}", flush=True)
        print(f"Freeze encoder: {args.freeze_encoder}", flush=True)
        print(f"Warmup epochs: {args.warmup_epochs}", flush=True)
        print(f"Early stopping patience: {args.early_stopping_patience if args.early_stopping_patience else 'Disabled'}", flush=True)
        print(f"CV folds: {args.cv_folds}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Load tokenizer
        # Resolve tokenizer path: prefer local directories (absolute or relative to project root).
        requested_tokenizer_path = args.tokenizer_path
        resolved_tokenizer_path = requested_tokenizer_path
        # If path isn't absolute and doesn't exist, try repo-root and LAMAR/ subfolder
        if not os.path.isabs(resolved_tokenizer_path) or not os.path.exists(resolved_tokenizer_path):
            alt1 = os.path.join(project_root, requested_tokenizer_path)
            alt2 = os.path.join(project_root, 'LAMAR', requested_tokenizer_path)
            if os.path.exists(alt1):
                resolved_tokenizer_path = alt1
            elif os.path.exists(alt2):
                resolved_tokenizer_path = alt2

        print(f"Loading tokenizer from {resolved_tokenizer_path} (requested: {requested_tokenizer_path})", flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer_path)
        except Exception as e:
            # If HF attempted and failed due to authentication or missing repo, give actionable message
            msg = str(e)
            if 'Repository Not Found' in msg or '401' in msg or 'is not a local folder' in msg:
                raise RuntimeError(
                    f"Failed to load tokenizer from '{resolved_tokenizer_path}'.\n"
                    "If this is a local tokenizer directory, ensure the path exists.\n"
                    "If you intended to load from the Hugging Face Hub, authenticate with `huggingface-cli login` or pass a valid repo id.\n"
                    f"Original error: {e}"
                ) from e
            else:
                raise
        print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})\n", flush=True)
        
        # Load and preprocess dataset
        print("Loading dataset...", flush=True)
        # If a directory with CSVs is provided, load CSV files; otherwise try load_from_disk
        data_path = args.data_path
        dataset = None
        if os.path.isdir(data_path):
            # look for train/validation/test csv files
            train_csv = os.path.join(data_path, 'train.csv')
            val_csv = os.path.join(data_path, 'validation.csv')
            # Accept common alternative name 'dev.csv' for validation
            dev_csv = os.path.join(data_path, 'dev.csv')
            dev_csv_alt = os.path.join(data_path, 'development.csv')
            test_csv = os.path.join(data_path, 'test.csv')
            # If validation file not present, prefer dev.csv or development.csv
            if not os.path.exists(val_csv):
                if os.path.exists(dev_csv):
                    val_csv = dev_csv
                elif os.path.exists(dev_csv_alt):
                    val_csv = dev_csv_alt
            csvs = {}
            if os.path.exists(train_csv):
                csvs['train'] = train_csv
            if os.path.exists(val_csv):
                csvs['validation'] = val_csv
            if os.path.exists(test_csv):
                csvs['test'] = test_csv

            # fallback: load any csvs in the directory and assign them to train/test if names differ
            if not csvs:
                files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
                files = sorted(files)
                if len(files) == 1:
                    csvs['train'] = os.path.join(data_path, files[0])
                elif len(files) >= 3:
                    csvs['train'] = os.path.join(data_path, files[0])
                    csvs['validation'] = os.path.join(data_path, files[1])
                    csvs['test'] = os.path.join(data_path, files[2])

            if csvs:
                from datasets import load_dataset as load_ds
                dataset = load_ds('csv', data_files=csvs)
                print(f"✓ CSV dataset loaded from {data_path}: splits={list(dataset.keys())}", flush=True)
        
        if dataset is None:
            try:
                dataset = load_from_disk(data_path)
                print(f"✓ Dataset loaded from disk: {list(dataset.keys())}\n", flush=True)
            except Exception:
                # try loading single csv file path
                if os.path.exists(data_path) and data_path.endswith('.csv'):
                    from datasets import load_dataset as load_ds
                    dataset = load_ds('csv', data_files={'train': data_path})
                    print(f"✓ Single CSV loaded as train from {data_path}", flush=True)
                else:
                    raise
        
        # Define preprocessing function
        def preprocess_function(examples):
            # convert RNA to DNA T if present
            seqs = [seq.replace('U', 'T').replace('u', 't') for seq in examples.get('seq', examples.get('sequence', []))]
            return tokenizer(seqs, truncation=True, padding='max_length', max_length=101)
        
        # Handle CV setup
        if args.cv_folds > 1:
            print(f"Setting up {args.cv_folds}-fold cross-validation...", flush=True)
            
            # If no validation split exists (common when CSV provided only train/test),
            # create a small validation split from the train set to enable CV.
            if 'validation' not in dataset:
                print("⚠ No 'validation' split found in dataset — creating 10% validation from 'train'", flush=True)
                try:
                    split = dataset['train'].train_test_split(test_size=0.1, seed=42)
                    # replace train with smaller train and add validation
                    dataset['train'] = split['train']
                    dataset['validation'] = split['test']
                    print(f"✓ Created 'validation' split ({len(dataset['validation'])} samples)", flush=True)
                except Exception as e:
                    print(f"❌ Failed to create validation split from 'train': {e}", flush=True)
                    raise

            # Concatenate train and validation for CV
            combined = concatenate_datasets([dataset['train'], dataset['validation']])
            print(f"✓ Combined train+val dataset: {len(combined)} samples", flush=True)
            
            # Subsample combined if requested
            if args.subsample_pos is not None and args.subsample_neg is not None:
                print(f"Subsampling combined dataset to {args.subsample_pos} pos and {args.subsample_neg} neg...", flush=True)
                pos_ds = combined.filter(lambda x: x['label'] == 1)
                neg_ds = combined.filter(lambda x: x['label'] == 0)
                pos_count = min(args.subsample_pos, len(pos_ds))
                neg_count = min(args.subsample_neg, len(neg_ds))
                pos_sample = pos_ds.shuffle(seed=42).select(range(pos_count))
                neg_sample = neg_ds.shuffle(seed=42).select(range(neg_count))
                combined = concatenate_datasets([pos_sample, neg_sample]).shuffle(seed=42)
                print(f"✓ Combined dataset subsampled to {len(combined)} samples", flush=True)
            
            # Tokenize combined
            print("Tokenizing combined dataset...", flush=True)
            # Determine which raw sequence column to remove after tokenization
            seq_column_candidates = ['seq', 'sequence']
            remove_cols = [c for c in seq_column_candidates if c in combined.column_names]
            encoded_combined = combined.map(preprocess_function, batched=True, remove_columns=remove_cols)
            print(f"✓ Combined dataset tokenized", flush=True)
            
            # Create folds
            kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
            indices = list(range(len(encoded_combined)))
            folds = list(kf.split(indices))
            print(f"✓ Created {args.cv_folds} folds", flush=True)
            
            # Tokenize test
            print("Tokenizing test dataset...", flush=True)
            seq_column_candidates = ['seq', 'sequence']
            remove_cols_test = [c for c in seq_column_candidates if c in dataset['test'].column_names]
            encoded_test = dataset['test'].map(preprocess_function, batched=True, remove_columns=remove_cols_test)
            print(f"✓ Test dataset tokenized", flush=True)
            
            cv_mode = True
        else:
            combined = None
            encoded_combined = None
            folds = None
            cv_mode = False
            
            # Subsample if requested (original logic for non-CV)
            if args.subsample_pos is not None and args.subsample_neg is not None:
                print(f"Subsampling to {args.subsample_pos} pos and {args.subsample_neg} neg per split...", flush=True)
                try:
                    subsampled_data = {}
                    for split in ['train', 'test']:
                        ds = dataset[split]
                        print(f"  Processing {split} split ({len(ds)} samples)...", flush=True)
                        
                        pos_ds = ds.filter(lambda x: x['label'] == 1)
                        neg_ds = ds.filter(lambda x: x['label'] == 0)
                        
                        print(f"    Found {len(pos_ds)} positives, {len(neg_ds)} negatives", flush=True)
                        
                        pos_count = min(args.subsample_pos, len(pos_ds))
                        neg_count = min(args.subsample_neg, len(neg_ds))
                        
                        pos_sample = pos_ds.shuffle(seed=42).select(range(pos_count))
                        neg_sample = neg_ds.shuffle(seed=42).select(range(neg_count))
                        
                        subsampled_data[split] = concatenate_datasets([pos_sample, neg_sample]).shuffle(seed=42)
                        print(f"    ✓ {split}: {len(subsampled_data[split])} samples after subsampling", flush=True)
                    
                    # Keep validation as is
                    subsampled_data['validation'] = dataset['validation']
                    
                    # Replace dataset with subsampled version
                    from datasets import DatasetDict
                    dataset = DatasetDict(subsampled_data)
                    print("✓ Subsampling complete\n", flush=True)
                except Exception as e:
                    print(f"❌ ERROR during subsampling: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    raise
            
            # Tokenize
            print("Tokenizing sequences...", flush=True)
            try:
                # Determine sequence column name(s) present in the dataset
                seq_column_candidates = ['seq', 'sequence']
                # If dataset is a DatasetDict, inspect one split to find columns
                sample_split = None
                if isinstance(dataset, dict) or hasattr(dataset, 'keys'):
                    # prefer train split if present
                    if 'train' in dataset:
                        sample_split = dataset['train']
                    else:
                        # pick first available split
                        sample_split = dataset[next(iter(dataset.keys()))]
                else:
                    sample_split = dataset

                remove_cols = [c for c in seq_column_candidates if c in sample_split.column_names]
                encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=remove_cols)
                print(f"✓ Tokenization complete\n", flush=True)
            except Exception as e:
                print(f"❌ ERROR during tokenization: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise
            
            encoded_test = encoded_dataset['test']
        
        # Create config
        num_labels = args.nlabels if hasattr(args, 'nlabels') else 2
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
            num_labels=num_labels
        )
        
        # Training logic
        # Build training arguments using supplied or default hyperparameters
        # Resolve eval/save/logging steps and ensure compatibility with load_best_model_at_end
        eval_steps = args.logging_steps if hasattr(args, 'logging_steps') else 100
        # Historically this script exposed `--save_epochs` but used it as steps; keep backward compatibility
        save_steps = args.save_epochs if hasattr(args, 'save_epochs') else 10
        logging_steps = eval_steps

        # If early stopping / load_best_model_at_end is enabled, transformers requires
        # save_steps to be a round multiple of eval_steps. Adjust save_steps if necessary.
        load_best = (args.early_stopping_patience is not None)
        if load_best and eval_steps and save_steps:
            if save_steps % eval_steps != 0:
                # Choose the safer adjustment: set save_steps to eval_steps (one evaluation between saves)
                print(f"⚠ Adjusting save_steps from {save_steps} to {eval_steps} to be a multiple of eval_steps", flush=True)
                save_steps = eval_steps

        train_args = TrainingArguments(
            disable_tqdm=False,
            save_total_limit=1,
            dataloader_drop_last=True,
            per_device_train_batch_size=args.batch_size if hasattr(args, 'batch_size') else 4,
            per_device_eval_batch_size=1,
            learning_rate=args.lr if hasattr(args, 'lr') else 5e-5,
            weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0.01,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-8,
            warmup_ratio=args.warmup_ratio if hasattr(args, 'warmup_ratio') else 0.05,
            num_train_epochs=args.epochs if hasattr(args, 'epochs') else 10,
            max_grad_norm=args.grad_clipping_norm if hasattr(args, 'grad_clipping_norm') else 1.0,
            gradient_accumulation_steps=args.accum_steps if hasattr(args, 'accum_steps') else 1,
            output_dir=args.output_dir,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy='steps',
            save_steps=save_steps,
            logging_strategy='steps',
            logging_steps=logging_steps,
            fp16=args.fp16 if hasattr(args, 'fp16') else False,
            # If early stopping is requested, ensure we load best model at end
            load_best_model_at_end=load_best,
            metric_for_best_model='auc',
            greater_is_better=True,
            report_to="none"
        )

        if cv_mode:
            # Cross-validation training
            all_results = []
            for fold_idx, (train_indices, val_indices) in enumerate(folds):
                print(f"\n{'='*60}", flush=True)
                print(f"FOLD {fold_idx + 1}/{args.cv_folds}", flush=True)
                print(f"{'='*60}\n", flush=True)
                
                if encoded_combined is None:
                    raise ValueError("encoded_combined is None")
                
                # Initialize model
                model = EsmForSequenceClassification(config)
                model.apply(init_weights)
                
                # Load pretrained encoder if provided
                if args.pretrain_path and os.path.exists(args.pretrain_path):
                    model = load_encoder_weights(model, args.pretrain_path)
                else:
                    print("No pretrained weights - using random initialization for entire model\n", flush=True)
                
                # Freeze encoder if requested
                if args.freeze_encoder:
                    freeze_encoder(model, freeze=True)
                
                # Training arguments
                fold_output_dir = f"{args.output_dir}/fold_{fold_idx+1}"
                os.makedirs(fold_output_dir, exist_ok=True)
                
                training_args = train_args
                
                # Create callbacks
                callbacks = []
                if args.early_stopping_patience is not None:
                    callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=0.001))
                
                # Datasets for this fold
                train_dataset = encoded_combined.select(train_indices)
                eval_dataset = encoded_combined.select(val_indices)
                
                # Create trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    compute_metrics=compute_metrics,
                    callbacks=callbacks,
                )
                
                # Warmup training
                if args.warmup_epochs > 0 and not args.freeze_encoder:
                    freeze_encoder(model, freeze=True)
                    
                    warmup_args = TrainingArguments(
                        output_dir=f"{fold_output_dir}/warmup",
                        evaluation_strategy="epoch",
                        save_strategy="epoch",
                        learning_rate=args.lr,
                        per_device_train_batch_size=args.batch_size,
                        per_device_eval_batch_size=2,
                        num_train_epochs=args.warmup_epochs,
                        weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0.01,
                        load_best_model_at_end=False,
                        save_total_limit=1,
                        logging_dir=f"{fold_output_dir}/warmup/logs",
                        logging_steps=50,
                        dataloader_num_workers=4,
                        report_to="none",
                        fp16=False,
                    )
                    
                    warmup_trainer = Trainer(
                        model=model,
                        args=warmup_args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        compute_metrics=compute_metrics,
                    )
                    warmup_trainer.train()
                    
                    freeze_encoder(model, freeze=False)
                
                # Main training
                trainer.train()
                
                # Evaluate on test
                eval_results = trainer.evaluate(encoded_test)
                all_results.append(eval_results)
                
                # Save model
                trainer.save_model(fold_output_dir)
            
            # Average results
            avg_results = {}
            for key in all_results[0].keys():
                if isinstance(all_results[0][key], (int, float)):
                    values = [r[key] for r in all_results]
                    avg_results[key] = np.mean(values)
            
            print(f"\n{'='*60}", flush=True)
            print("CROSS-VALIDATION RESULTS:", flush=True)
            for k, v in avg_results.items():
                print(f"  {k}: {v:.4f}", flush=True)
            print(f"{'='*60}\n", flush=True)
            # Save CV results to results.json in the output directory
            try:
                os.makedirs(args.output_dir, exist_ok=True)
                results_path = os.path.join(args.output_dir, 'results.json')
                # Convert fold results and averages to JSON-serializable types
                serializable_all = []
                for r in all_results:
                    serializable_all.append({k: (float(v) if (isinstance(v, (int, float)) or hasattr(v, 'item')) else v) for k, v in r.items()})
                serializable_avg = {k: (float(v) if (isinstance(v, (int, float)) or hasattr(v, 'item')) else v) for k, v in avg_results.items()}
                out = {'cv_fold_results': serializable_all, 'cv_avg': serializable_avg}
                with open(results_path, 'w') as _f:
                    json.dump(out, _f, indent=2)
                print(f"Saved CV results to {results_path}", flush=True)
            except Exception as _e:
                print(f"Failed to save CV results.json: {_e}", flush=True)
        else:
            # Original single training
            # Initialize model
            print("Initializing model...", flush=True)
            try:
                model = EsmForSequenceClassification(config)
                model.apply(init_weights)
                print("✓ Model initialized with random weights\n", flush=True)
            except Exception as e:
                print(f"❌ ERROR during model initialization: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise
            
            # Load pretrained encoder if provided
            if args.pretrain_path and os.path.exists(args.pretrain_path):
                try:
                    model = load_encoder_weights(model, args.pretrain_path)
                except Exception as e:
                    print(f"❌ ERROR loading pretrained weights: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    raise
            else:
                print("No pretrained weights - using random initialization for entire model\n", flush=True)
            
            # Freeze encoder if requested
            if args.freeze_encoder:
                print("Freezing encoder layers...", flush=True)
                freeze_encoder(model, freeze=True)
                print()
            # Training arguments
            training_args = train_args
            
            # Create callbacks
            callbacks = []
            if args.early_stopping_patience is not None:
                print(f"Early stopping enabled with patience={args.early_stopping_patience}\n", flush=True)
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=0.001))
            
            # Create trainer
            try:
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=encoded_dataset["train"],
                    eval_dataset=encoded_dataset["validation"],
                    compute_metrics=compute_metrics,
                    callbacks=callbacks,
                )
            except Exception as e:
                print(f"❌ ERROR creating trainer: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise
            
            # Warmup training
            if args.warmup_epochs > 0 and not args.freeze_encoder:
                print(f"\n{'='*60}", flush=True)
                print(f"WARMUP PHASE: Training classifier only for {args.warmup_epochs} epochs", flush=True)
                print(f"{'='*60}\n", flush=True)
                
                freeze_encoder(model, freeze=True)
                
                warmup_args = TrainingArguments(
                    output_dir=f"{args.output_dir}/warmup",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    learning_rate=args.lr,
                    per_device_train_batch_size=args.batch_size,
                    per_device_eval_batch_size=2,
                    num_train_epochs=args.warmup_epochs,
                    weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0.01,
                    load_best_model_at_end=False,
                    save_total_limit=1,
                    logging_dir=f"{args.output_dir}/warmup/logs",
                    logging_steps=50,
                    dataloader_num_workers=4,
                    report_to="none",
                    fp16=False,
                )
                
                try:
                    warmup_trainer = Trainer(
                        model=model,
                        args=warmup_args,
                        train_dataset=encoded_dataset["train"],
                        eval_dataset=encoded_dataset["validation"],
                        compute_metrics=compute_metrics,
                    )
                    warmup_trainer.train()
                except Exception as e:
                    print(f"❌ ERROR during warmup training: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    raise
                
                print(f"\n{'='*60}", flush=True)
                print(f"MAIN PHASE: Unfreezing encoder for full fine-tuning", flush=True)
                print(f"{'='*60}\n", flush=True)
                
                freeze_encoder(model, freeze=False)
                
                # Recreate main trainer with early stopping for main phase
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=encoded_dataset["train"],
                    eval_dataset=encoded_dataset["validation"],
                    compute_metrics=compute_metrics,
                    callbacks=callbacks,
                )
            
            # Main training
            print("Starting training...", flush=True)
            try:
                trainer.train()
            except Exception as e:
                print(f"❌ ERROR during training: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise
            
            # Final evaluation
            print("\nFinal evaluation...", flush=True)
            try:
                eval_results = trainer.evaluate(encoded_dataset["test"])
                print(f"\n{'='*60}", flush=True)
                print("Results:", flush=True)
                for k, v in eval_results.items():
                    print(f"  {k}: {v:.4f}", flush=True)
                print(f"{'='*60}\n", flush=True)
                # Save final evaluation results to results.json in the output directory
                try:
                    os.makedirs(args.output_dir, exist_ok=True)
                    results_path = os.path.join(args.output_dir, 'results.json')
                    serializable_eval = {k: (float(v) if (isinstance(v, (int, float)) or hasattr(v, 'item')) else v) for k, v in eval_results.items()}
                    with open(results_path, 'w') as _f:
                        json.dump(serializable_eval, _f, indent=2)
                    print(f"Saved evaluation results to {results_path}", flush=True)
                except Exception as _e:
                    print(f"Failed to save results.json: {_e}", flush=True)
            except Exception as e:
                print(f"❌ ERROR during evaluation: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise
            
            # Save model
            try:
                trainer.save_model(args.output_dir)
                print(f"✓ Model saved to {args.output_dir}\n", flush=True)
            except Exception as e:
                print(f"❌ ERROR saving model: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise
    except Exception as e:
        print(f"\n\n❌ MAIN ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user", flush=True)
        import sys
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
