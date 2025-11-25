from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
from LAMAR.modeling_nucESM2 import EsmForMaskedLM
from transformers import AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments
from datasets import load_dataset
import argparse
import torch
import numpy as np
from dataclasses import dataclass

# Conditional import to avoid dependency errors during local testing
try:
    from transformers import Trainer
    TRAINER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Trainer import failed: {e}")
    print("Running in test mode without Trainer")
    TRAINER_AVAILABLE = False
    Trainer = None

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


@dataclass
class DataCollatorForLanguageModelingCustom(DataCollatorForLanguageModeling):
    """
    This datacollator will choose different masking prb for different regions.
    High confidence regions will have higher masking coverage than flanking regions.
    """
    tokenizer: Any
    eclip_mlm: Tuple[float, float] = (0.2, 0.25)
    flanking_mlm: Tuple[float, float] = (0.1, 0.15)

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "Remove the MLM part of the training objective."
            )
        if not self.mlm:
            raise ValueError("This data collator only works for masked language modeling")

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of examples with region-specific masking.
        """
        # Extract eclip_regions before padding (datasets handles this as list of lists)
        eclip_peaks = [ex.pop("eclip_regions", []) for ex in examples]
        
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        batch['input_ids'], batch['labels'] = self.mask_tokens_custom(batch['input_ids'], eclip_peaks)

        return batch
    
    def mask_tokens_custom(self, inputs: Any, eclip_regions: List[List[Dict[str, int]]], special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Perform custom masking based on genomic regions.
        """
        labels = inputs.clone()
        # Initialize with flanking probability
        probability_matrix = torch.full(labels.shape, np.random.uniform(self.flanking_mlm[0], self.flanking_mlm[1]))
        
        # Calculate offset for special tokens (e.g., if [CLS] is at index 0)
        # Assuming standard ESM/BERT tokenizer that adds 1 special token at the start
        offset = 1 if self.tokenizer.cls_token_id is not None else 0

        if eclip_regions:
            for i, regions in enumerate(eclip_regions):
                for region in regions:
                    # FIX 1: Use correct keys 'peak_start'/'peak_end' matching preprocess.py
                    # FIX 2: Add offset to align with tokenized sequence
                    start = int(region.get('peak_start', 0)) + offset
                    end = int(region.get('peak_end', 0)) + offset
                    
                    # Ensure we don't go out of bounds
                    start = max(0, start)
                    end = min(inputs.shape[1], end)
                    
                    if start < end:
                        probability_matrix[i, start:end] = np.random.uniform(self.eclip_mlm[0], self.eclip_mlm[1])

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:   
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Standard masking logic
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # 80% replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% replace with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(low=0, high=len(self.tokenizer), size=labels.shape, dtype=torch.long) # Fixed high=vocab_size
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


def main(
        tokenizer_path, 
        model_max_length, 
        model_name, 
        token_dropout, 
        positional_embedding_type, 
        hidden_size, 
        intermediate_size, 
        num_attention_heads, 
        num_hidden_layers, 
        data_for_pretrain_path,  
        flash_attention, 
        disable_tqdm, 
        batch_size, 
        peak_lr, 
        warmup_ratio, 
        max_steps,  
        grad_clipping_norm, 
        accum_steps, 
        output_dir, 
        save_steps, 
        logging_steps, 
        fp16,
        data_for_validation_path,
        resume_training, 
        data_collator_patch
    ):
    
    if not TRAINER_AVAILABLE:
        print("ERROR: Trainer not available. Install dependencies on cluster.")
        print("Validating configuration only...\n")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length)
    
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer has no pad_token, setting to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Tokenizer loaded: pad_token={tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    
    # Config
    config = AutoConfig.from_pretrained(
        model_name, vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, 
        mask_token_id=tokenizer.mask_token_id, 
        token_dropout=token_dropout, positional_embedding_type=positional_embedding_type, 
        hidden_size=hidden_size, intermediate_size=intermediate_size, 
        num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers
    )
    print(f"✓ Config loaded: {config.num_hidden_layers} layers, {hidden_size} hidden size")
    
    # Training data
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['sequence'], 
            return_special_tokens_mask=True, 
            truncation=True, 
            padding='max_length', 
            max_length=tokenizer.model_max_length
        )
        if 'eclip_regions' in examples:
            tokenized['eclip_regions'] = examples['eclip_regions']
        return tokenized

    print(f"✓ Loading dataset from: {data_for_pretrain_path}")
    train_set = load_dataset("json", data_files=data_for_pretrain_path, streaming=True)
    
    data_for_pretrain = train_set.map(
        tokenize_function, 
        batched=True,
        remove_columns=["sequence", "seq_id", "method", "start", "end", "seq_len"]
    )

    data_for_validation = None
    if data_for_validation_path:
        print(f"✓ Loading validation dataset from: {data_for_validation_path}")
        val_set = load_dataset("json", data_files=data_for_validation_path, streaming=True)
        
        data_for_validation = val_set.map(
            tokenize_function, 
            batched=True,
            remove_columns=["sequence", "seq_id", "method", "start", "end", "seq_len"]
        )
        
        # Test one validation sample
        val_sample = next(iter(data_for_validation['train']))
        print(f"✓ Validation sample keys: {val_sample.keys()}")
        print(f"  - input_ids shape: {len(val_sample['input_ids'])}")
        print(f"  - eclip_regions: {len(val_sample.get('eclip_regions', []))} regions")
        
        # Test one sample
        sample = next(iter(data_for_pretrain['train']))
        print(f"✓ Dataset sample keys: {sample.keys()}")
        print(f"  - input_ids shape: {len(sample['input_ids'])}")
        print(f"  - eclip_regions: {len(sample.get('eclip_regions', []))} regions")

    # Data Collator
    pad_multiple = 8 if flash_attention else None
    
    if data_collator_patch:
        print("✓ Using Custom Adaptive Masking Collator")
        data_collator = DataCollatorForLanguageModelingCustom(
            tokenizer=tokenizer, 
            mlm=True, 
            eclip_mlm=(0.2, 0.25),
            flanking_mlm=(0.1, 0.15),
            pad_to_multiple_of=pad_multiple
        )
    else:
        print("✓ Using Standard Data Collator")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=True, 
            mlm_probability=0.15, 
            pad_to_multiple_of=pad_multiple
        )

    # Model
    model = EsmForMaskedLM(config)
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    if flash_attention:
        try:
            from LAMAR.flash_attn_patch import EsmSelfAttentionAddFlashAttnPatch
            for i in range(config.num_hidden_layers):
                model.esm.encoder.layer[i].attention.self = EsmSelfAttentionAddFlashAttnPatch(config, position_embedding_type='rotary')
            print("✓ Flash Attention enabled")
        except ImportError:
            print("⚠ Flash Attention requested but not available")
    
    if not TRAINER_AVAILABLE:
        print("\n✅ Configuration validated successfully!")
        print("📋 Summary:")
        print(f"  - Model: {config.num_hidden_layers}L, {hidden_size}H, {num_attention_heads}A")
        print(f"  - Batch size: {batch_size} × {accum_steps} accum steps")
        print(f"  - Max steps: {max_steps}")
        print(f"  - Learning rate: {peak_lr}")
        print("\n⚠️  Run this on the cluster with proper environment to start training.")
        return
            
    # Training arguments
    train_args = TrainingArguments(
        disable_tqdm=disable_tqdm, 
        save_total_limit=3,
        dataloader_drop_last=True, 
        per_device_train_batch_size=batch_size, 
        learning_rate=peak_lr, 
        weight_decay=0.01, 
        adam_beta1=0.9, 
        adam_beta2=0.98, 
        adam_epsilon=1e-8, 
        warmup_ratio=warmup_ratio, 
        max_steps=max_steps,
        max_grad_norm=grad_clipping_norm, 
        gradient_accumulation_steps=accum_steps, 
        output_dir=output_dir, 
        save_strategy='steps', 
        save_steps=save_steps, 
        logging_steps=logging_steps, 
        fp16=fp16,
        dispatch_batches=False, 
        ignore_data_skip=True, 
        report_to='tensorboard',
        evaluation_strategy='steps' if data_for_validation_path else 'no',
        eval_steps=save_steps,  # Evaluate at same frequency as saving
        load_best_model_at_end=True if data_for_validation_path else False,
        metric_for_best_model='loss',
        greater_is_better=False,
    )
    
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=data_for_pretrain['train'], 
        eval_dataset=data_for_validation['train'] if data_for_validation else None,
        data_collator=data_collator, 
        tokenizer=tokenizer
    )
    
    # Training
    trainer.train(resume_from_checkpoint=resume_training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretraining LAMAR')
    parser.add_argument('--tokenizer_path', default='tokenizer/single_nucleotide', type=str, help='Directory of tokenizer')
    parser.add_argument('--model_max_length', default=2050, type=int, help='Model input size')
    parser.add_argument('--model_name', default="config/config_150M.json", type=str, help='Name of training model')
    parser.add_argument('--token_dropout', action='store_true', help='Token dropout')
    parser.add_argument('--positional_embedding_type', default="rotary", type=str, help='Positional embedding type rotary or absolute')
    parser.add_argument('--hidden_size', type=int, help='Hidden size of token')
    parser.add_argument('--intermediate_size', type=int, help='Intermediate size in Linear Module')
    parser.add_argument('--num_attention_heads', type=int, help='Number of attention heads')
    parser.add_argument('--num_hidden_layers', type=int, help='Num of hidden layers')
    parser.add_argument('--data_for_pretrain_path', type=str, help='Path of the data for pretrain')
    parser.add_argument('--flash_attention', action='store_true', help='Whether to use flash attention')
    parser.add_argument('--disable_tqdm', action='store_true', help='Whether to disable tqdm')
    parser.add_argument('--batch_size', default=8, type=int, help='Input batch size on each device (default: 8)')
    parser.add_argument('--peak_lr', default=1e-4, type=float, help='Peak learning rate')
    parser.add_argument('--warmup_ratio', default=0.05, type=float, help='Warm up ratio')
    parser.add_argument('--max_steps', default=300000, type=int, help='Max training steps')
    parser.add_argument('--grad_clipping_norm', type=float, help='Max norm of the gradients in gradient clipping')
    parser.add_argument('--accum_steps', default=1, type=int, help='accumulation steps (default: 1)')
    parser.add_argument('--output_dir', type=str, help='Directory of training output')
    parser.add_argument('--save_steps', default=1000, type=int, help='Save steps')
    parser.add_argument('--logging_steps', default=100, type=int, help='when to compute the loss')
    parser.add_argument('--fp16', action='store_true', help='Training with fp16')
    parser.add_argument('--resume_training', action='store_true', help='Whether resume from training')
    parser.add_argument('--data_collator_patch', action='store_true', help='Whether use data collator patch')
    parser.add_argument('--data_for_validation_path', type=str, default=None, help='Path of the data for validation')   
    args = parser.parse_args()

    main(
    args.tokenizer_path, args.model_max_length, args.model_name, args.token_dropout, args.positional_embedding_type, args.hidden_size, 
    args.intermediate_size, args.num_attention_heads, args.num_hidden_layers, 
    args.data_for_pretrain_path, 
    args.flash_attention, args.disable_tqdm,  # Keep these here
    args.batch_size, args.peak_lr, args.warmup_ratio, args.max_steps, args.grad_clipping_norm, args.accum_steps, 
    args.output_dir, args.save_steps, args.logging_steps, args.fp16, 
    args.data_for_validation_path,  # Move this here (before resume_training)
    args.resume_training, args.data_collator_patch
)
