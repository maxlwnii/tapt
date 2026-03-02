from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
from LAMAR.modeling_nucESM2 import EsmForMaskedLM
from safetensors.torch import load_file as load_safetensors
from transformers import AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, EarlyStoppingCallback
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

def tokenize_function(tokenizer):
    def inner(examples):
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
    return inner

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
        Ensures overall masking rate is ~15% by adjusting region probabilities.
        """
        labels = inputs.clone()
        batch_size, seq_len = inputs.shape
        
        # Calculate offset for special tokens
        offset = 1 if self.tokenizer.cls_token_id is not None else 0
        
        # Initialize probability matrix with zeros
        probability_matrix = torch.zeros(labels.shape, dtype=torch.float32)
        
        # Create a mask to track which positions are in eCLIP regions
        eclip_mask = torch.zeros(labels.shape, dtype=torch.bool)
        
        if eclip_regions:
            for i, regions in enumerate(eclip_regions):
                for region in regions:
                    start = int(region.get('peak_start', 0)) + offset
                    end = int(region.get('peak_end', 0)) + offset
                    
                    # Ensure we don't go out of bounds
                    start = max(0, start)
                    end = min(seq_len, end)
                    
                    if start < end:
                        eclip_mask[i, start:end] = True
        
        # Get special tokens mask
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:   
            special_tokens_mask = special_tokens_mask.bool()
        
        # Robustly ensure special_tokens_mask is a boolean tensor on the correct device
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device)
        else:
            # If passed in as list/ndarray/tensor, convert to bool tensor on same device
            if not isinstance(special_tokens_mask, torch.Tensor):
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device)
            else:
                special_tokens_mask = special_tokens_mask.to(labels.device).bool()

        # Target masking rate (exact per-sample count)
        target_mask_rate = 0.15

        # We'll build masked indices deterministically so each sample masks ~15% of valid tokens
        masked_indices = torch.zeros_like(labels, dtype=torch.bool, device=labels.device)

        vocab_size = getattr(self.tokenizer, 'vocab_size', None) or len(self.tokenizer)
        mask_id = getattr(self.tokenizer, 'mask_token_id', self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token))

        for i in range(batch_size):
            valid_tokens = ~special_tokens_mask[i]
            n_valid = int(valid_tokens.sum().item())
            if n_valid == 0:
                continue

            eclip_tokens = (eclip_mask[i] & valid_tokens)
            flanking_tokens = (~eclip_mask[i] & valid_tokens)

            n_eclip = int(eclip_tokens.sum().item())
            n_flanking = int(flanking_tokens.sum().item())

            # Sample preferred region probabilities (used only as relative weights)
            eclip_pref = np.random.uniform(self.eclip_mlm[0], self.eclip_mlm[1])
            flanking_pref = np.random.uniform(self.flanking_mlm[0], self.flanking_mlm[1])

            # Determine exact number of tokens to mask for this sample
            n_to_mask = max(1, int(round(target_mask_rate * n_valid)))

            # If there are no eclip tokens or no flanking tokens, allocate accordingly
            if n_eclip == 0 and n_flanking == 0:
                continue
            if n_eclip == 0:
                n_flanking_mask = min(n_flanking, n_to_mask)
                n_eclip_mask = 0
            elif n_flanking == 0:
                n_eclip_mask = min(n_eclip, n_to_mask)
                n_flanking_mask = 0
            else:
                weight_e = eclip_pref * n_eclip
                weight_f = flanking_pref * n_flanking
                total_weight = weight_e + weight_f
                if total_weight <= 0:
                    # fallback to proportional by token counts
                    n_eclip_mask = int(round(n_to_mask * (n_eclip / n_valid)))
                else:
                    n_eclip_mask = int(round(n_to_mask * (weight_e / total_weight)))
                n_eclip_mask = max(0, min(n_eclip_mask, n_eclip))
                n_flanking_mask = n_to_mask - n_eclip_mask
                # fix bounds
                if n_flanking_mask > n_flanking:
                    n_flanking_mask = n_flanking
                    n_eclip_mask = min(n_to_mask - n_flanking_mask, n_eclip)

            # If rounding produced fewer than desired (due to bounds), fill from remaining valid tokens
            current_total = n_eclip_mask + n_flanking_mask
            if current_total < n_to_mask:
                need = n_to_mask - current_total
                # take from flanking first then eclip
                add_from_flank = min(need, n_flanking - n_flanking_mask)
                n_flanking_mask += add_from_flank
                need -= add_from_flank
                add_from_eclip = min(need, n_eclip - n_eclip_mask)
                n_eclip_mask += add_from_eclip

            # Select random positions within each region
            seq_range = torch.arange(labels.size(1), device=labels.device)
            if n_eclip_mask > 0 and n_eclip > 0:
                e_pos = seq_range[eclip_tokens]
                perm = torch.randperm(e_pos.size(0), device=labels.device)
                chosen_e = e_pos[perm[:n_eclip_mask]]
                masked_indices[i, chosen_e] = True
            if n_flanking_mask > 0 and n_flanking > 0:
                f_pos = seq_range[flanking_tokens]
                perm = torch.randperm(f_pos.size(0), device=labels.device)
                chosen_f = f_pos[perm[:n_flanking_mask]]
                masked_indices[i, chosen_f] = True

        # Ensure we never mask special tokens
        masked_indices.masked_fill_(special_tokens_mask, False)

        labels[~masked_indices] = -100  # only compute loss on masked tokens

        # Now implement the 80/10/10 replacement strategy on the exact masked positions
        # Get per-sample masked positions to choose replacements
        # Replace 80% with [MASK]
        mask_indices = masked_indices.clone()
        for i in range(batch_size):
            pos = torch.nonzero(mask_indices[i], as_tuple=False).squeeze(1)
            n_masked = pos.numel()
            if n_masked == 0:
                continue
            n_replace = int(round(0.8 * n_masked))
            n_random = int(round(0.1 * n_masked))
            # adjust to ensure totals equal n_masked
            if n_replace + n_random > n_masked:
                n_random = max(0, n_masked - n_replace)
            n_keep = n_masked - n_replace - n_random

            perm = torch.randperm(n_masked, device=labels.device)
            replace_idx = pos[perm[:n_replace]]
            random_idx = pos[perm[n_replace:n_replace + n_random]]

            inputs[i, replace_idx] = mask_id

            if n_random > 0:
                rand_words = torch.randint(low=0, high=vocab_size, size=(n_random,), device=labels.device)
                inputs[i, random_idx] = rand_words

            # Remaining n_keep positions are left unchanged
        
        return inputs, labels


def main(
        tokenizer_path, 
        model_max_length, 
        model_name,
        pretrained_model_path, 
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
        data_collator_patch,
        lr_scheduler_type,
        optim,
        gradient_checkpointing,
        dataloader_num_workers,
        dataloader_pin_memory,
        seed
    ):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length)
    
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer has no pad_token, setting to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Tokenizer loaded: pad_token={tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    
    tokenize_fn = tokenize_function(tokenizer)
    
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

    print(f"✓ Loading dataset from: {data_for_pretrain_path}")
    train_set = load_dataset("json", data_files=data_for_pretrain_path, streaming=True)
    
    data_for_pretrain = train_set.map(
        tokenize_fn, 
        batched=True,
        remove_columns=["sequence", "seq_id", "method", "start", "end", "seq_len"]
    )

    data_for_validation = None
    if data_for_validation_path:
        print(f"✓ Loading validation dataset from: {data_for_validation_path}")
        val_set = load_dataset("json", data_files=data_for_validation_path, streaming=True)
        
        data_for_validation = val_set.map(
            tokenize_fn, 
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
    
    if pretrained_model_path:
        import os
        # Resolve weight file: if path is a directory, look for model.safetensors inside
        if os.path.isdir(pretrained_model_path):
            wf = os.path.join(pretrained_model_path, "model.safetensors")
        else:
            wf = pretrained_model_path  # assume it's the file itself
        
        state_dict = load_safetensors(wf)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        print(f"✓ Pretrained weights loaded from: {wf}")
        if missing:
            print(f"  ⚠ Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"  ⚠ Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    else:
        print(f"⚠ No pretrained weights — training from random init — {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
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
            
    # Enable gradient checkpointing if requested
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
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
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        seed=seed,
    )
    
    # Add early stopping callback if validation data is provided
    callbacks = []
    if data_for_validation_path:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=20))
        print("✓ Early stopping enabled with patience=20")
    
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=data_for_pretrain['train'], 
        eval_dataset=data_for_validation['train'] if data_for_validation else None,
        data_collator=data_collator, 
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    
    # Training
    trainer.train(resume_from_checkpoint=resume_training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretraining LAMAR')
    parser.add_argument('--tokenizer_path', default='tokenizer/single_nucleotide', type=str, help='Directory of tokenizer')
    parser.add_argument('--model_max_length', default=2050, type=int, help='Model input size')
    parser.add_argument('--model_name', default="config/config_150M.json", type=str, help='Name of training model')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to pretrained model weights (safetensors file or directory) for continued pretraining (TAPT). If omitted, trains from scratch.')
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
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='Learning rate scheduler type')
    parser.add_argument('--optim', type=str, default='adamw_torch', help='Optimizer type')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--dataloader_num_workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--dataloader_pin_memory', action='store_true', help='Pin memory for dataloader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')   
    args = parser.parse_args()

    main(
    args.tokenizer_path, args.model_max_length, args.model_name, args.pretrained_model_path, args.token_dropout, args.positional_embedding_type, args.hidden_size, 
    args.intermediate_size, args.num_attention_heads, args.num_hidden_layers, 
    args.data_for_pretrain_path, 
    args.flash_attention, args.disable_tqdm,
    args.batch_size, args.peak_lr, args.warmup_ratio, args.max_steps, args.grad_clipping_norm, args.accum_steps, 
    args.output_dir, args.save_steps, args.logging_steps, args.fp16, 
    args.data_for_validation_path,
    args.resume_training, args.data_collator_patch,
    args.lr_scheduler_type, args.optim, args.gradient_checkpointing,
    args.dataloader_num_workers, args.dataloader_pin_memory, args.seed
)
