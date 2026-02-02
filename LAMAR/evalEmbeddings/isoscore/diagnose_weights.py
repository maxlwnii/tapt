"""
Diagnostic script to verify LAMAR weight loading.
Checks if pretrained weights are actually different from random initialization.
"""

import torch
import numpy as np
from safetensors.torch import load_file
from transformers import EsmConfig, EsmForSequenceClassification, AutoTokenizer

# Paths
TOKENIZER_PATH = "/home/fr/fr_fr/fr_ml642/Thesis/pretrain/saving_model/tapt_lamar/checkpoint-100000/"
PRETRAINED_PATH = "/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/weights"
TAPT_PATH = "/home/fr/fr_fr/fr_ml642/Thesis/pretrain/saving_model/tapt_lamar/checkpoint-100000/model.safetensors"

def init_weights(module):
    """Custom random initialization."""
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


def create_model():
    """Create a fresh model."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
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
        num_labels=2
    )
    return EsmForSequenceClassification(config), tokenizer


def get_embedding_stats(model):
    """Get statistics of the embedding layer."""
    emb = model.esm.embeddings.word_embeddings.weight.detach().cpu().numpy()
    return {
        'mean': float(np.mean(emb)),
        'std': float(np.std(emb)),
        'min': float(np.min(emb)),
        'max': float(np.max(emb)),
        'norm': float(np.linalg.norm(emb))
    }


def get_layer0_stats(model):
    """Get statistics of the first attention layer."""
    q = model.esm.encoder.layer[0].attention.self.query.weight.detach().cpu().numpy()
    return {
        'mean': float(np.mean(q)),
        'std': float(np.std(q)),
        'min': float(np.min(q)),
        'max': float(np.max(q)),
        'norm': float(np.linalg.norm(q))
    }


def load_weights(model, weights_path):
    """Load weights using the same logic as finetune_rbp.py."""
    state_dict = load_file(weights_path)
    
    encoder_weights = {}
    for k, v in state_dict.items():
        if 'lm_head' in k or 'classifier' in k:
            continue
        if k.startswith("esm."):
            encoder_weights[k] = v
        else:
            encoder_weights["esm." + k] = v
    
    missing, unexpected = model.load_state_dict(encoder_weights, strict=False)
    return len(encoder_weights), len(missing), len(unexpected)


def main():
    print("="*70)
    print("LAMAR WEIGHT LOADING DIAGNOSTIC")
    print("="*70)
    
    # 1. Create random model
    print("\n1. Creating random initialized model...")
    torch.manual_seed(42)
    random_model, tokenizer = create_model()
    random_model.apply(init_weights)
    
    random_emb_stats = get_embedding_stats(random_model)
    random_layer0_stats = get_layer0_stats(random_model)
    
    print(f"   Embedding stats: mean={random_emb_stats['mean']:.6f}, std={random_emb_stats['std']:.6f}")
    print(f"   Layer0 Q stats:  mean={random_layer0_stats['mean']:.6f}, std={random_layer0_stats['std']:.6f}")
    
    # 2. Load pretrained weights
    print("\n2. Loading PRETRAINED weights...")
    pretrained_model, _ = create_model()
    pretrained_model.apply(init_weights)  # Init random first (like in finetune_rbp.py)
    n_loaded, n_missing, n_unexpected = load_weights(pretrained_model, PRETRAINED_PATH)
    
    pretrained_emb_stats = get_embedding_stats(pretrained_model)
    pretrained_layer0_stats = get_layer0_stats(pretrained_model)
    
    print(f"   Loaded {n_loaded} tensors, {n_missing} missing, {n_unexpected} unexpected")
    print(f"   Embedding stats: mean={pretrained_emb_stats['mean']:.6f}, std={pretrained_emb_stats['std']:.6f}")
    print(f"   Layer0 Q stats:  mean={pretrained_layer0_stats['mean']:.6f}, std={pretrained_layer0_stats['std']:.6f}")
    
    # 3. Load TAPT weights
    print("\n3. Loading TAPT weights...")
    tapt_model, _ = create_model()
    tapt_model.apply(init_weights)
    n_loaded, n_missing, n_unexpected = load_weights(tapt_model, TAPT_PATH)
    
    tapt_emb_stats = get_embedding_stats(tapt_model)
    tapt_layer0_stats = get_layer0_stats(tapt_model)
    
    print(f"   Loaded {n_loaded} tensors, {n_missing} missing, {n_unexpected} unexpected")
    print(f"   Embedding stats: mean={tapt_emb_stats['mean']:.6f}, std={tapt_emb_stats['std']:.6f}")
    print(f"   Layer0 Q stats:  mean={tapt_layer0_stats['mean']:.6f}, std={tapt_layer0_stats['std']:.6f}")
    
    # 4. Compare embeddings directly
    print("\n" + "="*70)
    print("EMBEDDING COMPARISON (first 5 tokens, first 10 dims)")
    print("="*70)
    
    random_emb = random_model.esm.embeddings.word_embeddings.weight.detach().cpu().numpy()
    pretrained_emb = pretrained_model.esm.embeddings.word_embeddings.weight.detach().cpu().numpy()
    tapt_emb = tapt_model.esm.embeddings.word_embeddings.weight.detach().cpu().numpy()
    
    print("\nRandom embedding [token 0]:")
    print(f"  {random_emb[0, :10]}")
    
    print("\nPretrained embedding [token 0]:")
    print(f"  {pretrained_emb[0, :10]}")
    
    print("\nTAPT embedding [token 0]:")
    print(f"  {tapt_emb[0, :10]}")
    
    # 5. Check if embeddings are identical
    print("\n" + "="*70)
    print("IDENTITY CHECK")
    print("="*70)
    
    random_vs_pretrained = np.allclose(random_emb, pretrained_emb, atol=1e-5)
    random_vs_tapt = np.allclose(random_emb, tapt_emb, atol=1e-5)
    pretrained_vs_tapt = np.allclose(pretrained_emb, tapt_emb, atol=1e-5)
    
    print(f"\nRandom == Pretrained? {random_vs_pretrained}")
    print(f"Random == TAPT?       {random_vs_tapt}")
    print(f"Pretrained == TAPT?   {pretrained_vs_tapt}")
    
    if random_vs_pretrained:
        print("\n⚠️  WARNING: Random and Pretrained embeddings are IDENTICAL!")
        print("    This means pretrained weights are NOT being loaded correctly!")
    else:
        print("\n✓ Pretrained weights are different from random - loading works!")
    
    if random_vs_tapt:  
        print("\n⚠️  WARNING: Random and TAPT embeddings are IDENTICAL!")
    else:
        print("✓ TAPT weights are different from random - loading works!")
    
    # 6. Check L2 distance between embeddings
    print("\n" + "="*70)
    print("L2 DISTANCES")
    print("="*70)
    
    dist_random_pretrained = np.linalg.norm(random_emb - pretrained_emb)
    dist_random_tapt = np.linalg.norm(random_emb - tapt_emb)
    dist_pretrained_tapt = np.linalg.norm(pretrained_emb - tapt_emb)
    
    print(f"\n||Random - Pretrained|| = {dist_random_pretrained:.4f}")
    print(f"||Random - TAPT||       = {dist_random_tapt:.4f}")
    print(f"||Pretrained - TAPT||   = {dist_pretrained_tapt:.4f}")
    
    # 7. Check raw safetensors content
    print("\n" + "="*70)
    print("RAW SAFETENSORS INSPECTION")
    print("="*70)
    
    pretrained_state = load_file(PRETRAINED_PATH)
    tapt_state = load_file(TAPT_PATH)
    
    print(f"\nPretrained file: {PRETRAINED_PATH}")
    print(f"  Keys: {len(pretrained_state)}")
    emb_key = [k for k in pretrained_state.keys() if 'word_embeddings' in k or 'embed_tokens' in k]
    if emb_key:
        print(f"  Embedding key: {emb_key[0]}")
        emb_data = pretrained_state[emb_key[0]].numpy()
        print(f"  Embedding shape: {emb_data.shape}")
        print(f"  Sample values: {emb_data[0, :5]}")
    
    print(f"\nTAPT file: {TAPT_PATH}")
    print(f"  Keys: {len(tapt_state)}")
    emb_key = [k for k in tapt_state.keys() if 'word_embeddings' in k or 'embed_tokens' in k]
    if emb_key:
        print(f"  Embedding key: {emb_key[0]}")
        emb_data = tapt_state[emb_key[0]].numpy()
        print(f"  Embedding shape: {emb_data.shape}")
        print(f"  Sample values: {emb_data[0, :5]}")


if __name__ == "__main__":
    main()
