#!/usr/bin/env python3
"""
Compute IsoScore for LAMAR model embeddings on eCLIP data.
IsoScore measures embedding isotropy - how uniformly embeddings fill the space.

Higher IsoScore = more isotropic (embeddings spread uniformly)
Lower IsoScore = more anisotropic (embeddings clustered in narrow cone)

References:
- Ethayarajh (2019): "How Contextual are Contextualized Word Representations?"
- IsoScore paper
"""

import numpy as np
import h5py
import glob
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, EsmConfig
from LAMAR.modeling_nucESM2 import EsmForMaskedLM
from safetensors.torch import load_file

# Configuration
model_max_length = 512
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
# Use tokenizer from TAPT checkpoint (same vocab as Pretrained)
tokenizer_path = "/home/fr/fr_fr/fr_ml642/Thesis/pretrain/saving_model/tapt_lamar/checkpoint-100000/"
config_path = "/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/config/config_150M.json"
eclip_dir = "/home/fr/fr_fr/fr_ml642/Thesis/eclip"
output_dir = "/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/results"
os.makedirs(output_dir, exist_ok=True)

# Model variants
variants = {
    "Pretrained": "/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/weights",
    "TAPT": "/home/fr/fr_fr/fr_ml642/Thesis/pretrain/saving_model/tapt_lamar/checkpoint-100000/model.safetensors",
    "Random": None
}

# Layers to analyze
target_layers = [5, 11]  # Middle and last layer


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


def load_model(weights_path):
    """Load LAMAR model with specified weights or random init."""
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length)
    
    # Load or create config
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        config = AutoConfig.from_pretrained(config_path)
    else:
        print(f"Config not found at {config_path}, creating from scratch")
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
            problem_type="single_label_classification",
            num_labels=2
        )
    
    # Override key parameters to ensure consistency
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id
    config.mask_token_id = tokenizer.mask_token_id
    config.token_dropout = False
    config.positional_embedding_type = "rotary"
    
    model = EsmForMaskedLM(config)
    
    if weights_path:
        print(f"Loading weights from {weights_path}")
        state_dict = load_file(weights_path)
        
        # Map keys
        weight_dict = {}
        for k, v in state_dict.items():
            if k.startswith("esm.lm_head"):
                new_k = k.replace("esm.", '', 1)
            elif k.startswith("lm_head"):
                new_k = k
            elif k.startswith("esm."):
                new_k = k
            else:
                new_k = "esm." + k
            weight_dict[new_k] = v
        
        result = model.load_state_dict(weight_dict, strict=False)
        print(f"Loaded weights: {len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected")
    else:
        print("Using WOLF random initialization")
        model.apply(init_weights)
    
    model.to(device)
    model.eval()
    return model, tokenizer


def compute_isoscore(embeddings):
    """
    Compute IsoScore for a set of embeddings.
    
    IsoScore = 1 - (sum of top-k singular values / sum of all singular values)
    where k is chosen such that top-k captures 90% of variance
    
    More isotropic = higher score (closer to 1)
    More anisotropic = lower score (closer to 0)
    """
    # Center the embeddings
    embeddings = embeddings - embeddings.mean(axis=0)
    
    # Compute singular values
    _, S, _ = np.linalg.svd(embeddings, full_matrices=False)
    
    # Normalize singular values
    S_normalized = S / S.sum()
    
    # Compute cumulative variance
    cumsum = np.cumsum(S_normalized)
    
    # Find k where 90% variance is captured
    k_90 = np.searchsorted(cumsum, 0.9) + 1
    
    # IsoScore: how many dimensions needed to capture 90% variance
    # Normalized by total dimensions
    isoscore = k_90 / len(S)
    
    # Also compute partition function (another isotropy measure)
    # Higher = more isotropic
    partition_function = np.exp(-np.sum(S_normalized * np.log(S_normalized + 1e-10)))
    
    # Compute anisotropy (Ethayarajh 2019)
    # Average cosine similarity between random pairs
    n_samples = min(1000, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    sample_emb = embeddings[indices]
    sample_emb_norm = sample_emb / (np.linalg.norm(sample_emb, axis=1, keepdims=True) + 1e-10)
    cosine_sim = np.dot(sample_emb_norm, sample_emb_norm.T)
    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(cosine_sim, dtype=bool), k=1)
    avg_cosine = cosine_sim[mask].mean()
    
    return {
        'isoscore': isoscore,
        'k_90': k_90,
        'total_dims': len(S),
        'partition_function': partition_function,
        'avg_cosine_sim': avg_cosine,
        'anisotropy': avg_cosine,  # Higher = more anisotropic
        'top5_sv_ratio': S[:5].sum() / S.sum(),
        'top10_sv_ratio': S[:10].sum() / S.sum()
    }


def onehot_to_sequence(onehot):
    """Convert one-hot encoded sequence to string."""
    # Channels 0-3 are A, C, G, T (assumed order)
    nucleotides = ['A', 'C', 'G', 'T']
    seq = []
    for pos in range(onehot.shape[1]):
        idx = np.argmax(onehot[:4, pos])
        seq.append(nucleotides[idx])
    return ''.join(seq)


def extract_embeddings(model, tokenizer, sequences, layer_idx):
    """Extract embeddings from specified layer."""
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Layer {layer_idx}"):
            batch_seqs = sequences[i:i+batch_size]
            
            tokens = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=model_max_length
            ).to(device)
            
            outputs = model.esm(**tokens, output_hidden_states=True)
            
            # Get embeddings from specified layer
            # hidden_states[0] is embedding layer, hidden_states[1] is layer 0, etc.
            layer_embeddings = outputs.hidden_states[layer_idx + 1]
            
            # Mean pooling over sequence length (exclude padding)
            attention_mask = tokens['attention_mask'].unsqueeze(-1)
            masked_embeddings = layer_embeddings * attention_mask
            mean_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
            
            all_embeddings.append(mean_embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def main():
    # Get all h5 files
    h5_files = sorted(glob.glob(os.path.join(eclip_dir, "*.h5")))
    print(f"Found {len(h5_files)} eCLIP files")
    
    # Use first 3 files for speed (or all if you want)
    h5_files = h5_files[:3]  # Limit for faster computation
    print(f"Using {len(h5_files)} files for analysis")
    
    results = []
    
    for variant_name, weights_path in variants.items():
        print(f"\n{'='*60}")
        print(f"Processing variant: {variant_name}")
        print(f"{'='*60}")
        
        model, tokenizer = load_model(weights_path)
        
        for layer_idx in target_layers:
            print(f"\n--- Layer {layer_idx} ---")
            
            all_embeddings = []
            
            for h5_file in h5_files:
                rbp_name = os.path.basename(h5_file).replace('.h5', '')
                print(f"Processing {rbp_name}...")
                
                with h5py.File(h5_file, 'r') as f:
                    # Get sequences from train split
                    X_train = f['X_train'][:]
                    
                    # Convert one-hot to sequences (sample for speed)
                    n_samples = min(500, len(X_train))
                    indices = np.random.choice(len(X_train), n_samples, replace=False)
                    
                    sequences = []
                    for idx in indices:
                        seq = onehot_to_sequence(X_train[idx])
                        sequences.append(seq)
                
                # Extract embeddings
                embeddings = extract_embeddings(model, tokenizer, sequences, layer_idx)
                all_embeddings.append(embeddings)
            
            # Combine all embeddings
            combined_embeddings = np.vstack(all_embeddings)
            print(f"Total embeddings: {combined_embeddings.shape}")
            
            # Compute IsoScore
            scores = compute_isoscore(combined_embeddings)
            
            result = {
                'Variant': variant_name,
                'Layer': layer_idx,
                'IsoScore': scores['isoscore'],
                'K_90': scores['k_90'],
                'TotalDims': scores['total_dims'],
                'PartitionFunction': scores['partition_function'],
                'AvgCosineSim': scores['avg_cosine_sim'],
                'Top5_SV_Ratio': scores['top5_sv_ratio'],
                'Top10_SV_Ratio': scores['top10_sv_ratio']
            }
            results.append(result)
            
            print(f"  IsoScore: {scores['isoscore']:.4f} (k={scores['k_90']}/{scores['total_dims']})")
            print(f"  Avg Cosine Sim: {scores['avg_cosine_sim']:.4f}")
            print(f"  Top-5 SV Ratio: {scores['top5_sv_ratio']:.4f}")
            print(f"  Top-10 SV Ratio: {scores['top10_sv_ratio']:.4f}")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "isoscore_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n\nResults saved to: {csv_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("ISOSCORE SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save summary text
    summary_path = os.path.join(output_dir, "isoscore_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("ISOSCORE ANALYSIS FOR LAMAR VARIANTS\n")
        f.write("="*80 + "\n\n")
        f.write("Higher IsoScore = more isotropic (embeddings spread uniformly)\n")
        f.write("Lower Avg Cosine Sim = more isotropic\n")
        f.write("Lower Top-K SV Ratio = more isotropic (variance spread across more dimensions)\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # Interpretation
        f.write("INTERPRETATION:\n")
        f.write("-"*40 + "\n")
        for layer in target_layers:
            layer_df = df[df['Layer'] == layer]
            f.write(f"\nLayer {layer}:\n")
            for _, row in layer_df.iterrows():
                f.write(f"  {row['Variant']}: IsoScore={row['IsoScore']:.4f}, AvgCos={row['AvgCosineSim']:.4f}\n")
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
