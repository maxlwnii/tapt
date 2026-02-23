"""
Compute IsoScore for LAMAR model embeddings on CLIP training data.
IsoScore measures embedding isotropy - how uniformly embeddings fill the space.

Higher IsoScore = more isotropic (embeddings spread uniformly)
Lower IsoScore = more anisotropic (embeddings clustered in narrow cone)
"""

import numpy as np
import glob
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, EsmConfig
from LAMAR.modeling_nucESM2 import EsmForMaskedLM
from safetensors.torch import load_file
from Bio import SeqIO

# Configuration
model_max_length = 512
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
tokenizer_path = "/home/fr/fr_fr/fr_ml642/Thesis/pretrain/saving_model/tapt_lamar/checkpoint-100000/"
config_path = "/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/config/config_150M.json"
clip_data_dir = "/home/fr/fr_fr/fr_ml642/Thesis/data/clip_training_data"
output_dir = "/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/results_clip_isoscore"
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
        'anisotropy': avg_cosine,
        'top5_sv_ratio': S[:5].sum() / S.sum(),
        'top10_sv_ratio': S[:10].sum() / S.sum()
    }


def load_clip_sequences(rbp_name):
    """Load sequences from CLIP training data (positives and negatives)."""
    sequences = []
    
    # Load positive sequences
    pos_file = os.path.join(clip_data_dir, f"{rbp_name}.positives.fa")
    if os.path.exists(pos_file):
        for record in SeqIO.parse(pos_file, "fasta"):
            seq = str(record.seq).upper().replace("U", "T")
            sequences.append(seq)
    
    # Load negative sequences
    neg_file = os.path.join(clip_data_dir, f"{rbp_name}.negatives.fa")
    if os.path.exists(neg_file):
        for record in SeqIO.parse(neg_file, "fasta"):
            seq = str(record.seq).upper().replace("U", "T")
            sequences.append(seq)
    
    return sequences


def extract_embeddings(model, tokenizer, sequences, layer_idx):
    """Extract embeddings from specified layer."""
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Extracting embeddings"):
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
            layer_embeddings = outputs.hidden_states[layer_idx + 1]
            
            # Mean pooling over sequence length (exclude padding)
            attention_mask = tokens['attention_mask'].unsqueeze(-1)
            masked_embeddings = layer_embeddings * attention_mask
            mean_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
            
            all_embeddings.append(mean_embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def main():
    # Get all RBP files
    pos_files = sorted(glob.glob(os.path.join(clip_data_dir, "*.positives.fa")))
    rbp_names = [os.path.basename(f).replace(".positives.fa", "") for f in pos_files]
    
    print(f"Found {len(rbp_names)} RBPs in CLIP training data")
    print(f"RBPs: {', '.join(rbp_names[:3])}..." if len(rbp_names) > 3 else f"RBPs: {', '.join(rbp_names)}")
    
    results = []
    
    for variant_name, weights_path in variants.items():
        print(f"\n{'='*70}")
        print(f"Processing variant: {variant_name}")
        print(f"{'='*70}")
        
        model, tokenizer = load_model(weights_path)
        
        for layer_idx in target_layers:
            print(f"\n--- Layer {layer_idx} ---")
            
            all_embeddings = []
            processed_rbps = []
            
            for rbp_name in rbp_names:
                print(f"Processing {rbp_name}...")
                
                # Load sequences
                sequences = load_clip_sequences(rbp_name)
                
                if len(sequences) == 0:
                    print(f"  No sequences found for {rbp_name}")
                    continue
                
                print(f"  Loaded {len(sequences)} sequences")
                
                # Sample sequences if too many (for speed)
                if len(sequences) > 2000:
                    indices = np.random.choice(len(sequences), 2000, replace=False)
                    sequences = [sequences[i] for i in indices]
                    print(f"  Sampled 2000 sequences")
                
                # Extract embeddings
                embeddings = extract_embeddings(model, tokenizer, sequences, layer_idx)
                all_embeddings.append(embeddings)
                processed_rbps.append(rbp_name)
            
            if not all_embeddings:
                print("No embeddings extracted!")
                continue
            
            # Combine all embeddings
            combined_embeddings = np.vstack(all_embeddings)
            print(f"\nTotal embeddings combined: {combined_embeddings.shape}")
            
            # Compute IsoScore
            scores = compute_isoscore(combined_embeddings)
            
            result = {
                'Variant': variant_name,
                'Layer': layer_idx,
                'Num_RBPs': len(processed_rbps),
                'Total_Sequences': combined_embeddings.shape[0],
                'IsoScore': scores['isoscore'],
                'K_90': scores['k_90'],
                'TotalDims': scores['total_dims'],
                'PartitionFunction': scores['partition_function'],
                'AvgCosineSim': scores['avg_cosine_sim'],
                'Top5_SV_Ratio': scores['top5_sv_ratio'],
                'Top10_SV_Ratio': scores['top10_sv_ratio']
            }
            results.append(result)
            
            print(f"\n  ✓ IsoScore: {scores['isoscore']:.4f} (k={scores['k_90']}/{scores['total_dims']})")
            print(f"  ✓ Avg Cosine Sim: {scores['avg_cosine_sim']:.4f}")
            print(f"  ✓ Partition Function: {scores['partition_function']:.2f}")
            print(f"  ✓ Top-5 SV Ratio: {scores['top5_sv_ratio']:.4f}")
            print(f"  ✓ Top-10 SV Ratio: {scores['top10_sv_ratio']:.4f}")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "isoscore_clip_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n\nResults saved to: {csv_path}")
    
    # Print summary table
    print("\n" + "="*90)
    print("ISOSCORE SUMMARY (CLIP TRAINING DATA)")
    print("="*90)
    print(df.to_string(index=False))
    
    # Save summary text
    summary_path = os.path.join(output_dir, "isoscore_clip_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("ISOSCORE ANALYSIS FOR LAMAR VARIANTS (CLIP TRAINING DATA)\n")
        f.write("="*90 + "\n\n")
        f.write("Higher IsoScore = more isotropic (embeddings spread uniformly)\n")
        f.write("Lower Avg Cosine Sim = more isotropic\n")
        f.write("Lower Top-K SV Ratio = more isotropic (variance spread across more dimensions)\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # Interpretation
        f.write("INTERPRETATION:\n")
        f.write("-"*60 + "\n")
        for layer in target_layers:
            layer_df = df[df['Layer'] == layer]
            f.write(f"\nLayer {layer}:\n")
            for _, row in layer_df.iterrows():
                f.write(f"  {row['Variant']:12s} | IsoScore={row['IsoScore']:.4f} | ")
                f.write(f"AvgCos={row['AvgCosineSim']:.4f} | Seqs={row['Total_Sequences']}\n")
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
