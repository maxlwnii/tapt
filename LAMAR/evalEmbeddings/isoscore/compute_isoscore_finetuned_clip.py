#!/usr/bin/env python3
"""
Compute IsoScore for finetuned LAMAR models on CLIP training data.
"""

import numpy as np
import os
import torch
from tqdm import tqdm
from transformers import EsmTokenizer, EsmForSequenceClassification
from Bio import SeqIO
import pandas as pd

# Configuration
model_max_length = 512
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
tokenizer_path = "/home/fr/fr_fr/fr_ml642/Thesis/pretrain/saving_model/tapt_lamar/checkpoint-100000/"
clip_data_dir = "/home/fr/fr_fr/fr_ml642/Thesis/data/clip_training_data"
models_base_dir = "/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/models/finetuned_limited"
output_dir = "/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/results_finetuned_isoscore"
os.makedirs(output_dir, exist_ok=True)

# RBPs
rbps = [
    "GTF2F1_K562_IDR", "HNRNPL_K562_IDR", "HNRNPM_HepG2_IDR", "ILF3_HepG2_IDR",
    "KHSRP_K562_IDR", "MATR3_K562_IDR", "PTBP1_HepG2_IDR", "QKI_K562_IDR"
]

# Model types
model_types = ["Pretrained", "Random", "TAPT"]

# Subsample sizes
n_pos = 100
n_neg = 100

def load_tokenizer():
    tokenizer = EsmTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def load_model(rbp, model_type):
    model_path = os.path.join(models_base_dir, model_type, rbp)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    model = EsmForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model

def load_sequences(rbp, label, n_samples):
    fa_file = os.path.join(clip_data_dir, f"{rbp}.{label}s.fa")
    sequences = []
    for record in SeqIO.parse(fa_file, "fasta"):
        seq = str(record.seq).upper().replace("U", "T")  # Convert to DNA
        sequences.append(seq)
    # Subsample
    if len(sequences) > n_samples:
        indices = np.random.choice(len(sequences), n_samples, replace=False)
        sequences = [sequences[i] for i in indices]
    return sequences

def get_embeddings(model, tokenizer, sequences):
    embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
        batch_seqs = sequences[i:i+batch_size]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=model_max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.esm(**inputs)  # Get embeddings from ESM backbone
            # Use mean pooling over sequence length
            mask = inputs['attention_mask'].unsqueeze(-1)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

def compute_isoscore(embeddings):
    """
    Compute IsoScore and additional metrics for a set of embeddings.
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
    
    # Partition function (another isotropy measure)
    partition_function = np.exp(-np.sum(S_normalized * np.log(S_normalized + 1e-10)))
    
    # Average cosine similarity between random pairs
    n_samples = min(1000, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    sample_emb = embeddings[indices]
    sample_emb_norm = sample_emb / (np.linalg.norm(sample_emb, axis=1, keepdims=True) + 1e-10)
    cosine_sim = np.dot(sample_emb_norm, sample_emb_norm.T)
    mask = np.triu(np.ones_like(cosine_sim, dtype=bool), k=1)
    avg_cosine_sim = cosine_sim[mask].mean()
    
    top5_sv_ratio = S[:5].sum() / S.sum() if len(S) >= 5 else 1.0
    top10_sv_ratio = S[:10].sum() / S.sum() if len(S) >= 10 else 1.0
    
    return {
        'IsoScore': isoscore,
        'K_90': k_90,
        'TotalDims': len(S),
        'PartitionFunction': partition_function,
        'AvgCosineSim': avg_cosine_sim,
        'Top5_SV_Ratio': top5_sv_ratio,
        'Top10_SV_Ratio': top10_sv_ratio
    }

def main():
    tokenizer = load_tokenizer()
    results = []

    for rbp in rbps:
        print(f"\nProcessing RBP: {rbp}")
        pos_seqs = load_sequences(rbp, "positive", n_pos)
        neg_seqs = load_sequences(rbp, "negative", n_neg)
        all_seqs = pos_seqs + neg_seqs
        labels = [1] * len(pos_seqs) + [0] * len(neg_seqs)

        for model_type in model_types:
            print(f"  Model: {model_type}")
            model = load_model(rbp, model_type)
            if model is None:
                continue
            embeddings = get_embeddings(model, tokenizer, all_seqs)
            metrics = compute_isoscore(embeddings)
            result = {
                "RBP": rbp,
                "Model": model_type,
                "Num_Pos": len(pos_seqs),
                "Num_Neg": len(neg_seqs)
            }
            result.update(metrics)
            results.append(result)
            print(f"    IsoScore: {metrics['IsoScore']:.4f}, K_90: {metrics['K_90']}, AvgCosineSim: {metrics['AvgCosineSim']:.4f}")

    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "finetuned_isoscores.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    main()