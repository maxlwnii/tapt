#!/usr/bin/env python3
"""
Analyze GC content in eCLIP h5 files.
Compare GC content between positive and negative samples.
"""

import h5py
import numpy as np
import pandas as pd
import glob
import os

# Output directory
output_dir = "/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/results"
os.makedirs(output_dir, exist_ok=True)

# Get all h5 files
h5_files = glob.glob('/home/fr/fr_fr/fr_ml642/Thesis/eclip/*.h5')
print(f"Found {len(h5_files)} h5 files\n")

# First, let's understand the one-hot encoding
# Assuming first 4 channels are A, C, G, T (or similar)
# We need to figure out which channels correspond to which nucleotide

print("=" * 80)
print("GC CONTENT ANALYSIS - POSITIVES vs NEGATIVES")
print("=" * 80)

all_results = []

for h5_file in sorted(h5_files):
    rbp_name = os.path.basename(h5_file).replace('.h5', '')
    
    with h5py.File(h5_file, 'r') as f:
        # Combine all splits
        X_all = np.concatenate([f['X_train'][:], f['X_valid'][:], f['X_test'][:]], axis=0)
        Y_all = np.concatenate([f['Y_train'][:], f['Y_valid'][:], f['Y_test'][:]], axis=0)
        
        # Separate positives and negatives
        pos_mask = Y_all.flatten() == 1
        neg_mask = Y_all.flatten() == 0
        
        X_pos = X_all[pos_mask]
        X_neg = X_all[neg_mask]
        
        # The data shape is (samples, 9, 200)
        # First 4 channels appear to be one-hot encoded nucleotides
        # Looking at the data, rows 0-3 seem to be A, C, G, T (binary values)
        
        # Extract one-hot encoded nucleotides (first 4 channels)
        # Channel indices: 0=A, 1=C, 2=G, 3=T (assumption - needs verification)
        # Or could be: 0=A, 1=T, 2=C, 3=G
        
        # Let's compute sum of each channel to understand encoding
        channel_sums_pos = X_pos[:, :4, :].sum(axis=(0, 2)) / (X_pos.shape[0] * 200)
        channel_sums_neg = X_neg[:, :4, :].sum(axis=(0, 2)) / (X_neg.shape[0] * 200)
        
        # For GC content, we need to identify which channels are G and C
        # Typically in one-hot: A=0, C=1, G=2, T=3 OR A=0, T=1, C=2, G=3
        # Let's try both interpretations
        
        # Interpretation 1: A=0, C=1, G=2, T=3
        gc_pos_v1 = (X_pos[:, 1, :] + X_pos[:, 2, :]).mean()  # C + G
        gc_neg_v1 = (X_neg[:, 1, :] + X_neg[:, 2, :]).mean()
        
        # Interpretation 2: A=0, T=1, C=2, G=3
        gc_pos_v2 = (X_pos[:, 2, :] + X_pos[:, 3, :]).mean()  # C + G
        gc_neg_v2 = (X_neg[:, 2, :] + X_neg[:, 3, :]).mean()
        
        print(f"\n{rbp_name}")
        print(f"  Samples: {len(X_pos)} positives, {len(X_neg)} negatives")
        print(f"  Channel means (pos): {channel_sums_pos}")
        print(f"  Channel means (neg): {channel_sums_neg}")
        print(f"  GC content (v1: C=ch1, G=ch2): pos={gc_pos_v1:.4f}, neg={gc_neg_v1:.4f}, diff={gc_pos_v1-gc_neg_v1:.4f}")
        print(f"  GC content (v2: C=ch2, G=ch3): pos={gc_pos_v2:.4f}, neg={gc_neg_v2:.4f}, diff={gc_pos_v2-gc_neg_v2:.4f}")
        
        all_results.append({
            'RBP': rbp_name,
            'n_pos': len(X_pos),
            'n_neg': len(X_neg),
            'gc_pos_v1': gc_pos_v1,
            'gc_neg_v1': gc_neg_v1,
            'gc_diff_v1': gc_pos_v1 - gc_neg_v1,
            'gc_pos_v2': gc_pos_v2,
            'gc_neg_v2': gc_neg_v2,
            'gc_diff_v2': gc_pos_v2 - gc_neg_v2,
        })

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Check which interpretation makes more sense (should sum to ~1.0 for one-hot)
print("\nIf one-hot encoded properly, channels 0-3 should sum to 1.0 per position")
with h5py.File(h5_files[0], 'r') as f:
    X_sample = f['X_train'][0]
    for pos in [0, 50, 100, 150, 199]:
        print(f"  Position {pos}: sum of channels 0-3 = {X_sample[:4, pos].sum():.4f}")
        print(f"    Values: {X_sample[:4, pos]}")

# Print aggregate statistics
gc_diffs_v1 = [r['gc_diff_v1'] for r in all_results]
gc_diffs_v2 = [r['gc_diff_v2'] for r in all_results]

print(f"\nAggregate GC difference (positive - negative):")
print(f"  Version 1 (C=ch1, G=ch2): mean={np.mean(gc_diffs_v1):.4f}, std={np.std(gc_diffs_v1):.4f}")
print(f"  Version 2 (C=ch2, G=ch3): mean={np.mean(gc_diffs_v2):.4f}, std={np.std(gc_diffs_v2):.4f}")

# Per-RBP warnings
print("\n" + "=" * 80)
print("PER-RBP GC BIAS ANALYSIS (threshold: |diff| > 0.08)")
print("=" * 80)
biased_rbps = []
for r in all_results:
    max_diff = max(abs(r['gc_diff_v1']), abs(r['gc_diff_v2']))
    if max_diff > 0.08:
        biased_rbps.append(r['RBP'])
        print(f"  ⚠️  {r['RBP']}: GC diff = {r['gc_diff_v1']:.4f} (v1), {r['gc_diff_v2']:.4f} (v2)")

if biased_rbps:
    print(f"\n⚠️  WARNING: {len(biased_rbps)}/{len(all_results)} RBPs have significant GC content bias!")
    print(f"    Affected: {', '.join(biased_rbps)}")
    print("    The model might learn GC content rather than true RBP binding motifs.")
    print("    Consider GC-matched negative controls or GC-content normalization.")
else:
    print("\n✓ GC content appears balanced between positives and negatives.")

# Save results to CSV
df = pd.DataFrame(all_results)
csv_path = os.path.join(output_dir, "gc_content_analysis.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")

# Also save a summary text file
summary_path = os.path.join(output_dir, "gc_content_summary.txt")
with open(summary_path, 'w') as f:
    f.write("GC CONTENT ANALYSIS - POSITIVES vs NEGATIVES\n")
    f.write("=" * 80 + "\n\n")
    
    for r in all_results:
        f.write(f"{r['RBP']}\n")
        f.write(f"  Samples: {r['n_pos']} positives, {r['n_neg']} negatives\n")
        f.write(f"  GC content (v1): pos={r['gc_pos_v1']:.4f}, neg={r['gc_neg_v1']:.4f}, diff={r['gc_diff_v1']:.4f}\n")
        f.write(f"  GC content (v2): pos={r['gc_pos_v2']:.4f}, neg={r['gc_neg_v2']:.4f}, diff={r['gc_diff_v2']:.4f}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Aggregate GC difference (positive - negative):\n")
    f.write(f"  Version 1 (C=ch1, G=ch2): mean={np.mean(gc_diffs_v1):.4f}, std={np.std(gc_diffs_v1):.4f}\n")
    f.write(f"  Version 2 (C=ch2, G=ch3): mean={np.mean(gc_diffs_v2):.4f}, std={np.std(gc_diffs_v2):.4f}\n\n")
    
    f.write("RBPs with significant GC bias (|diff| > 0.08):\n")
    for r in all_results:
        max_diff = max(abs(r['gc_diff_v1']), abs(r['gc_diff_v2']))
        if max_diff > 0.08:
            f.write(f"  ⚠️  {r['RBP']}: GC diff = {r['gc_diff_v1']:.4f} (v1), {r['gc_diff_v2']:.4f} (v2)\n")
    
    f.write(f"\nTotal: {len(biased_rbps)}/{len(all_results)} RBPs have significant GC content bias\n")

print(f"Summary saved to: {summary_path}")
