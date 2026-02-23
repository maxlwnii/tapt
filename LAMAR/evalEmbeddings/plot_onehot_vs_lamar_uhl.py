"""
Plot OneHot CNN vs LAMAR variants (TAPT 256, Random 256, TAPT, Random) from clip_data_uhl
x-axis: OneHot CNN
y-axis: LAMAR variants
Each TF has a different color
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Paths
onehot_path = '/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings/results/results_clip_data/OneHot_CNN_clip_results.csv'
results_uhl_dir = '/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings/results/results_clip_data_uhl'
output_dir = results_uhl_dir

os.makedirs(output_dir, exist_ok=True)

# Load OneHot results
onehot_df = pd.read_csv(onehot_path)

# Load LAMAR variants from results_clip_data_uhl
lamar_tapt_256_df = pd.read_csv(os.path.join(results_uhl_dir, 'LAMAR_256_TAPT_L11_results.csv'))
lamar_random_256_df = pd.read_csv(os.path.join(results_uhl_dir, 'LAMAR_256_Random_L11_results.csv'))
lamar_tapt_df = pd.read_csv(os.path.join(results_uhl_dir, 'LAMAR_TAPT_L11_results.csv'))
lamar_random_df = pd.read_csv(os.path.join(results_uhl_dir, 'LAMAR_Random_L11_results.csv'))

# Extract just the TF name (first part before underscore)
def extract_tf_short_name(tf_name):
    return tf_name.split('_')[0]

# Merge all dataframes on TF
def prepare_data(df):
    df = df.copy()
    df['TF_SHORT'] = df['TF'].apply(extract_tf_short_name)
    return df

onehot_df = prepare_data(onehot_df)
lamar_tapt_256_df = prepare_data(lamar_tapt_256_df)
lamar_random_256_df = prepare_data(lamar_random_256_df)
lamar_tapt_df = prepare_data(lamar_tapt_df)
lamar_random_df = prepare_data(lamar_random_df)

# Merge on TF_SHORT
merged_auroc = pd.merge(
    onehot_df[['TF_SHORT', 'AUROC']].rename(columns={'AUROC': 'onehot_auroc'}),
    lamar_tapt_256_df[['TF_SHORT', 'AUROC']].rename(columns={'AUROC': 'tapt_256_auroc'}),
    on='TF_SHORT'
)
merged_auroc = pd.merge(merged_auroc, 
    lamar_random_256_df[['TF_SHORT', 'AUROC']].rename(columns={'AUROC': 'random_256_auroc'}),
    on='TF_SHORT'
)
merged_auroc = pd.merge(merged_auroc,
    lamar_tapt_df[['TF_SHORT', 'AUROC']].rename(columns={'AUROC': 'tapt_auroc'}),
    on='TF_SHORT'
)
merged_auroc = pd.merge(merged_auroc,
    lamar_random_df[['TF_SHORT', 'AUROC']].rename(columns={'AUROC': 'random_auroc'}),
    on='TF_SHORT'
)

merged_aupr = pd.merge(
    onehot_df[['TF_SHORT', 'AUPR']].rename(columns={'AUPR': 'onehot_aupr'}),
    lamar_tapt_256_df[['TF_SHORT', 'AUPR']].rename(columns={'AUPR': 'tapt_256_aupr'}),
    on='TF_SHORT'
)
merged_aupr = pd.merge(merged_aupr,
    lamar_random_256_df[['TF_SHORT', 'AUPR']].rename(columns={'AUPR': 'random_256_aupr'}),
    on='TF_SHORT'
)
merged_aupr = pd.merge(merged_aupr,
    lamar_tapt_df[['TF_SHORT', 'AUPR']].rename(columns={'AUPR': 'tapt_aupr'}),
    on='TF_SHORT'
)
merged_aupr = pd.merge(merged_aupr,
    lamar_random_df[['TF_SHORT', 'AUPR']].rename(columns={'AUPR': 'random_aupr'}),
    on='TF_SHORT'
)

merged_accuracy = pd.merge(
    onehot_df[['TF_SHORT', 'Accuracy']].rename(columns={'Accuracy': 'onehot_accuracy'}),
    lamar_tapt_256_df[['TF_SHORT', 'Accuracy']].rename(columns={'Accuracy': 'tapt_256_accuracy'}),
    on='TF_SHORT'
)
merged_accuracy = pd.merge(merged_accuracy,
    lamar_random_256_df[['TF_SHORT', 'Accuracy']].rename(columns={'Accuracy': 'random_256_accuracy'}),
    on='TF_SHORT'
)
merged_accuracy = pd.merge(merged_accuracy,
    lamar_tapt_df[['TF_SHORT', 'Accuracy']].rename(columns={'Accuracy': 'tapt_accuracy'}),
    on='TF_SHORT'
)
merged_accuracy = pd.merge(merged_accuracy,
    lamar_random_df[['TF_SHORT', 'Accuracy']].rename(columns={'Accuracy': 'random_accuracy'}),
    on='TF_SHORT'
)

print(f"Found {len(merged_auroc)} overlapping TFs")

# Create color palette for TFs
tfs = sorted(merged_auroc['TF_SHORT'].unique())
colors = sns.color_palette('tab10', len(tfs))
tf_color_map = dict(zip(tfs, colors))

# Define markers for different variants
markers = {
    'tapt_256': 'o',
    'random_256': 's',
    'tapt': '^',
    'random': 'D'
}
variant_labels = {
    'tapt_256': 'LAMAR TAPT 256',
    'random_256': 'LAMAR Random 256',
    'tapt': 'LAMAR TAPT',
    'random': 'LAMAR Random'
}

# Create plots for each metric
metrics = [
    ('AUROC', merged_auroc, ['onehot_auroc', 'tapt_256_auroc', 'random_256_auroc', 'tapt_auroc', 'random_auroc']),
    ('AUPR', merged_aupr, ['onehot_aupr', 'tapt_256_aupr', 'random_256_aupr', 'tapt_aupr', 'random_aupr']),
    ('Accuracy', merged_accuracy, ['onehot_accuracy', 'tapt_256_accuracy', 'random_256_accuracy', 'tapt_accuracy', 'random_accuracy'])
]

for metric_name, df_merged, columns in metrics:
    fig, ax = plt.subplots(figsize=(12, 10))
    
    x_col = columns[0]
    y_variants = [
        ('tapt_256', columns[1]),
        ('random_256', columns[2]),
        ('tapt', columns[3]),
        ('random', columns[4])
    ]
    
    # Plot points for each variant
    for var_key, y_col in y_variants:
        for _, row in df_merged.iterrows():
            ax.scatter(row[x_col], row[y_col], 
                      color=tf_color_map[row['TF_SHORT']], 
                      marker=markers[var_key], 
                      s=150, alpha=0.8, 
                      label=f"{row['TF_SHORT']} ({variant_labels[var_key]})" if var_key == 'tapt_256' else "")
    
    # Add diagonal line (y=x)
    min_val = min(df_merged[x_col].min(), 
                  df_merged[columns[1]].min(), df_merged[columns[2]].min(),
                  df_merged[columns[3]].min(), df_merged[columns[4]].min())
    max_val = max(df_merged[x_col].max(),
                  df_merged[columns[1]].max(), df_merged[columns[2]].max(),
                  df_merged[columns[3]].max(), df_merged[columns[4]].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='y = x')
    
    ax.set_xlabel(f'OneHot CNN {metric_name}', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'LAMAR Models {metric_name}', fontsize=13, fontweight='bold')
    ax.set_title(f'{metric_name} Comparison: OneHot CNN vs LAMAR Variants (UHL)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create custom legend for markers
    from matplotlib.lines import Line2D
    variant_elements = [
        Line2D([0], [0], marker=markers['tapt_256'], color='w', label='LAMAR TAPT 256',
               markerfacecolor='black', markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker=markers['random_256'], color='w', label='LAMAR Random 256',
               markerfacecolor='black', markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker=markers['tapt'], color='w', label='LAMAR TAPT',
               markerfacecolor='black', markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker=markers['random'], color='w', label='LAMAR Random',
               markerfacecolor='black', markeredgecolor='black', markersize=10),
        Line2D([0], [0], linestyle='--', color='black', linewidth=2, label='y = x')
    ]
    legend1 = ax.legend(handles=variant_elements, loc='lower right', fontsize=11, title='Models', title_fontsize=12)
    
    # Add TF color legend
    tf_elements = [Line2D([0], [0], marker='o', color='w', label=tf,
                         markerfacecolor=tf_color_map[tf], markeredgecolor=tf_color_map[tf], markersize=8) 
                  for tf in tfs]
    legend2 = ax.legend(handles=tf_elements, loc='upper left', fontsize=10, ncol=2, title='Transcription Factors', title_fontsize=11)
    ax.add_artist(legend1)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'{metric_name.lower()}_onehot_vs_lamar_uhl.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved {metric_name} plot to {output_path}')
    plt.close()

# Print summary statistics
print(f"\n{'='*60}")
print("Mean AUROC across TFs:")
print(f"{'='*60}")
print(f"  OneHot CNN:      {merged_auroc['onehot_auroc'].mean():.4f}")
print(f"  LAMAR TAPT 256:  {merged_auroc['tapt_256_auroc'].mean():.4f}")
print(f"  LAMAR Random 256:{merged_auroc['random_256_auroc'].mean():.4f}")
print(f"  LAMAR TAPT:      {merged_auroc['tapt_auroc'].mean():.4f}")
print(f"  LAMAR Random:    {merged_auroc['random_auroc'].mean():.4f}")

print(f"\n{'='*60}")
print("Mean AUPR across TFs:")
print(f"{'='*60}")
print(f"  OneHot CNN:      {merged_aupr['onehot_aupr'].mean():.4f}")
print(f"  LAMAR TAPT 256:  {merged_aupr['tapt_256_aupr'].mean():.4f}")
print(f"  LAMAR Random 256:{merged_aupr['random_256_aupr'].mean():.4f}")
print(f"  LAMAR TAPT:      {merged_aupr['tapt_aupr'].mean():.4f}")
print(f"  LAMAR Random:    {merged_aupr['random_aupr'].mean():.4f}")

print(f"\n{'='*60}")
print("Mean Accuracy across TFs:")
print(f"{'='*60}")
print(f"  OneHot CNN:      {merged_accuracy['onehot_accuracy'].mean():.4f}")
print(f"  LAMAR TAPT 256:  {merged_accuracy['tapt_256_accuracy'].mean():.4f}")
print(f"  LAMAR Random 256:{merged_accuracy['random_256_accuracy'].mean():.4f}")
print(f"  LAMAR TAPT:      {merged_accuracy['tapt_accuracy'].mean():.4f}")
print(f"  LAMAR Random:    {merged_accuracy['random_accuracy'].mean():.4f}")
