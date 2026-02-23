"""
Plot OneHot CNN vs LAMAR models and DNABERT2
x-axis: OneHot CNN
y-axis: LAMAR variants (Pretrained, Random, TAPT) + DNABERT2
Each TF has a different color, different symbols for different models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.lines import Line2D

# Set paths
results_dir = '/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings/results/results_clip_data'
dnabert2_path = '/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/DNABERT2/evalEmbeddings/results/results_clip_data_uhl/DNABERT2_results.csv'
output_dir = '/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings/results/results_clip_data'
os.makedirs(output_dir, exist_ok=True)

# Load the data
onehot_df = pd.read_csv(os.path.join(results_dir, 'OneHot_CNN_clip_results.csv'))
lamar_pretrained_df = pd.read_csv(os.path.join(results_dir, 'LAMAR_Pretrained_L11_results.csv'))
lamar_random_df = pd.read_csv(os.path.join(results_dir, 'LAMAR_Random_L11_results.csv'))
lamar_tapt_df = pd.read_csv(os.path.join(results_dir, 'LAMAR_TAPT_L11_results.csv'))
dnabert2_df = pd.read_csv(dnabert2_path)

# Extract short TF names
def extract_tf_short(tf_name):
    return tf_name.split('_')[0]

onehot_df['TF_SHORT'] = onehot_df['TF'].apply(extract_tf_short)
lamar_pretrained_df['TF_SHORT'] = lamar_pretrained_df['TF'].apply(extract_tf_short)
lamar_random_df['TF_SHORT'] = lamar_random_df['TF'].apply(extract_tf_short)
lamar_tapt_df['TF_SHORT'] = lamar_tapt_df['TF'].apply(extract_tf_short)
dnabert2_df['TF_SHORT'] = dnabert2_df['TF'].apply(extract_tf_short)

# Merge all dataframes on TF
merged_df = pd.merge(
    onehot_df[['TF_SHORT', 'Accuracy', 'AUROC', 'AUPR']].rename(columns={
        'Accuracy': 'onehot_accuracy',
        'AUROC': 'onehot_auroc',
        'AUPR': 'onehot_aupr'
    }),
    lamar_pretrained_df[['TF_SHORT', 'Accuracy', 'AUROC', 'AUPR']].rename(columns={
        'Accuracy': 'pretrained_accuracy',
        'AUROC': 'pretrained_auroc',
        'AUPR': 'pretrained_aupr'
    }),
    on='TF_SHORT'
)

merged_df = pd.merge(merged_df,
    lamar_random_df[['TF_SHORT', 'Accuracy', 'AUROC', 'AUPR']].rename(columns={
        'Accuracy': 'random_accuracy',
        'AUROC': 'random_auroc',
        'AUPR': 'random_aupr'
    }),
    on='TF_SHORT'
)

merged_df = pd.merge(merged_df,
    lamar_tapt_df[['TF_SHORT', 'Accuracy', 'AUROC', 'AUPR']].rename(columns={
        'Accuracy': 'tapt_accuracy',
        'AUROC': 'tapt_auroc',
        'AUPR': 'tapt_aupr'
    }),
    on='TF_SHORT'
)

merged_df = pd.merge(merged_df,
    dnabert2_df[['TF_SHORT', 'Accuracy', 'AUROC', 'AUPR']].rename(columns={
        'Accuracy': 'dnabert2_accuracy',
        'AUROC': 'dnabert2_auroc',
        'AUPR': 'dnabert2_aupr'
    }),
    on='TF_SHORT'
)

print(f"Found {len(merged_df)} overlapping TFs:")
print(merged_df[['TF_SHORT']])

# Create color palette for TFs
tfs_short = sorted(merged_df['TF_SHORT'].unique())
colors = sns.color_palette('tab10', len(tfs_short))
tf_color_map = dict(zip(tfs_short, colors))

# Define markers for different models
markers = {
    'pretrained': '^',
    'random': 's',
    'tapt': 'o',
    'dnabert2': 'D'
}
variant_labels = {
    'pretrained': 'LAMAR Pretrained L11',
    'random': 'LAMAR Random L11',
    'tapt': 'LAMAR TAPT L11',
    'dnabert2': 'DNABERT2'
}

# Create plots for each metric
metrics = [
    ('auroc', 'AUROC', 'auroc_onehot_vs_all_models_with_dnabert2.png'),
    ('aupr', 'AUPR', 'aupr_onehot_vs_all_models_with_dnabert2.png'),
    ('accuracy', 'Accuracy', 'accuracy_onehot_vs_all_models_with_dnabert2.png')
]

for metric_key, metric_label, output_filename in metrics:
    fig, ax = plt.subplots(figsize=(12, 10))
    
    x_col = f'onehot_{metric_key}'
    y_variants = [
        ('pretrained', f'pretrained_{metric_key}'),
        ('random', f'random_{metric_key}'),
        ('tapt', f'tapt_{metric_key}'),
        ('dnabert2', f'dnabert2_{metric_key}')
    ]
    
    # Plot points for each variant
    for var_key, y_col in y_variants:
        for _, row in merged_df.iterrows():
            tf_short = row['TF_SHORT']
            ax.scatter(row[x_col], row[y_col], 
                      color=tf_color_map[tf_short], 
                      marker=markers[var_key], 
                      s=150, alpha=0.8, 
                      edgecolors='black', linewidth=0.5)
    
    # Add TF labels to points (use pretrained for positioning)
    for _, row in merged_df.iterrows():
        tf_short = row['TF_SHORT']
        ax.text(row[x_col] + 0.003, row[f'pretrained_{metric_key}'], tf_short, 
               fontsize=9, alpha=0.7, va='center', ha='left')
    
    # Add diagonal line (y=x)
    all_vals = []
    all_vals.extend(merged_df[x_col].values)
    for _, y_col in y_variants:
        all_vals.extend(merged_df[y_col].values)
    
    min_val = min(all_vals)
    max_val = max(all_vals)
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin], 
            [min_val - margin, max_val + margin], 
            'k--', linewidth=2, alpha=0.5, label='y = x')
    
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    ax.set_xlabel(f'OneHot CNN {metric_label}', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Model {metric_label}', fontsize=13, fontweight='bold')
    ax.set_title(f'{metric_label} Comparison: OneHot CNN vs All Models', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create custom legend for markers
    variant_elements = [
        Line2D([0], [0], marker=markers[var], color='w', label=variant_labels[var],
               markerfacecolor='gray', markeredgecolor='black', markersize=10, linewidth=1.5)
        for var, _ in y_variants
    ]
    variant_elements.append(Line2D([0], [0], linestyle='--', color='black', linewidth=2, label='y = x'))
    legend1 = ax.legend(handles=variant_elements, loc='lower right', fontsize=11, title='Models', title_fontsize=12)
    
    # Add TF color legend
    tf_elements = [Line2D([0], [0], marker='o', color='w', label=tf,
                         markerfacecolor=tf_color_map[tf], markeredgecolor=tf_color_map[tf], markersize=8) 
                  for tf in tfs_short]
    legend2 = ax.legend(handles=tf_elements, loc='upper left', fontsize=10, ncol=2, 
                       title='Transcription Factors', title_fontsize=11)
    ax.add_artist(legend1)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nSaved {metric_label} plot to {output_path}')
    plt.close()

# Print summary statistics
print(f"\n{'='*80}")
print("Mean Results across TFs:")
print(f"{'='*80}")
print(f"{'Metric':<20} {'OneHot':>12} {'Pretrained':>12} {'Random':>12} {'TAPT':>12} {'DNABERT2':>12}")
print(f"{'-'*80}")

for metric_key, metric_label, _ in metrics:
    x_col = f'onehot_{metric_key}'
    vals = [merged_df[x_col].mean()]
    
    for var_key, y_col in [('pretrained', f'pretrained_{metric_key}'),
                            ('random', f'random_{metric_key}'),
                            ('tapt', f'tapt_{metric_key}'),
                            ('dnabert2', f'dnabert2_{metric_key}')]:
        vals.append(merged_df[y_col].mean())
    
    print(f"{metric_label:<20} {vals[0]:>12.4f} {vals[1]:>12.4f} {vals[2]:>12.4f} {vals[3]:>12.4f} {vals[4]:>12.4f}")

print(f"\n{'='*80}")
print("Individual TF Results (AUROC):")
print(f"{'='*80}")
print(f"{'TF':<15} {'OneHot':>10} {'Pretrained':>12} {'Random':>10} {'TAPT':>10} {'DNABERT2':>10}")
print(f"{'-'*80}")

for _, row in merged_df.iterrows():
    tf_short = row['TF_SHORT']
    print(f"{tf_short:<15} {row['onehot_auroc']:>10.4f} {row['pretrained_auroc']:>12.4f} "
          f"{row['random_auroc']:>10.4f} {row['tapt_auroc']:>10.4f} {row['dnabert2_auroc']:>10.4f}")
