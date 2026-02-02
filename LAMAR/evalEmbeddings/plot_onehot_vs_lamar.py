import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set paths
results_dir = '/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings/results_clip_data'
output_dir = '/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings/plots'
os.makedirs(output_dir, exist_ok=True)

# Load the data
lamar_df = pd.read_csv(os.path.join(results_dir, 'LAMAR_Pretrained_L11_results.csv'))
onehot_df = pd.read_csv(os.path.join(results_dir, 'OneHot_CNN_clip_results.csv'))

# Rename columns for consistency
lamar_df = lamar_df.rename(columns={'AUROC': 'AUROC_LAMAR', 'AUPR': 'AUPR_LAMAR'})
onehot_df = onehot_df.rename(columns={'AUROC': 'AUROC_OneHot', 'AUPR': 'AUPR_OneHot'})

# Merge on TF
merged_df = pd.merge(lamar_df[['TF', 'AUROC_LAMAR', 'AUPR_LAMAR']], 
                     onehot_df[['TF', 'AUROC_OneHot', 'AUPR_OneHot']], 
                     on='TF')

# Print results
print("AUROC and AUPR Results:")
print("=" * 50)
avg_auroc_lamar = merged_df['AUROC_LAMAR'].mean()
avg_aupr_lamar = merged_df['AUPR_LAMAR'].mean()
avg_auroc_onehot = merged_df['AUROC_OneHot'].mean()
avg_aupr_onehot = merged_df['AUPR_OneHot'].mean()

print(f"LAMAR Pretrained L11:")
print(f"  Average AUROC: {avg_auroc_lamar:.4f}")
print(f"  Average AUPR: {avg_aupr_lamar:.4f}")
print()
print(f"OneHot CNN:")
print(f"  Average AUROC: {avg_auroc_onehot:.4f}")
print(f"  Average AUPR: {avg_aupr_onehot:.4f}")
print()

# Create color palette for TFs
tfs = merged_df['TF'].unique()
colors = sns.color_palette('tab10', len(tfs))
tf_color_map = dict(zip(tfs, colors))

# Plot AUROC scatter
plt.figure(figsize=(10, 8))
for tf in tfs:
    subset = merged_df[merged_df['TF'] == tf]
    plt.scatter(subset['AUROC_OneHot'], subset['AUROC_LAMAR'], 
                color=tf_color_map[tf], label=tf, s=100, alpha=0.8)

# Add diagonal line
min_val = min(merged_df['AUROC_OneHot'].min(), merged_df['AUROC_LAMAR'].min())
max_val = max(merged_df['AUROC_OneHot'].max(), merged_df['AUROC_LAMAR'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal performance')

plt.xlabel('OneHot CNN AUROC')
plt.ylabel('LAMAR Pretrained L11 AUROC')
plt.title('AUROC Comparison: OneHot CNN vs LAMAR Pretrained Layer 11')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'OneHot_vs_LAMAR_L11_AUROC_scatter.png'), dpi=150, bbox_inches='tight')
plt.show()

# Plot AUPR scatter
plt.figure(figsize=(10, 8))
for tf in tfs:
    subset = merged_df[merged_df['TF'] == tf]
    plt.scatter(subset['AUPR_OneHot'], subset['AUPR_LAMAR'], 
                color=tf_color_map[tf], label=tf, s=100, alpha=0.8)

# Add diagonal line
min_val = min(merged_df['AUPR_OneHot'].min(), merged_df['AUPR_LAMAR'].min())
max_val = max(merged_df['AUPR_OneHot'].max(), merged_df['AUPR_LAMAR'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal performance')

plt.xlabel('OneHot CNN AUPR')
plt.ylabel('LAMAR Pretrained L11 AUPR')
plt.title('AUPR Comparison: OneHot CNN vs LAMAR Pretrained Layer 11')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'OneHot_vs_LAMAR_L11_AUPR_scatter.png'), dpi=150, bbox_inches='tight')
plt.show()

print("Scatter plots saved to:", output_dir)