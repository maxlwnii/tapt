import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set paths
lamar_results_dir = '/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings/results/results_clip_data'
dnabert2_results_dir = '/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/DNABERT2/evalEmbeddings/results/results_clip_data_uhl'
output_dir = '/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/DNABERT2/evalEmbeddings/results/results_clip_data_uhl'
os.makedirs(output_dir, exist_ok=True)

# Load LAMAR results
random_df = pd.read_csv(os.path.join(lamar_results_dir, 'LAMAR_Random_L11_results.csv'))
tapt_df = pd.read_csv(os.path.join(lamar_results_dir, 'LAMAR_TAPT_L11_results.csv'))
pretrained_df = pd.read_csv(os.path.join(lamar_results_dir, 'LAMAR_Pretrained_L11_results.csv'))

# Load DNABERT2 results
dnabert2_df = pd.read_csv(os.path.join(dnabert2_results_dir, 'DNABERT2_results.csv'))

# Load OneHot CNN results
onehot_df = pd.read_csv(os.path.join(lamar_results_dir, 'OneHot_CNN_clip_results.csv'))

# Extract just the TF name (first part before underscore) for matching
def extract_tf_short(df):
    df = df.copy()
    df['TF_SHORT'] = df['TF'].astype(str).str.split('_').str[0]
    return df

random_df = extract_tf_short(random_df)
tapt_df = extract_tf_short(tapt_df)
pretrained_df = extract_tf_short(pretrained_df)
dnabert2_df = extract_tf_short(dnabert2_df)
onehot_df = extract_tf_short(onehot_df)

# Add model labels
random_df['Model'] = 'Random'
tapt_df['Model'] = 'TAPT'
pretrained_df['Model'] = 'Pretrained'
dnabert2_df['Model'] = 'DNABERT2'
onehot_df['Model'] = 'OneHot'

# Select relevant columns and combine (use TF_SHORT for matching)
cols = ['TF_SHORT', 'Accuracy', 'AUROC', 'AUPR', 'Model']
all_data = pd.concat([
    onehot_df[cols],
    random_df[cols],
    tapt_df[cols], 
    pretrained_df[cols],
    dnabert2_df[cols]
], ignore_index=True)

# Rename TF_SHORT back to TF for display
all_data = all_data.rename(columns={'TF_SHORT': 'TF'})

print("Data loaded:")
print(f"  OneHot: {len(onehot_df)} RBPs")
print(f"  Random: {len(random_df)} RBPs")
print(f"  TAPT: {len(tapt_df)} RBPs")
print(f"  Pretrained: {len(pretrained_df)} RBPs")
print(f"  DNABERT2: {len(dnabert2_df)} RBPs")

# Get common TFs across all models (using TF_SHORT)
common_tfs = set(random_df['TF_SHORT']) & set(tapt_df['TF_SHORT']) & set(pretrained_df['TF_SHORT']) & set(dnabert2_df['TF_SHORT']) & set(onehot_df['TF_SHORT'])
print(f"\nCommon TFs across all models: {len(common_tfs)}")
print(common_tfs)

# Filter to common TFs only
all_data = all_data[all_data['TF'].isin(common_tfs)]

# Define model order and colors
model_order = ['OneHot', 'Random', 'Pretrained', 'TAPT', 'DNABERT2']
model_colors = {
    'OneHot': '#9b59b6',      # Purple
    'Random': '#808080',      # Gray
    'Pretrained': '#2ecc71',  # Green
    'TAPT': '#3498db',        # Blue
    'DNABERT2': '#e74c3c'     # Red
}

# Create figure with 3 subplots for Accuracy, AUROC, AUPR
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = ['Accuracy', 'AUROC', 'AUPR']
titles = ['Accuracy Comparison', 'AUROC Comparison', 'AUPR Comparison']

for ax, metric, title in zip(axes, metrics, titles):
    # Pivot data for grouped bar chart
    pivot_data = all_data.pivot(index='TF', columns='Model', values=metric)
    pivot_data = pivot_data[model_order]  # Reorder columns
    
    # Create grouped bar chart
    x = np.arange(len(pivot_data))
    width = 0.15  # Narrower for 5 models
    
    for i, model in enumerate(model_order):
        bars = ax.bar(x + i*width, pivot_data[model], width, 
                     label=model, color=model_colors[model], alpha=0.8)
    
    ax.set_xlabel('RBP', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)  # Center for 5 bars
    ax.set_xticklabels(pivot_data.index, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='lower right')
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'all_models_comparison_bars.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved bar chart to {os.path.join(output_dir, 'all_models_comparison_bars.png')}")

# Create scatter plot: Y-axis for all models, X-axis for RBPs
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, metric, title in zip(axes, metrics, titles):
    for i, tf in enumerate(sorted(common_tfs)):
        for j, model in enumerate(model_order):
            value = all_data[(all_data['TF'] == tf) & (all_data['Model'] == model)][metric].values
            if len(value) > 0:
                ax.scatter(i + j*0.12, value[0], color=model_colors[model], s=80, alpha=0.8,
                          label=model if i == 0 else '')
    
    ax.set_xlabel('RBP', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(len(common_tfs)) + 0.24)  # Center for 5 models
    ax.set_xticklabels(sorted(common_tfs), rotation=45, ha='right', fontsize=9)
    ax.legend(loc='lower right')
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'all_models_comparison_scatter.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved scatter plot to {os.path.join(output_dir, 'all_models_comparison_scatter.png')}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS (Mean across common RBPs)")
print("="*60)
summary = all_data.groupby('Model')[metrics].mean()
summary = summary.reindex(model_order)
print(summary.to_string())

# Create summary bar plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.2

for i, model in enumerate(model_order):
    means = [summary.loc[model, m] for m in metrics]
    ax.bar(x + i*width, means, width, label=model, color=model_colors[model], alpha=0.8)

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Mean Value', fontsize=12)
ax.set_title('Average Performance Across RBPs', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(loc='lower right')
ax.set_ylim(0.7, 1.0)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'all_models_summary.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved summary plot to {os.path.join(output_dir, 'all_models_summary.png')}")

print("\nDone!")
