import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

# Set the path to the results directory
results_dir = '/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/results'
output_dir = '/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/plots'
os.makedirs(output_dir, exist_ok=True)

# List of files to process
files = {
    ('Layer 5', 'LAMAR Pre-trained'): 'LAMAR_layer5_perf.csv',
    ('Layer 5', 'LAMAR Random'): 'LAMAR_layer5_random_perf.csv',
    ('Layer 11', 'LAMAR Pre-trained'): 'LAMAR_layer11_perf.csv',
    ('Layer 11', 'LAMAR Random'): 'LAMAR_layer11_random_perf.csv',
    ('Layer 5', 'LAMAR TAPT'): 'LAMAR_tapt_layer5_perf.csv',
    ('Layer 11', 'LAMAR TAPT'): 'LAMAR_tapt_layer11_perf.csv',
    ('Layer 10', 'NT Pre-trained'): 'NT10_perf.csv',
    ('Layer 10', 'NT Random'): 'NT_random_10_perf.csv',
    ('Layer 32', 'NT Pre-trained'): 'NT32_perf.csv',
    ('Layer 32', 'NT Random'): 'NT_random_32_perf.csv'
}

# Load and combine all data
all_data = []
for (layer, model_type), filename in files.items():
    path = os.path.join(results_dir, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['Layer'] = layer
        df['Model Type'] = model_type
        all_data.append(df)
    else:
        print(f"Warning: {path} not found.")

if not all_data:
    print("No data found to plot.")
    exit()

df_combined = pd.concat(all_data, ignore_index=True)

# Define metrics to plot
metrics = ['Accuracy', 'AUROC', 'AUPR']

for metric in metrics:
    plt.figure(figsize=(12, 8))
    
    # Create a boxplot to show the distribution
    sns.boxplot(data=df_combined, x='Layer', y=metric, hue='Model Type', 
                palette='Set3', showfliers=False)
    
    # Add an individual point for each TF, colored by TF
    # We use dodge=True to align points with the boxes
    sns.stripplot(data=df_combined, x='Layer', y=metric, hue='TF', 
                  palette='tab10', dodge=True, jitter=True, marker='o', alpha=0.7)
    
    # Adjust legend
    # The strip plot adds many entries to the legend. We want to clean it up.
    # We'll show Model Type in one part and TFs in another, or just handle it cleanly.
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Identify labels for Model Type (usually first 2 in boxplot)
    # And labels for TF (remaining)
    # Actually, seaborn combines them if we are not careful.
    # Let's just create a custom legend or filter the current one.
    
    # Filter legend to avoid duplicates and show only relevant info
    unique_labels = dict(zip(labels, handles))
    # We want model types first, then TFs
    model_types = ['LAMAR Pre-trained', 'LAMAR Random', 'LAMAR TAPT', 'NT Pre-trained', 'NT Random']
    tfs = sorted([l for l in labels if l not in model_types and l not in [f[1] for f in files.keys()]])
    
    legend_keys = [m for m in model_types if m in unique_labels] + tfs
    new_handles = [unique_labels[k] for k in legend_keys if k in unique_labels]
    new_labels = [k for k in legend_keys if k in unique_labels]
    
    plt.legend(new_handles, new_labels, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Calculate and print averages on the plot
    means = df_combined.groupby(['Layer', 'Model Type'])[metric].mean()
    summary_text = f"Averages for {metric}:\n"
    for (layer, model_type), mean_val in means.items():
        summary_text += f"{layer} ({model_type}): {mean_val:.4f}\n"
    
    # Place text box in the upper right (inside axes)
    plt.text(0.95, 0.05, summary_text, transform=plt.gca().transAxes, 
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title(f'Comparison of {metric} across Layers and Models')
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(output_dir, f'{metric.lower()}_comparison.png')
    plt.savefig(save_path)
    print(f'Saved {metric} plot to {save_path}')
    plt.close()

print("All plots created successfully.")
