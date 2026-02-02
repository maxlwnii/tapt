import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set paths
results_dir = '/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/results'
output_dir = '/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/plots'
os.makedirs(output_dir, exist_ok=True)

# Define models and their files
models = {
    'LAMAR Pre-trained': ['LAMAR_layer5_perf.csv', 'LAMAR_layer11_perf.csv'],
    'LAMAR TAPT': ['LAMAR_tapt_layer5_perf.csv', 'LAMAR_tapt_layer11_perf.csv'],
    'LAMAR Random': ['LAMAR_layer5_random_perf.csv', 'LAMAR_layer11_random_perf.csv'],
    'OneHot Baseline': ['OneHot_perf.csv']
}

# Data accumulation for comparison
all_model_avgs = []

# Get unique TFs for color mapping
all_tfs = set()
for model_name, files in models.items():
    for file in files:
        path = os.path.join(results_dir, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Ensure TF column exists, some CSVs might differ slightly? 
            # The scripts used 'TF' as header.
            if 'TF' in df.columns:
                all_tfs.update(df['TF'].unique())
all_tfs = sorted(list(all_tfs))

# Create color palette
# Use a palette that can handle enough colors if many TFs, otherwise tab10 repeats
if len(all_tfs) <= 10:
    colors = sns.color_palette('tab10', len(all_tfs))
else:
    colors = sns.color_palette('husl', len(all_tfs))
    
tf_color_map = dict(zip(all_tfs, colors))

for model_name, files in models.items():
    dfs = []
    for file in files:
        path = os.path.join(results_dir, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
        else:
            print(f"Warning: {path} not found.")
    
    if not dfs:
        print(f"No data for {model_name}")
        continue
    
    # Combine data
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Fix: Clean OneHot Baseline TF names (remove suffixes like _K562_200)
    if model_name == 'OneHot Baseline':
        # Remove suffix matching _CellLine_Number (e.g., _K562_200)
        df_combined['TF'] = df_combined['TF'].str.replace(r'_[^_]+_\d+$', '', regex=True)
        print(f"Cleaned OneHot TF names example: {df_combined['TF'].head().tolist()}")

    # Group by TF and average AUROC
    avg_auroc = df_combined.groupby('TF')['AUROC'].mean().reset_index()
    
    # Add to all models collection for summary plot
    avg_auroc['Model'] = model_name
    all_model_avgs.append(avg_auroc)

    # Overall average AUROC
    overall_avg = avg_auroc['AUROC'].mean()
    
    # Plot
    plt.figure(figsize=(12, 6))
    for tf in all_tfs:
        if tf in avg_auroc['TF'].values:
            value = avg_auroc.loc[avg_auroc['TF'] == tf, 'AUROC'].values[0]
            plt.scatter(tf, value, color=tf_color_map[tf], s=100, label=tf if tf not in plt.gca().get_legend_handles_labels()[1] else "")
        # else:
            # plt.scatter(tf, 0, color=tf_color_map[tf], s=100, alpha=0.3)  # Placeholder if missing - removed to make cleaner
    
    plt.xlabel('RBPS')
    plt.ylabel('Average AUROC')
    plt.title(f'Average AUROC per TF for {model_name}')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(title='TF', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add overall average as text
    plt.text(0.95, 0.95, f'Overall Average AUROC: {overall_avg:.4f}', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(output_dir, f'{model_name.replace(" ", "_").lower()}_auroc_per_tf_dots.png')
    plt.savefig(save_path)
    print(f'Saved plot for {model_name} to {save_path}')
    print(f'Overall Average AUROC for {model_name}: {overall_avg:.4f}')
    plt.close()

# Combined Comparison Plot
if all_model_avgs:
    combined_df = pd.concat(all_model_avgs, ignore_index=True)
    
    # 1. Bar plot of Overall Averages
    overall_performance = combined_df.groupby('Model')['AUROC'].mean().sort_values()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(overall_performance.index, overall_performance.values, color=sns.color_palette("viridis", len(overall_performance)))
    plt.ylim(0, 1.05)
    plt.ylabel('Overall Average AUROC')
    plt.title('Overall Model Comparison (Average across all TFs)')
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
                
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_comparison_bar.png'))
    print("Saved overall_comparison_bar.png")
    
    # 2. Dot plot comparison per TF
    plt.figure(figsize=(14, 8))
    sns.stripplot(data=combined_df, x='TF', y='AUROC', hue='Model', dodge=True, size=8, alpha=0.8)
    plt.title('Performance Comparison per TF')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_per_tf_dots.png'))
    print("Saved comparison_per_tf_dots.png")

    # 3. Scatter plot: LAMAR models vs OneHot
    print("Generating LAMAR vs OneHot scatter plot...")
    
    # Extract OneHot data, group by TF to get average
    onehot_df = combined_df[combined_df['Model'] == 'OneHot Baseline'].groupby('TF')['AUROC'].mean().reset_index().rename(columns={'AUROC': 'AUROC_OneHot'})
    print("OneHot TFs:", onehot_df['TF'].tolist())

    # Check if we have OneHot data
    if not onehot_df.empty:
        # Define LAMAR models to plot
        lamar_models = ['LAMAR Pre-trained', 'LAMAR TAPT', 'LAMAR Random']
        # Remove markers from here as we will use different colors for TFs
        
        for model in lamar_models:
            plt.figure(figsize=(8, 8))
            
            # Plot diagonal line
            plt.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.5) # Removed label to avoid y=x box
            
            # Get model data, group by TF to get average
            model_df = combined_df[combined_df['Model'] == model].groupby('TF')['AUROC'].mean().reset_index()
            print(f"{model} TFs found: {len(model_df)}")
            
            # Merge with OneHot
            merged_df = pd.merge(onehot_df, model_df, on='TF', suffixes=('_OneHot', '_LAMAR'))
            print(f"Comparison pairs for {model}: {len(merged_df)}")
            
            if not merged_df.empty:
                # Calculate averages
                avg_onehot = merged_df['AUROC_OneHot'].mean()
                avg_model = merged_df['AUROC'].mean()

                # Plot each TF with its unique color
                for _, row in merged_df.iterrows():
                    tf = row['TF']
                    plt.scatter(row['AUROC_OneHot'], row['AUROC'], 
                               color=tf_color_map.get(tf, 'black'), 
                               label=tf, alpha=0.9, s=120, edgecolors='white', linewidths=0.5)
                
                # Add text for averages in bottom middle
                stats_text = (f"{model} Avg: {avg_model:.4f}\n"
                              f"OneHot Avg: {avg_onehot:.4f}")
                plt.text(0.5, 0.05, stats_text, transform=plt.gca().transAxes,
                         fontsize=12, verticalalignment='bottom', horizontalalignment='center',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

                plt.title(f'Performance Comparison: {model} vs OneHot Baseline')
                plt.xlabel('OneHot Baseline Average AUROC')
                plt.ylabel(f'{model} Average AUROC')
                plt.xlim(0.8, 1.01) # Zoom in to relevant high performance area
                plt.ylim(0.8, 1.01)
                
                # Add legend for RBPS in top left
                plt.legend(title='RBPS', loc='upper left', fontsize='small', ncol=2)
                
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.gca().set_aspect('equal', adjustable='box')
                
                filename = f"{model.replace(' ', '_').lower()}_vs_onehot_scatter.png"
                plt.savefig(os.path.join(output_dir, filename))
                print(f"Saved {filename}")
            
            plt.close()
    else:
        print("Skipping scatter plot: OneHot Baseline data missing in combined results.")