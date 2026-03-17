"""Plot AUROC: x=OneHot, y=LAMAR (Random, TAPT, Pretrained) + DNABERT2 per TF.

Usage:
  python plot_onehot_vs_all_models.py \
      --onehot path/to/OneHot_results.csv \
      --random path/to/LAMAR_Random_L11_results.csv \
      --tapt path/to/LAMAR_TAPT_L11_results.csv \
      --pretrained path/to/LAMAR_Pretrained_L11_results.csv \
      --dnabert2 path/to/DNABERT2_results.csv \
      --out auroc_onehot_vs_all_models.png

The script tries to infer TF and AUROC column names; common names supported.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

COL_CANDIDATES = ['tf', 'TF', 'name', 'RBP', 'rbp', 'protein']
METRIC_CANDIDATES = {
    'AUROC': ['auroc', 'AUROC', 'auc', 'AUC', 'au_roc', 'au_auroc'],
    'AUPR': ['aupr', 'AUPR', 'au_pr', 'AUPRC', 'auprc'],
    'Accuracy': ['accuracy', 'Accuracy', 'acc', 'Acc']
}


def find_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_and_extract(path: Path, metric_key='AUROC', tf_col=None, metric_col=None):
    df = pd.read_csv(path)
    if tf_col is None:
        tf_col = find_column(df, COL_CANDIDATES)
    if metric_col is None:
        candidates = METRIC_CANDIDATES.get(metric_key, [])
        metric_col = find_column(df, candidates)
    if tf_col is None or metric_col is None:
        raise ValueError(f"Couldn't infer TF or {metric_key} column in {path}. Available: {list(df.columns)}")
    out = df[[tf_col, metric_col]].rename(columns={tf_col: 'TF', metric_col: 'METRIC'})
    out['TF'] = out['TF'].astype(str)
    out['METRIC'] = pd.to_numeric(out['METRIC'], errors='coerce')
    return out.dropna(subset=['METRIC'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onehot', required=True, type=Path)
    parser.add_argument('--random', required=True, type=Path)
    parser.add_argument('--tapt', required=True, type=Path)
    parser.add_argument('--pretrained', required=True, type=Path)
    parser.add_argument('--dnabert2', required=True, type=Path)
    parser.add_argument('--metric', choices=['AUROC', 'AUPR', 'Accuracy'], default='AUROC', help='Metric to plot')
    parser.add_argument('--out', default='metric_onehot_vs_all_models.png')
    parser.add_argument('--figsize', default='10,8')
    parser.add_argument('--dataset-label', default='', help='Short label for dataset (e.g. Uhl or Koo) to show in legend/title')
    args = parser.parse_args()

    metric_key = args.metric
    onehot = load_and_extract(args.onehot, metric_key=metric_key)
    rand = load_and_extract(args.random, metric_key=metric_key)
    tapt = load_and_extract(args.tapt, metric_key=metric_key)
    pre = load_and_extract(args.pretrained, metric_key=metric_key)
    dnabert2 = load_and_extract(args.dnabert2, metric_key=metric_key)

    # Extract just the TF name (first part before underscore)
    def extract_tf_name(df):
        df = df.copy()
        df['TF_SHORT'] = df['TF'].astype(str).str.split('_').str[0]
        return df

    onehot = extract_tf_name(onehot)
    rand = extract_tf_name(rand)
    tapt = extract_tf_name(tapt)
    pre = extract_tf_name(pre)
    dnabert2 = extract_tf_name(dnabert2)

    def aggregate_by_tf(df):
        return (
            df.groupby('TF_SHORT', as_index=False)
            .agg(METRIC=('METRIC', 'mean'))
        )

    onehot = aggregate_by_tf(onehot)
    rand = aggregate_by_tf(rand)
    tapt = aggregate_by_tf(tapt)
    pre = aggregate_by_tf(pre)
    dnabert2 = aggregate_by_tf(dnabert2)

    # Merge on TF_SHORT (strip suffixes like _K562_200)
    dfs = [
        onehot.set_index('TF_SHORT')[['METRIC']].rename(columns={'METRIC': 'onehot'})
    ]
    dfs.append(rand.set_index('TF_SHORT')[['METRIC']].rename(columns={'METRIC': 'random'}))
    dfs.append(tapt.set_index('TF_SHORT')[['METRIC']].rename(columns={'METRIC': 'tapt'}))
    dfs.append(pre.set_index('TF_SHORT')[['METRIC']].rename(columns={'METRIC': 'pretrained'}))
    dfs.append(dnabert2.set_index('TF_SHORT')[['METRIC']].rename(columns={'METRIC': 'dnabert2'}))

    merged = pd.concat(dfs, axis=1, join='inner').reset_index().rename(columns={'TF_SHORT': 'TF'})
    merged['TF_SHORT'] = merged['TF']
    merged = merged[['TF', 'TF_SHORT', 'onehot', 'random', 'tapt', 'pretrained', 'dnabert2']]
    
    if merged.empty:
        raise SystemExit('No overlapping TFs between provided CSVs. Check TF names.')

    print(f"Found {len(merged)} overlapping TFs: {list(merged['TF_SHORT'].values)}")

    TFs = [str(x) for x in merged['TF_SHORT'].values]
    n = len(TFs)
    try:
        cmap = mpl.colormaps.get_cmap('tab20')
    except AttributeError:
        cmap = mpl.cm.get_cmap('tab20')
    palette = [cmap(i) for i in np.linspace(0, 1, max(3, n))]
    color_map = {tf: palette[i % len(palette)] for i, tf in enumerate(TFs)}

    # Plot all four model variants vs onehot on same axes
    figsize = tuple(map(float, args.figsize.split(',')))
    plt.figure(figsize=figsize)
    ax = plt.gca()

    markers = {'random': 'o', 'tapt': 's', 'pretrained': '^', 'dnabert2': 'D'}
    variants = ['random', 'tapt', 'pretrained', 'dnabert2']
    variant_labels = {'random': 'Random', 'tapt': 'TAPT', 'pretrained': 'Pretrained', 'dnabert2': 'DNABERT2'}

    for var in variants:
        for _, row in merged.iterrows():
            ax.scatter(row['onehot'], row[var], color=color_map[row['TF_SHORT']], marker=markers[var], s=70, alpha=0.85)
            # Add TF name label once (use dnabert2 points - topmost typically)
            if var == 'dnabert2':
                ax.text(row['onehot'] + 0.002, row[var], row['TF_SHORT'], fontsize=7, ha='left', va='center', alpha=0.7)

    # Make axes symmetric (same limits)
    cols = ['onehot', 'random', 'tapt', 'pretrained', 'dnabert2']
    vals = merged[cols].to_numpy(dtype=float)
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    margin = (vmax - vmin) * 0.08 if vmax > vmin else 0.05
    ax.set_xlim(vmin - margin, vmax + margin)
    ax.set_ylim(vmin - margin, vmax + margin)

    # draw y=x line (slope 1) for easy comparison
    x_vals = np.linspace(vmin - margin, vmax + margin, 100)
    ax.plot(x_vals, x_vals, linestyle='--', color='gray', linewidth=1.5, label='y = x')

    ax.set_xlabel(f'OneHot {metric_key}', fontsize=12)
    ax.set_ylabel(f'Model {metric_key}', fontsize=12)
    dataset_label = args.dataset_label.strip()
    title_suffix = f' — {dataset_label} data' if dataset_label else ''
    ax.set_title(f"{metric_key}: OneHot (x) vs All Models (y){title_suffix}", fontsize=14, fontweight='bold')

    # Legends: variant markers and TF legend
    from matplotlib.lines import Line2D
    variant_elements = [Line2D([0], [0], marker=markers[v], color='w', label=variant_labels[v], 
                               markerfacecolor='black', markeredgecolor='black', markersize=10) for v in variants]
    line_element = Line2D([0], [0], linestyle='--', color='gray', linewidth=1.5, label='y = x')
    legend1 = ax.legend(handles=variant_elements + [line_element], loc='lower right', fontsize=10, title='Models')

    # TF legend (color legend)
    if n <= 30:
        tf_elements = [Line2D([0], [0], marker='o', color='w', label=tf, 
                             markerfacecolor=color_map[tf], markeredgecolor=color_map[tf], markersize=8) for tf in TFs]
        legend2 = ax.legend(handles=tf_elements, loc='upper left', ncol=1, fontsize='small', title='RBPs')
        ax.add_artist(legend1)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = Path(args.out)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Wrote plot to {out_path}')

    # Print summary
    print(f"\n{'='*50}")
    print(f"Mean {metric_key} across {n} RBPs:")
    print(f"{'='*50}")
    print(f"  OneHot:     {merged['onehot'].mean():.4f}")
    print(f"  Random:     {merged['random'].mean():.4f}")
    print(f"  Pretrained: {merged['pretrained'].mean():.4f}")
    print(f"  TAPT:       {merged['tapt'].mean():.4f}")
    print(f"  DNABERT2:   {merged['dnabert2'].mean():.4f}")


if __name__ == '__main__':
    main()
