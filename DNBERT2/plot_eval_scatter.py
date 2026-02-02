#!/usr/bin/env python3
import os
import json
import glob
import matplotlib.pyplot as plt

ROOT = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(ROOT, "output")

def find_eval_files(root):
    pattern = os.path.join(root, "**", "eval_results.json")
    return glob.glob(pattern, recursive=True)

def parse_results(paths):
    normal_auc = {}
    random_auc = {}
    normal_auprc = {}
    random_auprc = {}

    for p in paths:
        try:
            with open(p, 'r') as f:
                data = json.load(f)
        except Exception:
            continue

        auc = data.get('eval_auc') or data.get('eval_roc_auc') or data.get('auc')
        auprc = data.get('eval_auprc') or data.get('auprc')

        # Determine model dir part containing 'dnabert2_'
        parts = p.split(os.sep)
        model_dir = None
        for part in parts:
            if part.startswith('dnabert2_'):
                model_dir = part
                break
        if model_dir is None:
            continue

        is_random = 'random' in model_dir
        rbp = model_dir.replace('dnabert2_', '').split('__fold5')[0]

        if auc is not None:
            if is_random:
                random_auc[rbp] = float(auc)
            else:
                normal_auc[rbp] = float(auc)

        if auprc is not None:
            if is_random:
                random_auprc[rbp] = float(auprc)
            else:
                normal_auprc[rbp] = float(auprc)

    return normal_auc, random_auc, normal_auprc, random_auprc

def plot_scatter(metric_name, normal, random, outpath):
    keys = sorted(set(normal.keys()) & set(random.keys()))
    if not keys:
        raise SystemExit(f'No matching RBPs found between normal and random models for {metric_name}.')

    xs = [random[k] for k in keys]
    ys = [normal[k] for k in keys]

    # dynamic axis limits with small margin
    all_vals = xs + ys
    vmin = min(all_vals)
    vmax = max(all_vals)
    margin = max(0.02, (vmax - vmin) * 0.05)
    xmin = max(0.0, vmin - margin)
    xmax = min(1.0, vmax + margin)

    cmap = plt.get_cmap('tab20')
    n = len(keys)
    colors = [cmap(i % cmap.N) for i in range(n)]

    plt.figure(figsize=(6,6))
    for i,k in enumerate(keys):
        plt.scatter(random[k], normal[k], s=100, color=colors[i], label=k if i < 20 else None)
        plt.text(random[k], normal[k], k.split('_')[0], fontsize=8, ha='right', va='bottom')

    plt.plot([0.84,xmax],[0.84,xmax], linestyle='--', color='gray', linewidth=1)
    plt.xlabel(f'Random model {metric_name.upper()}')
    plt.ylabel(f'Normal model {metric_name.upper()}')
    plt.title(f'DNABERT2: Normal vs Random {metric_name.upper()} per RBP')
    plt.xlim(xmin, xmax)
    plt.ylim(xmin, xmax)
    plt.grid(alpha=0.3)
    # add legend if many RBPs (avoid duplicate labels)
    if n <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    print(f'Plot saved to: {outpath}')
    plt.close()

def main():
    eval_files = find_eval_files(OUTPUT_DIR)
    normal_auc, random_auc, normal_auprc, random_auprc = parse_results(eval_files)

    out_auc = os.path.join(ROOT, 'output', 'auc_scatter_normal_vs_random.png')
    out_auprc = os.path.join(ROOT, 'output', 'auprc_scatter_normal_vs_random.png')

    # Plot AUC
    try:
        plot_scatter('auc', normal_auc, random_auc, out_auc)
    except SystemExit as e:
        print(e)

    # Plot AUPRC
    try:
        plot_scatter('auprc', normal_auprc, random_auprc, out_auprc)
    except SystemExit as e:
        print(e)

if __name__ == '__main__':
    main()
