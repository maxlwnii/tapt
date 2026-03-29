"""
Plot linear probe results for selected DNABERT2 variants and one-hot baseline.

Creates:
- Boxplots (distribution across RBPs) for metrics (AUROC, AUPRC, F1, Accuracy)
  across selected model variants. Mean shown as diamond and std annotated.
- Dot (strip) plots per task group (`data`, `finetune_data_koo`, `splits_csv`) showing
  per-RBP values for each variant.

Usage: run from repository root using the project's .venv, e.g.
  source .venv/bin/activate
  python linear_probe/plots/plot_linear_probe_results.py

Outputs are written to `linear_probe/plots`.
"""

import re
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="darkgrid")
palette = sns.color_palette("Set2")
MIN_TRAIN_SAMPLES = 500


def strip_timestamp(name: str) -> str:
    # Remove trailing date/time/id suffixes that start with an 8-digit date (YYYYMMDD)
    # e.g. dnabert2_random_6_20260327_172328_12320466 -> dnabert2_random_6
    m = re.sub(r'_[0-9]{8}.*$', '', name)
    return m


def find_result_dirs(results_dir: Path):
    # Collect: all dnabert2 last variants, all dnabert2 layer-6 variants, and one_hot dir
    # Collect: all LAMAR and DNABERT2 last variants, their layer-6 variants, and one_hot dir
    lamar_last = []
    lamar_layer6 = []
    dnabert2_last = []
    dnabert2_layer6 = []
    one_hot = None

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        n = d.name
        if n.startswith("one_hot"):
            one_hot = d

        if n.startswith("dnabert2"):
            if "layer6" in n or "_layer6" in n or "_6_" in n:
                dnabert2_layer6.append(d)
            else:
                dnabert2_last.append(d)

        if n.startswith("lamar"):
            if "layer6" in n or "_layer6" in n or "_6_" in n:
                lamar_layer6.append(d)
            else:
                lamar_last.append(d)

    # prefer last variants; also include layer6 variants to compare (if present)
    variants = lamar_last + dnabert2_last + lamar_layer6 + dnabert2_layer6
    if one_hot is not None:
        variants.append(one_hot)

    return variants


def load_per_rbp_csv(dirpath: Path, mode: str = "both") -> pd.DataFrame:
    """Load per_rbp_metrics.csv from `dirpath`.

    mode: 'layer6' -> keep only layer-6 rows (or rows whose layer_name contains '6')
          'last'   -> keep rows that are NOT layer-6 (plus one_hot always)
          'both'   -> keep all rows
    """
    f = dirpath / "per_rbp_metrics.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df["source_dir"] = dirpath.name
    df["variant"] = strip_timestamp(dirpath.name)
    # derive top-level task group from task_id prefix before '/'
    df["task_group"] = df["task_id"].astype(str).apply(lambda s: s.split("/")[0])
    # Determine layer-6 mask
    if "layer_index" in df.columns:
        df["layer_index"] = pd.to_numeric(df["layer_index"], errors="coerce")
        mask_idx = df["layer_index"] == 6
    else:
        mask_idx = pd.Series(False, index=df.index)

    df["layer_name"] = df.get("layer_name", "").astype(str)
    mask_name = df["layer_name"].str.contains("6", na=False)

    # Always include one_hot variant rows
    mask_onehot = df["variant"].astype(str).str.startswith("one_hot")

    if mode == "layer6":
        df = df[mask_idx | mask_name | mask_onehot].reset_index(drop=True)
    elif mode == "last":
        # keep rows that are not flagged as layer6, but keep one_hot
        df = df[(~(mask_idx | mask_name)) | mask_onehot].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def build_low_train_task_ids(thesis_root: Path, min_train_samples: int = MIN_TRAIN_SAMPLES) -> set:
    """Return task_id values to exclude for diff-cells tasks with too few training rows."""
    splits_root = thesis_root / "data" / "diff_cells_data" / "splits_csv"
    low_ids = set()
    if not splits_root.exists():
        print(f"[WARN] diff-cells root not found: {splits_root}")
        return low_ids

    for d in sorted(splits_root.iterdir()):
        train_csv = d / "train.csv"
        if not d.is_dir() or not train_csv.exists():
            continue
        # Count data rows (exclude header)
        n_train = sum(1 for _ in open(train_csv, "r")) - 1
        if n_train < min_train_samples:
            low_ids.add(f"splits_csv/{d.name}")

    print(f"[filter] low-train diff-cells tasks (<{min_train_samples}): {len(low_ids)}")
    return low_ids


def apply_low_train_filter(df: pd.DataFrame, low_task_ids: set) -> pd.DataFrame:
    """Remove rows whose task_id belongs to low-train diff-cells tasks."""
    if not low_task_ids or df.empty:
        return df

    if "task_id" in df.columns:
        mask_drop = df["task_id"].astype(str).isin(low_task_ids)
    else:
        # Fallback path if task_id is unavailable
        task_group = df.get("task_group", "").astype(str)
        rbp = df.get("rbp", "").astype(str)
        mask_drop = (task_group == "splits_csv") & ("splits_csv/" + rbp).isin(low_task_ids)

    removed = int(mask_drop.sum())
    if removed > 0:
        print(f"[filter] removed {removed} rows with low-train diff-cells tasks")
    return df.loc[~mask_drop].reset_index(drop=True)


def build_max_train_task_ids(thesis_root: Path, max_train_samples: int = MIN_TRAIN_SAMPLES) -> set:
    """Return task_id values for diff-cells tasks with train rows <= max_train_samples."""
    splits_root = thesis_root / "data" / "diff_cells_data" / "splits_csv"
    max_ids = set()
    if not splits_root.exists():
        print(f"[WARN] diff-cells root not found: {splits_root}")
        return max_ids

    for d in sorted(splits_root.iterdir()):
        train_csv = d / "train.csv"
        if not d.is_dir() or not train_csv.exists():
            continue
        # Count data rows (exclude header)
        n_train = sum(1 for _ in open(train_csv, "r")) - 1
        if n_train <= max_train_samples:
            max_ids.add(f"splits_csv/{d.name}")

    print(f"[filter] diff-cells tasks with train <= {max_train_samples}: {len(max_ids)}")
    return max_ids


def apply_max_train_filter(df: pd.DataFrame, max_task_ids: set, max_train_samples: int = MIN_TRAIN_SAMPLES) -> pd.DataFrame:
    """Keep only rows whose task_id belongs to diff-cells tasks with train rows <= max_train_samples."""
    if not max_task_ids or df.empty:
        return df

    if "task_id" in df.columns:
        mask_keep = df["task_id"].astype(str).isin(max_task_ids)
    else:
        task_group = df.get("task_group", "").astype(str)
        rbp = df.get("rbp", "").astype(str)
        mask_keep = (task_group == "splits_csv") & ("splits_csv/" + rbp).isin(max_task_ids)

    kept = int(mask_keep.sum())
    if kept > 0:
        print(f"[filter] kept {kept} rows with train <= {max_train_samples} (max) for diff-cells tasks")
    return df.loc[mask_keep].reset_index(drop=True)


def plot_boxplots(df: pd.DataFrame, outdir: Path, metrics: list, file_suffix: str = ""):
    # boxplot per metric across variants (distribution over RBPs)
    variants = sorted(df["variant"].unique())

    for metric in metrics:
        # Use matplotlib boxplot to avoid seaborn legend/palette issues
        plt.figure(figsize=(14, 7))
        ax = plt.gca()

        # prepare data per variant
        data_by_variant = [df[df["variant"] == v][metric].dropna().values for v in variants]

        # draw boxplot
        bp = ax.boxplot(data_by_variant, labels=variants, patch_artist=True)

        # color boxes
        pal = sns.color_palette("Set2", n_colors=max(1, len(variants)))
        for patch, color in zip(bp['boxes'], pal):
            patch.set_facecolor(color)

        # overlay means and annotate std
        means = [np.nan if len(arr) == 0 else np.nanmean(arr) for arr in data_by_variant]
        stds = [np.nan if len(arr) == 0 else np.nanstd(arr) for arr in data_by_variant]
        positions = range(1, len(variants) + 1)
        ax.scatter(positions, means, color='red', s=90, zorder=5, marker='D', edgecolor='darkred')

        for pos, m, sdev in zip(positions, means, stds):
            if np.isfinite(m):
                ax.text(pos, m + 0.005, f"{m:.3f}\n±{sdev:.3f}", ha='center', va='bottom', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

        ax.set_title(f"{metric.upper()} distribution across variants (per-RBP)")
        ax.set_xlabel("Variant")
        ax.set_ylabel(metric.upper())
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        suffix = f"_{file_suffix}" if file_suffix else ""
        outpath = outdir / f"boxplot_{metric}{suffix}.png"
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved:", outpath)


def plot_dotplots_by_task(df: pd.DataFrame, outdir: Path, metrics: list, file_suffix: str = ""):
    # For each top-level task group create dot plots showing per-RBP values across variants.
    groups = ["data", "finetune_data_koo", "splits_csv"]

    for grp in groups:
        sub = df[df["task_group"] == grp]
        if sub.empty:
            print(f"No data for group {grp}, skipping")
            continue

        for metric in metrics:
            plt.figure(figsize=(14, 8))
            # Use seaborn stripplot for dot plot
            pal = sns.color_palette("Set2", n_colors=max(1, sub["variant"].nunique()))
            ax = sns.stripplot(x="variant", y=metric, data=sub, jitter=True, palette=pal, size=6, alpha=0.8)

            # overlay mean and std as errorbars
            stats = sub.groupby("variant")[metric].agg(["mean", "std"]).reindex(sorted(sub["variant"].unique()))
            xs = range(len(stats))
            ax.errorbar(xs, stats["mean"].values, yerr=stats["std"].values, fmt='o', color='black',
                        markersize=8, capsize=5, label='mean ± std')

            ax.set_title(f"Per-RBP {metric.upper()} — group: {grp}")
            ax.set_xlabel("Variant")
            ax.set_ylabel(metric.upper())
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            suffix = f"_{file_suffix}" if file_suffix else ""
            outpath = outdir / f"dotplot_{grp}_{metric}{suffix}.png"
            plt.savefig(outpath, dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved:", outpath)


def main():
    base = Path(__file__).resolve().parents[1]
    thesis_root = base.parent
    results_dir = base / "results"
    outdir = base / "plots_new"
    ensure_outdir(outdir)
    low_task_ids = build_low_train_task_ids(thesis_root, MIN_TRAIN_SAMPLES)

    metrics = ["auprc", "auroc", "f1", "accuracy"]

    variants_dirs = find_result_dirs(results_dir)
    if not variants_dirs:
        print("No variant directories found under", results_dir)
        return

    # --- Layer-6 plots ---
    frames_l6 = []
    for d in variants_dirs:
        df = load_per_rbp_csv(d, mode="layer6")
        if not df.empty:
            df = apply_low_train_filter(df, low_task_ids)
            frames_l6.append(df)

    if frames_l6:
        df_l6 = pd.concat(frames_l6, ignore_index=True)
        available_metrics_l6 = [m for m in metrics if m in df_l6.columns]
        if available_metrics_l6:
            plot_boxplots(df_l6, outdir, available_metrics_l6, file_suffix="layer6_mintrain500")
            plot_dotplots_by_task(df_l6, outdir, available_metrics_l6, file_suffix="layer6_mintrain500")
            print("Layer-6 plots written to:", outdir)
    else:
        print("No layer-6 per_rbp_metrics.csv files found in selected directories.")

    # --- Layer-6 plots (only tasks with train rows <= MIN_TRAIN_SAMPLES) ---
    max_task_ids = build_max_train_task_ids(thesis_root, MIN_TRAIN_SAMPLES)
    frames_l6_max = []
    for d in variants_dirs:
        df = load_per_rbp_csv(d, mode="layer6")
        if not df.empty:
            df = apply_max_train_filter(df, max_task_ids, MIN_TRAIN_SAMPLES)
            if not df.empty:
                frames_l6_max.append(df)

    if frames_l6_max:
        df_l6_max = pd.concat(frames_l6_max, ignore_index=True)
        available_metrics_l6_max = [m for m in metrics if m in df_l6_max.columns]
        if available_metrics_l6_max:
            plot_boxplots(df_l6_max, outdir, available_metrics_l6_max, file_suffix="layer6_maxtrain500")
            plot_dotplots_by_task(df_l6_max, outdir, available_metrics_l6_max, file_suffix="layer6_maxtrain500")
            print("Layer-6 (max-train=500) plots written to:", outdir)
    else:
        print("No layer-6 per_rbp_metrics.csv files found for max-train filter.")

    # --- Last-layer plots ---
    frames_last = []
    for d in variants_dirs:
        df = load_per_rbp_csv(d, mode="last")
        if not df.empty:
            df = apply_low_train_filter(df, low_task_ids)
            frames_last.append(df)

    if frames_last:
        df_last = pd.concat(frames_last, ignore_index=True)
        available_metrics_last = [m for m in metrics if m in df_last.columns]
        if available_metrics_last:
            plot_boxplots(df_last, outdir, available_metrics_last, file_suffix="last_mintrain500")
            plot_dotplots_by_task(df_last, outdir, available_metrics_last, file_suffix="last_mintrain500")
            print("Last-layer plots written to:", outdir)
    else:
        print("No last-layer per_rbp_metrics.csv files found in selected directories.")

    # --- Last-layer plots (only tasks with train rows <= MIN_TRAIN_SAMPLES) ---
    frames_last_max = []
    for d in variants_dirs:
        df = load_per_rbp_csv(d, mode="last")
        if not df.empty:
            df = apply_max_train_filter(df, max_task_ids, MIN_TRAIN_SAMPLES)
            if not df.empty:
                frames_last_max.append(df)

    if frames_last_max:
        df_last_max = pd.concat(frames_last_max, ignore_index=True)
        available_metrics_last_max = [m for m in metrics if m in df_last_max.columns]
        if available_metrics_last_max:
            plot_boxplots(df_last_max, outdir, available_metrics_last_max, file_suffix="last_maxtrain500")
            plot_dotplots_by_task(df_last_max, outdir, available_metrics_last_max, file_suffix="last_maxtrain500")
            print("Last-layer (max-train=500) plots written to:", outdir)
    else:
        print("No last-layer per_rbp_metrics.csv files found for max-train filter.")

    print("All requested plots written to:", outdir)


if __name__ == "__main__":
    main()
