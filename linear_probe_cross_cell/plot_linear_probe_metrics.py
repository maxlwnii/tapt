#!/usr/bin/env python3
"""
Plot linear-probe cross-cell metrics (AUROC, AUPRC, Accuracy, F1).

Reads per_rbp_metrics.csv produced by linear_probe_cross_cell.py and
generates the same suite of plots as finetune_cross/plot_cross_cell_metrics.py.

Usage:
  python plot_linear_probe_metrics.py
  python plot_linear_probe_metrics.py \
      --results_csv ./results/<run_tag>/per_rbp_metrics.csv \
      --output_dir  ./results/<run_tag>/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ("auroc",    "auroc_std",    "AUROC"),
    ("auprc",    None,           "AUPRC"),
    ("accuracy", None,           "Accuracy"),
    ("f1",       None,           "F1"),
]

MODEL_DISPLAY_NAMES: dict[str, str] = {}


def _display(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(csv_path)

    required = {"rbp", "model", "auroc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"per_rbp_metrics.csv is missing columns: {missing}")

    # Fill absent std columns with 0
    for _, std_col, _ in METRICS:
        if std_col and std_col not in df.columns:
            df[std_col] = 0.0
        elif std_col:
            df[std_col] = df[std_col].fillna(0.0)

    # Keep only rows where all metric values are present
    metric_cols = [mc for mc, _, _ in METRICS if mc in df.columns]
    df = df.dropna(subset=metric_cols)

    model_names = sorted(df["model"].unique().tolist())

    # Keep only RBPs that have results for ALL models
    counts = df.groupby("rbp")["model"].nunique()
    complete_rbps = sorted(counts[counts == len(model_names)].index.tolist())
    df = df[df["rbp"].isin(complete_rbps)].copy()

    if df.empty:
        raise RuntimeError("No RBP has complete entries across all models.")

    return df, model_names, complete_rbps


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_bars(
    df: pd.DataFrame,
    models: list[str],
    rbps: list[str],
    metric_col: str,
    std_col: str | None,
    metric_label: str,
    output_path: Path,
) -> None:
    n_rbps   = len(rbps)
    n_models = len(models)
    x        = np.arange(n_rbps)
    width    = 0.85 / n_models

    fig_w = max(14, min(0.45 * n_rbps + 8, 70))
    fig, ax = plt.subplots(figsize=(fig_w, 8))

    for idx, model in enumerate(models):
        sub   = df[df["model"] == model].set_index("rbp").reindex(rbps)
        means = sub[metric_col].to_numpy()
        stds  = sub[std_col].to_numpy() if std_col else np.zeros(n_rbps)

        xpos = x - 0.425 + width / 2 + idx * width
        ax.bar(
            xpos, means, width=width, yerr=stds, capsize=2,
            label=_display(model), alpha=0.9, linewidth=0.6, edgecolor="black",
            error_kw={"elinewidth": 0.8, "alpha": 0.9},
        )

    ax.set_title(f"Linear probe cross-cell {metric_label} (mean ± std)", fontsize=13)
    ax.set_ylabel(metric_label)
    ax.set_xlabel("RBP task")
    ax.set_xticks(x)
    ax.set_xticklabels(rbps, rotation=90, fontsize=7)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=min(3, n_models), fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_metric_boxplot(
    df: pd.DataFrame,
    models: list[str],
    metric_col: str,
    metric_label: str,
    output_path: Path,
) -> None:
    data        = [df[df["model"] == m][metric_col].to_numpy() for m in models]
    mean_vals   = [float(np.mean(v)) if len(v) else np.nan for v in data]
    disp_models = [_display(m) for m in models]

    fig_w = max(10, 1.2 * len(models) + 6)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    bp = ax.boxplot(data, tick_labels=disp_models, patch_artist=True, showmeans=True)

    cmap = matplotlib.colormaps.get_cmap("tab10")
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(idx % cmap.N))
        patch.set_alpha(0.65)

    x_pos = np.arange(1, len(models) + 1)
    ax.scatter(x_pos, mean_vals, color="black", s=26, zorder=5, label="Mean")
    for xp, mv in zip(x_pos, mean_vals):
        if np.isnan(mv):
            continue
        ax.text(float(xp), min(mv + 0.015, 1.03), f"{mv:.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_title(f"Linear probe cross-cell {metric_label} distribution across RBPs", fontsize=13)
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Model")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_model_mean_metrics(avg_df: pd.DataFrame, output_path: Path) -> None:
    models       = avg_df["model"].tolist()
    disp_models  = [_display(m) for m in models]
    x            = np.arange(len(models))
    width        = 0.20

    specs = [
        ("auroc_mean_over_rbps",    "auroc_std_over_rbps",    "AUROC"),
        ("auprc_mean_over_rbps",    "auprc_std_over_rbps",    "AUPRC"),
        ("accuracy_mean_over_rbps", "accuracy_std_over_rbps", "Accuracy"),
        ("f1_mean_over_rbps",       "f1_std_over_rbps",       "F1"),
    ]
    # only include metrics that actually exist in avg_df
    specs = [(mc, sc, lbl) for mc, sc, lbl in specs if mc in avg_df.columns]

    n_metrics = len(specs)
    offset    = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    fig_w = max(10, 1.4 * len(models) + 6)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    cmap = matplotlib.colormaps.get_cmap("tab10")

    for idx, (mc, sc, lbl) in enumerate(specs):
        means = avg_df[mc].to_numpy()
        stds  = avg_df[sc].to_numpy() if sc in avg_df.columns else np.zeros(len(models))
        xpos  = x + offset[idx]
        bars  = ax.bar(
            xpos, means, width=width, yerr=stds, capsize=3,
            label=lbl, alpha=0.88, linewidth=0.7, edgecolor="black",
            color=cmap(idx % cmap.N), error_kw={"elinewidth": 0.9, "alpha": 0.9},
        )
        for bar, mv in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(float(mv) + 0.012, 1.03),
                f"{float(mv):.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_title("Linear probe cross-cell model comparison (mean over RBPs ± std)", fontsize=13)
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(disp_models, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot linear-probe cross-cell metrics from per_rbp_metrics.csv"
    )
    parser.add_argument(
        "--results_csv",
        default="./results/per_rbp_metrics.csv",
        help="Path to per_rbp_metrics.csv produced by linear_probe_cross_cell.py",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write plots (default: <results_csv parent>/plots)",
    )
    args = parser.parse_args()

    csv_path   = Path(args.results_csv).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else csv_path.parent / "plots"

    if not csv_path.is_file():
        raise SystemExit(f"ERROR: results_csv not found: {csv_path}")

    df, model_names, rbps = load_data(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"models     : {len(model_names)}  →  {', '.join(model_names)}")
    print(f"complete RBPs: {len(rbps)}")

    # ── per-metric bar + boxplot ──────────────────────────────────────────────
    for metric_col, std_col, metric_label in METRICS:
        if metric_col not in df.columns:
            print(f"  [skip] {metric_col} not in CSV")
            continue

        plot_metric_bars(
            df, model_names, rbps, metric_col, std_col, metric_label,
            output_dir / f"linprobe_{metric_col}.png",
        )
        plot_metric_boxplot(
            df, model_names, metric_col, metric_label,
            output_dir / f"linprobe_{metric_col}_boxplot.png",
        )
        print(f"  wrote linprobe_{metric_col}.png  +  linprobe_{metric_col}_boxplot.png")

    # ── model-average summary bar chart ──────────────────────────────────────
    avg_rows = []
    for model in model_names:
        sub = df[df["model"] == model]
        row: dict = {"model": model, "n_rbps": int(sub["rbp"].nunique())}
        for metric_col, _, _ in METRICS:
            if metric_col not in df.columns:
                continue
            row[f"{metric_col}_mean_over_rbps"] = float(sub[metric_col].mean())
            row[f"{metric_col}_std_over_rbps"]  = float(sub[metric_col].std(ddof=0))
        avg_rows.append(row)

    avg_df = pd.DataFrame(avg_rows).sort_values("model")

    avg_csv = output_dir / "linprobe_model_averages.csv"
    avg_df.to_csv(avg_csv, index=False)
    print(f"  wrote {avg_csv}")

    summary_csv = output_dir / "linprobe_complete_metrics.csv"
    df.sort_values(["rbp", "model"]).to_csv(summary_csv, index=False)
    print(f"  wrote {summary_csv}")

    plot_model_mean_metrics(avg_df, output_dir / "linprobe_model_mean_metrics.png")
    print(f"  wrote linprobe_model_mean_metrics.png")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
