#!/usr/bin/env python3
"""
Plot cross-cell CV metrics (AUC, AUPRC, Accuracy) with std error bars.

Only RBP pair directories that contain `results.json` for ALL model folders are used.

Usage:
  python plot_cross_cell_metrics.py
  python plot_cross_cell_metrics.py --results_root ./results/cross_cell --output_dir ./results/cross_cell/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ("eval_auc", "AUC"),
    ("eval_auprc", "AUPRC"),
    ("eval_matthews_correlation", "Matthews"),
    ("eval_accuracy", "Accuracy"),
]

MODEL_DISPLAY_NAMES = {
    "lamar_tapt_512": "lamar_scratch_512",
}


def _display_model_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def _read_json(path: Path) -> dict | None:
    try:
        with path.open("r") as handle:
            return json.load(handle)
    except Exception:
        return None


def _metric_mean_std(payload: dict, metric_key: str) -> tuple[float | None, float | None]:
    cv = payload.get("cv_metrics", {})
    mean_key = f"{metric_key}_mean"
    std_key = f"{metric_key}_std"

    mean_val = cv.get(mean_key)
    std_val = cv.get(std_key)

    if mean_val is None:
        val = payload.get("val_metrics", {})
        mean_val = val.get(metric_key)
    if std_val is None:
        std_val = 0.0

    try:
        mean_val = float(mean_val) if mean_val is not None else None
    except (TypeError, ValueError):
        mean_val = None
    try:
        std_val = float(std_val) if std_val is not None else 0.0
    except (TypeError, ValueError):
        std_val = 0.0

    return mean_val, std_val


def collect_complete_rows(results_root: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    candidate_dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])
    model_dirs = []
    for d in candidate_dirs:
        has_any_results = any((child / "results.json").exists() for child in d.iterdir() if child.is_dir())
        if has_any_results:
            model_dirs.append(d)
    if not model_dirs:
        raise RuntimeError(f"No model directories with results.json found under: {results_root}")

    model_names = [d.name for d in model_dirs]

    rbps_per_model: dict[str, set[str]] = {}
    for model_dir in model_dirs:
        rbps = set()
        for rbp_dir in model_dir.iterdir():
            if not rbp_dir.is_dir():
                continue
            if (rbp_dir / "results.json").exists():
                rbps.add(rbp_dir.name)
        rbps_per_model[model_dir.name] = rbps

    common_rbps = sorted(set.intersection(*(rbps_per_model[m] for m in model_names)))
    if not common_rbps:
        raise RuntimeError("No common RBP directories with results.json across all models.")

    rows: list[dict] = []
    for rbp in common_rbps:
        for model in model_names:
            payload = _read_json(results_root / model / rbp / "results.json")
            if not payload:
                continue

            row = {"rbp": rbp, "model": model}
            valid = True
            for metric_key, _ in METRICS:
                mean_val, std_val = _metric_mean_std(payload, metric_key)
                if mean_val is None:
                    valid = False
                    break
                row[f"{metric_key}_mean"] = mean_val
                row[f"{metric_key}_std"] = std_val

            if valid:
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid metric rows found after parsing common result files.")

    counts = df.groupby("rbp")["model"].nunique()
    full_rbps = sorted(counts[counts == len(model_names)].index.tolist())
    df = df[df["rbp"].isin(full_rbps)].copy()

    if df.empty:
        raise RuntimeError("No RBP has complete metric entries across all models.")

    return df, model_names, full_rbps


def plot_metric(df: pd.DataFrame, models: list[str], rbps: list[str], metric_key: str, metric_label: str, output_path: Path) -> None:
    n_rbps = len(rbps)
    n_models = len(models)

    x = np.arange(n_rbps)
    width = 0.85 / n_models

    fig_w = max(14, min(0.45 * n_rbps + 8, 70))
    fig, ax = plt.subplots(figsize=(fig_w, 8))

    for idx, model in enumerate(models):
        sub = (
            df[df["model"] == model]
            .set_index("rbp")
            .reindex(rbps)
        )
        means = sub[f"{metric_key}_mean"].to_numpy()
        stds = sub[f"{metric_key}_std"].to_numpy()

        xpos = x - 0.425 + width / 2 + idx * width
        ax.bar(
            xpos,
            means,
            width=width,
            yerr=stds,
            capsize=2,
            label=_display_model_name(model),
            alpha=0.9,
            linewidth=0.6,
            edgecolor="black",
            error_kw={"elinewidth": 0.8, "alpha": 0.9},
        )

    ax.set_title(f"Cross-cell {metric_label} (mean ± std)", fontsize=13)
    ax.set_ylabel(metric_label)
    ax.set_xlabel("RBP pair")
    ax.set_xticks(x)
    ax.set_xticklabels(rbps, rotation=90, fontsize=7)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=min(3, n_models), fontsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_metric_boxplot(df: pd.DataFrame, models: list[str], metric_key: str, metric_label: str, output_path: Path) -> None:
    data = [
        df[df["model"] == model][f"{metric_key}_mean"].to_numpy()
        for model in models
    ]
    mean_vals = [float(np.mean(vals)) if len(vals) else np.nan for vals in data]
    display_models = [_display_model_name(model) for model in models]

    fig_w = max(10, 1.2 * len(models) + 6)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    bp = ax.boxplot(data, tick_labels=display_models, patch_artist=True, showmeans=True)

    cmap = matplotlib.colormaps.get_cmap("tab10")
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(idx % cmap.N))
        patch.set_alpha(0.65)

    x_positions = np.arange(1, len(models) + 1)
    ax.scatter(x_positions, mean_vals, color="black", s=26, zorder=5, label="Mean")
    for xpos, mean_val in zip(x_positions, mean_vals):
        if np.isnan(mean_val):
            continue
        y_text = min(mean_val + 0.015, 1.03)
        ax.text(float(xpos), y_text, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(f"Cross-cell {metric_label} distribution across RBPs", fontsize=13)
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
    models = avg_df["model"].tolist()
    display_models = [_display_model_name(model) for model in models]
    x = np.arange(len(models))
    width = 0.24

    metric_specs = [
        ("eval_auc_mean_over_rbps", "eval_auc_std_over_rbps", "AUC"),
        ("eval_auprc_mean_over_rbps", "eval_auprc_std_over_rbps", "AUPRC"),
        ("eval_accuracy_mean_over_rbps", "eval_accuracy_std_over_rbps", "Accuracy"),
    ]

    fig_w = max(10, 1.4 * len(models) + 6)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    cmap = matplotlib.colormaps.get_cmap("tab10")

    for idx, (mean_col, std_col, label) in enumerate(metric_specs):
        means = avg_df[mean_col].to_numpy()
        stds = avg_df[std_col].to_numpy()
        xpos = x + (idx - 1) * width
        bars = ax.bar(
            xpos,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=label,
            alpha=0.88,
            linewidth=0.7,
            edgecolor="black",
            color=cmap(idx % cmap.N),
            error_kw={"elinewidth": 0.9, "alpha": 0.9},
        )
        for bar, mean_val in zip(bars, means):
            y_text = min(float(mean_val) + 0.012, 1.03)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_text,
                f"{float(mean_val):.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Cross-cell model comparison (mean over RBPs ± std)", fontsize=13)
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(display_models, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cross-cell AUC/AUPRC/Accuracy for RBP pairs that are complete across all models."
    )
    parser.add_argument(
        "--results_root",
        default="./results/cross_cell",
        help="Directory containing model subfolders for cross_cell results.",
    )
    parser.add_argument(
        "--output_dir",
        default="./results/cross_cell/plots",
        help="Directory where plots and csv summary are written.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not results_root.is_dir():
        raise SystemExit(f"ERROR: results_root does not exist: {results_root}")

    df, model_names, rbps = collect_complete_rows(results_root)

    summary_csv = output_dir / "cross_cell_complete_metrics.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.sort_values(["rbp", "model"]).to_csv(summary_csv, index=False)

    avg_rows = []
    for model in model_names:
        sub = df[df["model"] == model]
        row = {"model": model, "n_rbps": int(sub["rbp"].nunique())}
        for metric_key, _ in METRICS:
            row[f"{metric_key}_mean_over_rbps"] = float(sub[f"{metric_key}_mean"].mean())
            row[f"{metric_key}_std_over_rbps"] = float(sub[f"{metric_key}_mean"].std(ddof=0))
        avg_rows.append(row)

    avg_df = pd.DataFrame(avg_rows).sort_values("model")
    avg_csv = output_dir / "cross_cell_model_averages.csv"
    avg_df.to_csv(avg_csv, index=False)

    model_compare_out = output_dir / "cross_cell_model_mean_metrics.png"
    plot_model_mean_metrics(avg_df, model_compare_out)

    for metric_key, metric_label in METRICS:
        out = output_dir / f"cross_cell_{metric_key}.png"
        plot_metric(df, model_names, rbps, metric_key, metric_label, out)

        box_out = output_dir / f"cross_cell_{metric_key}_boxplot.png"
        plot_metric_boxplot(df, model_names, metric_key, metric_label, box_out)

    print(f"models: {len(model_names)} -> {', '.join(model_names)}")
    print(f"complete rbps used: {len(rbps)}")
    print(f"summary csv: {summary_csv}")
    print(f"averages csv: {avg_csv}")
    for metric_key, _ in METRICS:
        print(f"plot: {output_dir / f'cross_cell_{metric_key}.png'}")
        print(f"boxplot: {output_dir / f'cross_cell_{metric_key}_boxplot.png'}")


if __name__ == "__main__":
    main()
