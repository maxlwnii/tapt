#!/usr/bin/env python3
"""
Plot cross-length CV mean metrics as model-wise boxplots.

Only RBP pair directories that contain results.json for all model folders are used.

Usage:
  python plot_cross_length_metrics.py
  python plot_cross_length_metrics.py --results_root ./results/cross_length --output_dir ./results/cross_length/plots
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


PLOT_METRICS = [
    ("eval_auprc", "AUPRC", "auprc"),
    ("eval_auc", "AUC", "auc"),
    ("eval_auc", "AUROC", "auroc"),
    ("eval_f1", "F1", "f1"),
    ("eval_matthews_correlation", "Matthews", "matthews"),
    ("eval_accuracy", "Accuracy", "accuracy"),
]

REQUIRED_METRICS = [
    "eval_auprc",
    "eval_auc",
    "eval_f1",
    "eval_matthews_correlation",
    "eval_accuracy",
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


def _metric_mean(payload: dict, metric_key: str) -> float | None:
    cv = payload.get("cv_metrics", {})
    mean_key = f"{metric_key}_mean"
    mean_val = cv.get(mean_key)

    if mean_val is None:
        val = payload.get("val_metrics", {})
        mean_val = val.get(metric_key)

    try:
        return float(mean_val) if mean_val is not None else None
    except (TypeError, ValueError):
        return None


def collect_complete_rows(results_root: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    candidate_dirs = sorted(d for d in results_root.iterdir() if d.is_dir())
    model_dirs = []
    for model_dir in candidate_dirs:
        has_any_results = any((child / "results.json").exists() for child in model_dir.iterdir() if child.is_dir())
        if has_any_results:
            model_dirs.append(model_dir)

    if not model_dirs:
        raise RuntimeError(f"No model directories with results.json found under: {results_root}")

    model_names = [model_dir.name for model_dir in model_dirs]

    rbps_per_model: dict[str, set[str]] = {}
    for model_dir in model_dirs:
        rbps = set()
        for rbp_dir in model_dir.iterdir():
            if rbp_dir.is_dir() and (rbp_dir / "results.json").exists():
                rbps.add(rbp_dir.name)
        rbps_per_model[model_dir.name] = rbps

    common_rbps = sorted(set.intersection(*(rbps_per_model[model_name] for model_name in model_names)))
    if not common_rbps:
        raise RuntimeError("No common RBP directories with results.json across all models.")

    rows: list[dict[str, float | str]] = []
    for rbp in common_rbps:
        for model_name in model_names:
            payload = _read_json(results_root / model_name / rbp / "results.json")
            if not payload:
                continue

            row: dict[str, float | str] = {"rbp": rbp, "model": model_name}
            valid = True
            for metric_key in REQUIRED_METRICS:
                mean_val = _metric_mean(payload, metric_key)
                if mean_val is None:
                    valid = False
                    break
                row[f"{metric_key}_mean"] = mean_val

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


def plot_metric_boxplot(
    df: pd.DataFrame,
    models: list[str],
    metric_key: str,
    metric_label: str,
    output_path: Path,
) -> None:
    data = [
        df[df["model"] == model][f"{metric_key}_mean"].to_numpy()
        for model in models
    ]
    mean_vals = [float(np.mean(values)) if len(values) else np.nan for values in data]
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

    ax.set_title(f"Cross-length {metric_label} distribution across RBPs", fontsize=13)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cross-length mean CV metrics as boxplots for RBP pairs complete across all models."
    )
    parser.add_argument(
        "--results_root",
        default="./results/cross_length",
        help="Directory containing model subfolders for cross_length results.",
    )
    parser.add_argument(
        "--output_dir",
        default="./results/cross_length/plots",
        help="Directory where plots and csv summary are written.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not results_root.is_dir():
        raise SystemExit(f"ERROR: results_root does not exist: {results_root}")

    df, model_names, rbps = collect_complete_rows(results_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "cross_length_complete_metrics.csv"
    df.sort_values(["rbp", "model"]).to_csv(summary_csv, index=False)

    avg_rows = []
    for model_name in model_names:
        sub = df[df["model"] == model_name]
        row: dict[str, float | int | str] = {
            "model": model_name,
            "n_rbps": int(sub["rbp"].nunique()),
        }
        for metric_key in REQUIRED_METRICS:
            row[f"{metric_key}_mean_over_rbps"] = float(sub[f"{metric_key}_mean"].mean())
            row[f"{metric_key}_std_over_rbps"] = float(sub[f"{metric_key}_mean"].std(ddof=0))
        avg_rows.append(row)

    avg_df = pd.DataFrame(avg_rows).sort_values("model")
    avg_csv = output_dir / "cross_length_model_averages.csv"
    avg_df.to_csv(avg_csv, index=False)

    for metric_key, metric_label, metric_slug in PLOT_METRICS:
        output_path = output_dir / f"cross_length_{metric_slug}_boxplot.png"
        plot_metric_boxplot(df, model_names, metric_key, metric_label, output_path)

    print(f"models: {len(model_names)} -> {', '.join(model_names)}")
    print(f"complete rbps used: {len(rbps)}")
    print(f"summary csv: {summary_csv}")
    print(f"averages csv: {avg_csv}")
    for _, _, metric_slug in PLOT_METRICS:
        print(f"boxplot: {output_dir / f'cross_length_{metric_slug}_boxplot.png'}")


if __name__ == "__main__":
    main()