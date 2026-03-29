#!/usr/bin/env python3
"""
Plot RBP fine-tuning metric boxplots from finished runs.

This script reads:
  <results_root>/<variant>/<rbp>/results.json

and creates model-wise boxplots for:
  - AUROC
  - AUPRC
  - MCC
  - F1

It uses `cv_metrics.<metric>_mean` when available and falls back to
`test_metrics.<metric>` if CV means are missing.

Usage:
  python plot_rbp_metrics.py
  python plot_rbp_metrics.py --results_root ./results/rbp --output_dir ./results/rbp/plots
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
    ("eval_auc", "AUROC"),
    ("eval_auprc", "AUPRC"),
    ("eval_matthews_correlation", "MCC"),
    ("eval_f1", "F1"),
]

# Optional renaming of variants for plots/CSV output
VARIANT_RENAME: dict[str, str] = {
    "lamar_tapt_512": "lamar_scratch_512",
}


def _read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _extract_metric(payload: dict, metric_key: str) -> float | None:
    cv = payload.get("cv_metrics", {})
    test = payload.get("test_metrics", {})

    cv_key = f"{metric_key}_mean"
    val = cv.get(cv_key)
    if val is None:
        val = test.get(metric_key)

    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def collect_rows(results_root: Path) -> pd.DataFrame:
    rows: list[dict] = []

    variant_dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])
    for variant_dir in variant_dirs:
        variant = variant_dir.name
        for rbp_dir in sorted([d for d in variant_dir.iterdir() if d.is_dir()]):
            res_path = rbp_dir / "results.json"
            if not res_path.exists():
                continue

            payload = _read_json(res_path)
            if payload is None:
                continue

            row = {
                "variant": variant,
                "rbp": rbp_dir.name,
            }

            valid = True
            for metric_key, _ in METRICS:
                value = _extract_metric(payload, metric_key)
                if value is None:
                    valid = False
                    break
                row[metric_key] = value

            if valid:
                rows.append(row)

    if not rows:
        raise RuntimeError(f"No parseable results.json files found under: {results_root}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid metric rows found.")

    return df


def plot_metric_boxplot(df: pd.DataFrame, metric_key: str, metric_label: str, output_path: Path) -> None:
    variants = sorted(df["variant"].unique().tolist())

    # Keep only variants with at least one datapoint for the metric.
    variants = [v for v in variants if len(df[df["variant"] == v][metric_key]) > 0]
    if not variants:
        return

    data = [df[df["variant"] == variant][metric_key].to_numpy() for variant in variants]
    means = [float(np.mean(vals)) if len(vals) else np.nan for vals in data]
    counts = [int(len(vals)) for vals in data]

    labels = [f"{variant}\n(n={count})" for variant, count in zip(variants, counts)]

    fig_w = max(10, min(2.0 * len(variants) + 4, 40))
    fig, ax = plt.subplots(figsize=(fig_w, 7))

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)

    cmap = matplotlib.colormaps.get_cmap("tab20")
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(idx % cmap.N))
        patch.set_alpha(0.65)

    x_positions = np.arange(1, len(variants) + 1)
    ax.scatter(x_positions, means, color="black", s=26, zorder=5, label="Mean")
    for xpos, mean_val in zip(x_positions, means):
        if np.isnan(mean_val):
            continue
        y_text = min(mean_val + 0.015, 1.03)
        ax.text(float(xpos), y_text, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(f"RBP fine-tuning {metric_label} distribution across finished RBPs", fontsize=13)
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Model variant")
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
        description="Plot AUROC/AUPRC/MCC/F1 boxplots for finished finetune_rbp runs."
    )
    parser.add_argument(
        "--results_root",
        default="./results/rbp",
        help="Directory containing <variant>/<rbp>/results.json.",
    )
    parser.add_argument(
        "--output_dir",
        default="./results/rbp/plots",
        help="Directory where plots and CSV summaries are written.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not results_root.is_dir():
        raise SystemExit(f"ERROR: results_root does not exist: {results_root}")

    df = collect_rows(results_root)

    # Apply any variant renames so plots and summary CSVs use the desired labels
    if not df.empty and "variant" in df.columns:
        df["variant"] = df["variant"].replace(VARIANT_RENAME)

    output_dir.mkdir(parents=True, exist_ok=True)

    details_csv = output_dir / "rbp_finished_metrics_long.csv"
    df.sort_values(["variant", "rbp"]).to_csv(details_csv, index=False)

    avg_rows = []
    for variant in sorted(df["variant"].unique().tolist()):
        sub = df[df["variant"] == variant]
        row = {
            "variant": variant,
            "n_finished_rbps": int(sub["rbp"].nunique()),
        }
        for metric_key, _ in METRICS:
            row[f"{metric_key}_mean_over_finished_rbps"] = float(sub[metric_key].mean())
            row[f"{metric_key}_std_over_finished_rbps"] = float(sub[metric_key].std(ddof=0))
        avg_rows.append(row)

    avg_df = pd.DataFrame(avg_rows).sort_values("variant")
    avg_csv = output_dir / "rbp_model_averages.csv"
    avg_df.to_csv(avg_csv, index=False)

    for metric_key, metric_label in METRICS:
        out = output_dir / f"rbp_{metric_key}_boxplot.png"
        plot_metric_boxplot(df, metric_key, metric_label, out)

    print(f"finished result rows: {len(df)}")
    print(f"variants: {len(df['variant'].unique())} -> {', '.join(sorted(df['variant'].unique()))}")
    print(f"details csv: {details_csv}")
    print(f"averages csv: {avg_csv}")
    for metric_key, _ in METRICS:
        print(f"boxplot: {output_dir / f'rbp_{metric_key}_boxplot.png'}")


if __name__ == "__main__":
    main()
