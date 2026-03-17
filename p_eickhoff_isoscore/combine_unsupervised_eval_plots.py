#!/usr/bin/env python3
"""Create one combined plot from unsupervised_eval and unsupervised_eval2 metrics.

Reads:
  - results/unsupervised_eval/metrics.csv
  - results/unsupervised_eval2/metrics.csv

Writes:
  - results/unsupervised_eval/plots/combined_unsupervised_eval_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["IsoScore", "RankMe", "NESum", "StableRank"]

MODEL_DISPLAY = {
    "lamar_tapt_512": "lamar_scratch_512",
}


def _display_model_name(model_name: str) -> str:
    return MODEL_DISPLAY.get(model_name, model_name)


def _load_and_aggregate(metrics_csv: Path, run_label: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    required = {"model", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {metrics_csv}: {sorted(missing)}")

    agg = df.groupby("model", as_index=False).agg(
        IsoScore_mean=("IsoScore", "mean"),
        IsoScore_std=("IsoScore", "std"),
        RankMe_mean=("RankMe", "mean"),
        RankMe_std=("RankMe", "std"),
        NESum_mean=("NESum", "mean"),
        NESum_std=("NESum", "std"),
        StableRank_mean=("StableRank", "mean"),
        StableRank_std=("StableRank", "std"),
    )
    agg["run"] = run_label
    return agg


def make_combined_plot(run1_csv: Path, run2_csv: Path, output_png: Path) -> None:
    agg1 = _load_and_aggregate(run1_csv, "unsupervised_eval")
    agg2 = _load_and_aggregate(run2_csv, "unsupervised_eval2")
    combined = pd.concat([agg1, agg2], ignore_index=True)

    models = sorted(combined["model"].unique().tolist())
    runs = ["unsupervised_eval", "unsupervised_eval2"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()

    x = np.arange(len(models))
    width = 0.36
    run_offsets = {
        "unsupervised_eval": -width / 2,
        "unsupervised_eval2": width / 2,
    }

    colors = {
        "unsupervised_eval": "#4C78A8",
        "unsupervised_eval2": "#F58518",
    }

    for ax, metric in zip(axes, METRICS):
        for run in runs:
            run_df = combined[combined["run"] == run].set_index("model")
            means = []
            stds = []
            for model in models:
                if model in run_df.index:
                    mean_val = run_df.at[model, f"{metric}_mean"]
                    std_val = run_df.at[model, f"{metric}_std"]
                    means.append(float(np.asarray(mean_val, dtype=np.float64).item()))
                    stds.append(float(np.asarray(std_val, dtype=np.float64).item()))
                else:
                    means.append(np.nan)
                    stds.append(0.0)

            xpos = x + run_offsets[run]
            ax.bar(
                xpos,
                means,
                width=width,
                yerr=stds,
                capsize=3,
                label=run,
                color=colors[run],
                alpha=0.85,
                edgecolor="black",
                linewidth=0.7,
                error_kw={"elinewidth": 0.9, "alpha": 0.9},
            )

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels([_display_model_name(m) for m in models], rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if metric == "IsoScore":
            ax.set_ylim(0.0, 0.08)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Combined Unsupervised Eval Comparison (mean ± std over tasks)", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine unsupervised_eval and unsupervised_eval2 into one plot.")
    parser.add_argument(
        "--run1_csv",
        type=Path,
        default=Path("./results/unsupervised_eval/metrics.csv"),
        help="Path to unsupervised_eval metrics.csv",
    )
    parser.add_argument(
        "--run2_csv",
        type=Path,
        default=Path("./results/unsupervised_eval2/metrics.csv"),
        help="Path to unsupervised_eval2 metrics.csv",
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=Path("./results/unsupervised_eval/plots/combined_unsupervised_eval_comparison.png"),
        help="Output combined figure path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_combined_plot(args.run1_csv, args.run2_csv, args.output_png)
    print(f"saved: {args.output_png.resolve()}")


if __name__ == "__main__":
    main()
