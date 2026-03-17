"""
combine_results.py
------------------
Merges per_rbp_metrics.csv from every linear-probe result directory and
re-generates all plots + summary statistics + Wilcoxon tests across ALL
model variants in one place.

Result directories (hard-coded, override via --result_dirs):
  results/linear_probe                  → lamar_pretrained_L*, lamar_tapt_L*
  results/linear_probe_full             → lamar_tapt_standard_1gpu_L*, lamar_tapt_custom_1gpu_L*
  results/linear_probe_full_scratch_random → lamar_random_L*, scratch_lamar_L*

Output:
  results/combined/per_rbp_metrics.csv
  results/combined/summary_metrics.csv
  results/combined/statistical_tests.json
  results/combined/plots/auroc_boxplot.png
  results/combined/plots/per_rbp_auroc_bars.png
  results/combined/plots/delta_auroc_*.png   (one per pair)
  results/combined/plots/layer_search_*.png  (if layer-search JSONs present)

Usage:
  python combine_results.py
  python combine_results.py --output_dir results/combined_v2
  python combine_results.py --ref_model lamar_pretrained_L1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_BASE = Path(__file__).resolve().parent   # …/evalEmbeddings/

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LAMAR linear-probe results and replot")

    p.add_argument(
        "--result_dirs",
        nargs="+",
        default=[
            str(_BASE / "results" / "linear_probe"),
            str(_BASE / "results" / "linear_probe_full"),
            str(_BASE / "results" / "linear_probe_full_scratch_random"),
        ],
        help="Directories containing per_rbp_metrics.csv files to merge",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(_BASE / "results" / "combined"),
        help="Directory to write combined outputs",
    )
    p.add_argument(
        "--ref_model",
        type=str,
        default=None,
        help="Model name to use as Wilcoxon reference. Default: lowest-mean-AUROC model.",
    )
    p.add_argument(
        "--model_order",
        nargs="+",
        default=None,
        help="Explicit ordering of model names for plots (left→right). Default: alphabetical.",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--skip_plots", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading & merging
# ---------------------------------------------------------------------------

def load_and_merge(result_dirs: List[str]) -> pd.DataFrame:
    frames = []
    for d in result_dirs:
        csv = Path(d) / "per_rbp_metrics.csv"
        if not csv.exists():
            print(f"[WARN] Not found, skipping: {csv}")
            continue
        df = pd.read_csv(csv)
        print(f"  Loaded {len(df)} rows ({df['model'].nunique()} models) from {csv}")
        frames.append(df)

    if not frames:
        raise RuntimeError("No per_rbp_metrics.csv files found in the provided directories.")

    combined = pd.concat(frames, ignore_index=True)

    # Drop duplicate (rbp, model) pairs – keep first occurrence
    before = len(combined)
    combined = combined.drop_duplicates(subset=["rbp", "model"], keep="first")
    if len(combined) < before:
        print(f"[INFO] Dropped {before - len(combined)} duplicate (rbp, model) rows")

    print(f"\nCombined: {len(combined)} rows, {combined['model'].nunique()} model variants, "
          f"{combined['rbp'].nunique()} RBPs")
    return combined


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, grp in df.groupby("model"):
        rows.append({
            "Model":       model_name,
            "Mean AUROC":  grp["auroc"].mean(),
            "Std AUROC":   grp["auroc"].std(ddof=0),
            "Mean F1":     grp["f1"].mean(),
            "Std F1":      grp["f1"].std(ddof=0),
            "Mean AUPRC":  grp["auprc"].mean(),
            "Std AUPRC":   grp["auprc"].std(ddof=0),
            "Mean Acc":    grp["accuracy"].mean(),
            "N RBPs":      len(grp),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("Mean AUROC", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def statistical_tests(df: pd.DataFrame, ref_model: Optional[str] = None) -> Dict:
    pivot = df.pivot(index="rbp", columns="model", values="auroc")
    all_cols = list(pivot.columns)

    if ref_model is not None:
        if ref_model not in all_cols:
            print(f"[WARN] ref_model '{ref_model}' not in data; picking lowest-mean-AUROC model")
            ref_model = None

    if ref_model is None:
        # Use the model with the LOWEST mean AUROC as reference (random/baseline)
        means = {c: pivot[c].mean() for c in all_cols}
        ref_model = min(means, key=means.get)
        print(f"[INFO] Reference model for Wilcoxon tests: '{ref_model}' "
              f"(mean AUROC={means[ref_model]:.4f})")

    results = {}
    others = [c for c in all_cols if c != ref_model]

    for other in sorted(others):
        paired = pivot[[ref_model, other]].dropna()
        if len(paired) < 5:
            print(f"[WARN] Not enough shared RBPs for Wilcoxon {other} vs {ref_model}: {len(paired)}")
            continue
        try:
            w = wilcoxon(paired[other], paired[ref_model], alternative="two-sided")
            key = f"wilcoxon_{other}_vs_{ref_model}"
            delta = float((paired[other] - paired[ref_model]).mean())
            results[key] = {
                "statistic":       float(w.statistic),
                "pvalue":          float(w.pvalue),
                "n_rbps":          int(len(paired)),
                "mean_delta_auroc": delta,
                "reference_model": ref_model,
            }
            sig = "***" if w.pvalue < 0.001 else ("**" if w.pvalue < 0.01 else ("*" if w.pvalue < 0.05 else "ns"))
            print(f"  {other:45s} Δ={delta:+.4f}  p={w.pvalue:.4f} {sig}")
        except Exception as exc:
            print(f"[WARN] Wilcoxon {other} vs {ref_model} failed: {exc}")

    # Also do all pairwise tests and store separately
    pairwise = {}
    for i, a in enumerate(all_cols):
        for b in all_cols[i + 1:]:
            paired = pivot[[a, b]].dropna()
            if len(paired) < 5:
                continue
            try:
                w = wilcoxon(paired[a], paired[b], alternative="two-sided")
                pairwise[f"{a}_vs_{b}"] = {
                    "statistic":        float(w.statistic),
                    "pvalue":           float(w.pvalue),
                    "n_rbps":           int(len(paired)),
                    "mean_delta_auroc": float((paired[a] - paired[b]).mean()),
                }
            except Exception:
                pass

    return {"vs_reference": results, "pairwise": pairwise}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _model_order(df: pd.DataFrame, explicit: Optional[List[str]] = None) -> List[str]:
    if explicit:
        # Put any models in explicit list first (in order), others alphabetically after
        present = set(df["model"].unique())
        ordered = [m for m in explicit if m in present]
        ordered += sorted(present - set(ordered))
        return ordered
    return sorted(df["model"].unique())


def plot_boxplot(df: pd.DataFrame, out_path: Path, dpi: int, order: List[str]) -> None:
    data = [df.loc[df["model"] == m, "auroc"].dropna().values for m in order]
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.4), 5))
    import matplotlib
    _mpl_ver = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
    _bp_kwarg = "tick_labels" if _mpl_ver >= (3, 9) else "labels"
    bp = ax.boxplot(data, **{_bp_kwarg: order}, showmeans=True, patch_artist=True)
    colours = plt.cm.tab10(np.linspace(0, 1, len(order)))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor((*c[:3], 0.55))
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC distribution across RBPs — all model variants")
    ax.set_xticklabels(order, rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"  → {out_path}")


def plot_per_rbp_bars(df: pd.DataFrame, out_path: Path, dpi: int, order: List[str]) -> None:
    pivot = df.pivot(index="rbp", columns="model", values="auroc")[order]
    pivot = pivot.sort_values(order[0])
    ax = pivot.plot(kind="bar", figsize=(max(14, len(pivot) * 0.18), 6), width=0.85)
    ax.set_ylabel("AUROC")
    ax.set_title("Per-RBP AUROC — all model variants")
    ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=60, ha="right", fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"  → {out_path}")


def plot_delta(df: pd.DataFrame, ref: str, other: str, out_path: Path, dpi: int) -> None:
    pivot = df.pivot(index="rbp", columns="model", values="auroc")
    if ref not in pivot.columns or other not in pivot.columns:
        return
    delta = (pivot[other] - pivot[ref]).dropna().sort_values()
    n_pos = (delta > 0).sum()
    n_neg = (delta < 0).sum()
    fig, ax = plt.subplots(figsize=(max(12, len(delta) * 0.18), 5))
    colours = ["steelblue" if v >= 0 else "tomato" for v in delta.values]
    ax.bar(range(len(delta)), delta.values, color=colours)
    ax.set_xticks(range(len(delta)))
    ax.set_xticklabels(delta.index, rotation=60, ha="right", fontsize=6)
    ax.axhline(0.0, linestyle="--", color="black", linewidth=0.8)
    ax.set_ylabel(f"ΔAUROC ({other} − {ref})")
    ax.set_title(f"Per-RBP AUROC delta  |  ↑{n_pos} RBPs  ↓{n_neg} RBPs")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"  → {out_path}")


def plot_mean_auroc_bar(summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    """Horizontal bar chart of mean AUROC ± std for each model."""
    s = summary.sort_values("Mean AUROC")
    fig, ax = plt.subplots(figsize=(7, max(4, len(s) * 0.55)))
    y = range(len(s))
    ax.barh(y, s["Mean AUROC"], xerr=s["Std AUROC"], color="steelblue", alpha=0.75,
            ecolor="dimgray", capsize=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(s["Model"])
    ax.set_xlabel("Mean AUROC across RBPs")
    ax.set_title("Summary: mean AUROC per model variant")
    ax.axvline(0.5, linestyle="--", color="red", linewidth=0.8, label="random (0.5)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"  → {out_path}")


def plot_layer_search_curves(result_dirs: List[str], out_dir: Path, dpi: int) -> None:
    """Re-plot layer-search AUROC curves for all models found in the result dirs."""
    for d in result_dirs:
        ls_dir = Path(d) / "layer_search"
        if not ls_dir.exists():
            continue
        for js in sorted(ls_dir.glob("layer_search_*.json")):
            with open(js) as f:
                data = json.load(f)
            layer_aurocs = {int(k): v for k, v in data.get("layer_aurocs", {}).items()}
            if not layer_aurocs:
                continue
            model = data.get("model", js.stem.replace("layer_search_", ""))
            best = data.get("best_layer", max(layer_aurocs, key=layer_aurocs.get))
            layers = sorted(layer_aurocs)
            aurocs = [layer_aurocs[l] for l in layers]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(layers, aurocs, marker="o", linewidth=1.5)
            ax.axvline(best, linestyle="--", color="red", label=f"Best: layer {best}")
            ax.scatter([best], [layer_aurocs[best]], color="red", zorder=5)
            ax.set_xlabel("Layer index  (0 = token embeddings)")
            ax.set_ylabel("Mean AUROC (CV)")
            ax.set_title(f"Layer search — {model}  (pilot: {data.get('pilot_rbp', '')})")
            ax.legend()
            plt.tight_layout()
            out = out_dir / f"layer_search_{model}.png"
            plt.savefig(out, dpi=dpi)
            plt.close()
            print(f"  → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    plots_dir  = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading result files ===")
    df = load_and_merge(args.result_dirs)

    # Save merged per-RBP data
    df.to_csv(output_dir / "per_rbp_metrics.csv", index=False)
    print(f"\nSaved merged data → {output_dir / 'per_rbp_metrics.csv'}")

    # Model ordering
    order = _model_order(df, args.model_order)
    print(f"\nModel order for plots: {order}")

    # Summary
    print("\n=== Summary (mean ± std across RBPs) ===")
    summary = summarize(df)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    # Statistical tests
    print("\n=== Wilcoxon signed-rank tests (vs reference) ===")
    stat_results = statistical_tests(df, ref_model=args.ref_model)
    with open(output_dir / "statistical_tests.json", "w") as f:
        json.dump(stat_results, f, indent=2)
    print(f"Saved → {output_dir / 'statistical_tests.json'}")

    if args.skip_plots:
        print("\nPlots skipped (--skip_plots)")
        return

    print("\n=== Generating plots ===")

    # 1. Boxplot
    plot_boxplot(df, plots_dir / "auroc_boxplot.png", args.dpi, order)

    # 2. Per-RBP bar chart
    plot_per_rbp_bars(df, plots_dir / "per_rbp_auroc_bars.png", args.dpi, order)

    # 3. Mean AUROC summary bar
    plot_mean_auroc_bar(summary, plots_dir / "mean_auroc_summary.png", args.dpi)

    # 4. Delta plots: each model vs the reference (lowest AUROC = lamar_random)
    ref = stat_results["vs_reference"]
    if ref:
        ref_model = next(iter(ref.values()))["reference_model"]
        for other in order:
            if other == ref_model:
                continue
            safe = other.replace("/", "_")
            plot_delta(df, ref_model, other,
                       plots_dir / f"delta_auroc_{safe}_vs_{ref_model}.png",
                       args.dpi)

    # 5. Layer-search curves
    plot_layer_search_curves(args.result_dirs, plots_dir, args.dpi)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
