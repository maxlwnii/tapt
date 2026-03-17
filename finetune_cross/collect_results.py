#!/usr/bin/env python3
"""
Collect all results.json files from finetune_cross runs into a single CSV.

Usage:
  python collect_results.py                          # scan ./results/
  python collect_results.py --results_dir /path/to/results --output summary.csv

Output columns:
  experiment, model_type, variant, pair_name, max_length,
  val_auc, val_auprc, val_accuracy, val_f1, val_mcc,
  test_auc, test_auprc, test_accuracy, test_f1, test_mcc
"""

import argparse
import csv
import json
import os
import sys


def collect(results_dir):
    rows = []
    for root, dirs, files in os.walk(results_dir):
        if "results.json" in files:
            path = os.path.join(root, "results.json")
            try:
                with open(path) as f:
                    d = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"WARN: skipping {path}: {e}", file=sys.stderr)
                continue

            val = d.get("val_metrics", {})
            test = d.get("test_metrics", {})

            rows.append({
                "experiment": d.get("experiment", ""),
                "model_type": d.get("model_type", ""),
                "variant": d.get("variant", ""),
                "pair_name": d.get("pair_name", ""),
                "max_length": d.get("max_length", ""),
                "val_auc": val.get("eval_auc", ""),
                "val_auprc": val.get("eval_auprc", ""),
                "val_accuracy": val.get("eval_accuracy", ""),
                "val_f1": val.get("eval_f1", ""),
                "val_mcc": val.get("eval_matthews_correlation", ""),
                "test_auc": test.get("eval_auc", ""),
                "test_auprc": test.get("eval_auprc", ""),
                "test_accuracy": test.get("eval_accuracy", ""),
                "test_f1": test.get("eval_f1", ""),
                "test_mcc": test.get("eval_matthews_correlation", ""),
            })

    rows.sort(key=lambda r: (r["experiment"], r["model_type"], r["variant"], r["pair_name"]))
    return rows


def main():
    p = argparse.ArgumentParser(description="Collect finetune_cross results into CSV")
    p.add_argument("--results_dir", default="./results",
                   help="Root directory containing results (default: ./results)")
    p.add_argument("--output", default="results_summary.csv",
                   help="Output CSV path (default: results_summary.csv)")
    args = p.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: {args.results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    rows = collect(args.results_dir)
    if not rows:
        print("No results.json files found.", file=sys.stderr)
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Collected {len(rows)} results → {args.output}")

    # Print summary by variant
    from collections import defaultdict
    summary = defaultdict(list)
    for r in rows:
        key = (r["experiment"], r["model_type"], r["variant"])
        if r["test_auc"]:
            summary[key].append(float(r["test_auc"]))

    if summary:
        print("\n  Experiment            Model     Variant       N   mean_AUC  std_AUC")
        print("  " + "─" * 72)
        for (exp, model, var), aucs in sorted(summary.items()):
            import numpy as np
            arr = np.array(aucs)
            print(f"  {exp:<22s} {model:<9s} {var:<13s} {len(arr):3d}   "
                  f"{arr.mean():.4f}    {arr.std():.4f}")


if __name__ == "__main__":
    main()
