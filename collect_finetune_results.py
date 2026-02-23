"""
Collect finetuning results from all 5 model variants.

Scans the HPO-finetuning output directories, finds eval_results.json /
results.json for each RBP, and produces a consolidated summary CSV + table.

Usage:
  python collect_finetune_results.py [--thesis_root /path/to/Thesis]

Author: Maximilian Lewin
"""

import argparse
import csv
import glob
import json
import os
import sys


VARIANTS = [
    "dnabert2_pretrained",
    "dnabert2_random",
    "lamar_tapt",
    "lamar_pretrained",
    "lamar_random",
]

RBPS_KOO = [
    "HNRNPK_K562_200", "PTBP1_K562_200", "PUM2_K562_200", "QKI_K562_200",
    "RBFOX2_K562_200", "SF3B4_K562_200", "SRSF1_K562_200", "TARDBP_K562_200",
    "TIA1_K562_200", "U2AF1_K562_200",
]
RBPS_CSV = [
    "GTF2F1_K562_IDR", "HNRNPL_K562_IDR", "HNRNPM_HepG2_IDR", "ILF3_HepG2_IDR",
    "KHSRP_K562_IDR", "MATR3_K562_IDR", "PTBP1_HepG2_IDR", "QKI_K562_IDR",
]

METRICS_OF_INTEREST = [
    "eval_auc", "eval_accuracy", "eval_f1", "eval_mcc",
    "eval_loss", "eval_precision", "eval_recall",
]


def find_dnabert2_results(output_base, variant, rbp, dataset):
    """Find eval_results.json from DNABERT2 finetuning output."""
    dir_name = f"{rbp}_{dataset}"
    result_dir = os.path.join(output_base, variant, dir_name)

    # DNABERT2 saves to: output_dir/results/{run_name}/eval_results.json
    results_glob = os.path.join(result_dir, "results", "*", "eval_results.json")
    matches = glob.glob(results_glob)
    if matches:
        return matches[0]

    # Also check directly in the output dir
    direct = os.path.join(result_dir, "eval_results.json")
    if os.path.exists(direct):
        return direct

    return None


def find_lamar_results(output_base, variant, rbp, dataset):
    """Find results.json from LAMAR finetuning output."""
    dir_name = f"{rbp}_{dataset}"
    result_dir = os.path.join(output_base, variant, dir_name)

    # LAMAR saves to: output_dir/results.json
    path = os.path.join(result_dir, "results.json")
    if os.path.exists(path):
        return path

    # CV mode: check fold dirs
    fold_path = os.path.join(result_dir, "fold_0", "results.json")
    if os.path.exists(fold_path):
        return fold_path

    return None


def check_model_saved(output_base, variant, rbp, dataset):
    """Check if model weights were saved."""
    dir_name = f"{rbp}_{dataset}"
    result_dir = os.path.join(output_base, variant, dir_name)

    # DNABERT2: pytorch_model.bin or model.safetensors
    for fname in ["pytorch_model.bin", "model.safetensors"]:
        if os.path.exists(os.path.join(result_dir, fname)):
            return True

    # LAMAR CV: fold_0/pytorch_model.bin
    fold = os.path.join(result_dir, "fold_0")
    for fname in ["pytorch_model.bin", "model.safetensors"]:
        if os.path.exists(os.path.join(fold, fname)):
            return True

    # Check any safetensors/bin in subdirs
    for pattern in ["**/*.safetensors", "**/*.bin"]:
        matches = glob.glob(os.path.join(result_dir, pattern), recursive=True)
        # Exclude optimizer states
        matches = [m for m in matches if "optimizer" not in m and "training_args" not in m]
        if matches:
            return True

    return False


def load_results(path):
    """Load and normalize results from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    # LAMAR CV format: has cv_avg
    if "cv_avg" in data:
        return data["cv_avg"]

    return data


def main():
    parser = argparse.ArgumentParser(description="Collect finetuning results")
    parser.add_argument("--thesis_root", type=str,
                        default="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: thesis_root/finetune_results_summary.csv)")
    args = parser.parse_args()

    dnabert2_output = os.path.join(args.thesis_root, "DNABERT2", "hpo_finetune", "output")
    lamar_output = os.path.join(args.thesis_root, "LAMAR", "hpo_finetune", "output")

    all_rbps = [(rbp, "koo") for rbp in RBPS_KOO] + [(rbp, "csv") for rbp in RBPS_CSV]

    rows = []
    missing = []
    no_model = []

    for variant in VARIANTS:
        is_dnabert2 = variant.startswith("dnabert2")
        output_base = dnabert2_output if is_dnabert2 else lamar_output

        for rbp, dataset in all_rbps:
            if is_dnabert2:
                result_path = find_dnabert2_results(output_base, variant, rbp, dataset)
            else:
                result_path = find_lamar_results(output_base, variant, rbp, dataset)

            model_saved = check_model_saved(output_base, variant, rbp, dataset)

            if result_path is None:
                missing.append((variant, rbp, dataset))
                continue

            try:
                metrics = load_results(result_path)
            except Exception as e:
                print(f"  ERROR loading {result_path}: {e}")
                missing.append((variant, rbp, dataset))
                continue

            row = {
                "variant": variant,
                "rbp": rbp,
                "dataset": dataset,
                "model_saved": model_saved,
            }
            for m in METRICS_OF_INTEREST:
                row[m] = metrics.get(m, None)
            rows.append(row)

            if not model_saved:
                no_model.append((variant, rbp, dataset))

    # Summary
    print(f"\n{'='*80}")
    print(f"  FINETUNING RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  Total expected: {len(VARIANTS) * len(all_rbps)} ({len(VARIANTS)} variants × {len(all_rbps)} RBPs)")
    print(f"  Found:          {len(rows)}")
    print(f"  Missing:        {len(missing)}")
    print(f"  Model saved:    {sum(1 for r in rows if r['model_saved'])}")
    print(f"  No model:       {len(no_model)}")

    if missing:
        print(f"\n  MISSING RESULTS:")
        for v, r, d in missing:
            print(f"    {v} / {r}_{d}")

    if no_model:
        print(f"\n  NO MODEL WEIGHTS:")
        for v, r, d in no_model:
            print(f"    {v} / {r}_{d}")

    # Per-variant summary
    print(f"\n{'='*80}")
    print(f"  PER-VARIANT AVERAGE METRICS")
    print(f"{'='*80}")
    print(f"  {'Variant':<25s} {'AUC':>8s} {'Acc':>8s} {'F1':>8s} {'MCC':>8s} {'N':>4s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*4}")

    for variant in VARIANTS:
        v_rows = [r for r in rows if r["variant"] == variant]
        if not v_rows:
            print(f"  {variant:<25s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {0:>4d}")
            continue
        avg_auc = sum(r["eval_auc"] for r in v_rows if r["eval_auc"]) / max(1, len([r for r in v_rows if r["eval_auc"]]))
        avg_acc = sum(r["eval_accuracy"] for r in v_rows if r["eval_accuracy"]) / max(1, len([r for r in v_rows if r["eval_accuracy"]]))
        avg_f1 = sum(r["eval_f1"] for r in v_rows if r.get("eval_f1")) / max(1, len([r for r in v_rows if r.get("eval_f1")]))
        avg_mcc = sum(r["eval_mcc"] for r in v_rows if r.get("eval_mcc")) / max(1, len([r for r in v_rows if r.get("eval_mcc")]))
        print(f"  {variant:<25s} {avg_auc:>8.4f} {avg_acc:>8.4f} {avg_f1:>8.4f} {avg_mcc:>8.4f} {len(v_rows):>4d}")

    # Per-dataset summary
    for ds_label, ds_key in [("KOO (eCLIP)", "koo"), ("CSV (IDR)", "csv")]:
        print(f"\n  {ds_label}:")
        print(f"  {'Variant':<25s} {'AUC':>8s} {'Acc':>8s} {'N':>4s}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*4}")
        for variant in VARIANTS:
            v_rows = [r for r in rows if r["variant"] == variant and r["dataset"] == ds_key]
            if not v_rows:
                print(f"  {variant:<25s} {'N/A':>8s} {'N/A':>8s} {0:>4d}")
                continue
            avg_auc = sum(r["eval_auc"] for r in v_rows if r["eval_auc"]) / max(1, len([r for r in v_rows if r["eval_auc"]]))
            avg_acc = sum(r["eval_accuracy"] for r in v_rows if r["eval_accuracy"]) / max(1, len([r for r in v_rows if r["eval_accuracy"]]))
            print(f"  {variant:<25s} {avg_auc:>8.4f} {avg_acc:>8.4f} {len(v_rows):>4d}")

    # Save CSV
    output_csv = args.output or os.path.join(args.thesis_root, "finetune_results_summary.csv")
    if rows:
        fieldnames = ["variant", "rbp", "dataset", "model_saved"] + METRICS_OF_INTEREST
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Saved detailed CSV to: {output_csv}")
    else:
        print("\n  No results to save.")

    print(f"\n{'='*80}\n")

    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
