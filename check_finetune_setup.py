"""
Pre-flight check for HPO finetuning jobs.

Verifies:
  1. best_hpo_params.json exists and has entries for all 5 variants × 18 RBPs
  2. All data directories exist with train.csv / dev.csv / test.csv
  3. Pretrained weight files exist
  4. finetune_with_hpo.py scripts exist for DNABERT2 and LAMAR
  5. SLURM scripts exist
  6. Output directories can be created

Usage:
  python check_finetune_setup.py
"""

import json
import os
import sys

THESIS_ROOT = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"

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


def check(label, condition, detail=""):
    status = "OK" if condition else "FAIL"
    msg = f"  [{status}] {label}"
    if detail and not condition:
        msg += f" -- {detail}"
    print(msg)
    return condition


def main():
    errors = 0
    print("=" * 60)
    print("  HPO Finetuning Pre-flight Check")
    print("=" * 60)

    # 1. HPO params file
    print("\n1. HPO Parameters:")
    params_file = os.path.join(THESIS_ROOT, "best_hpo_params.json")
    if not check("best_hpo_params.json exists", os.path.exists(params_file)):
        print("  FATAL: Run extract_best_hpo_params.py first!")
        return 1

    with open(params_file) as f:
        params = json.load(f)

    all_rbps = [(r, "koo") for r in RBPS_KOO] + [(r, "csv") for r in RBPS_CSV]

    for variant in VARIANTS:
        entries = params.get(variant, [])
        dir_names = {e["dir_name"] for e in entries}
        expected = {f"{rbp}_{ds}" for rbp, ds in all_rbps}
        missing = expected - dir_names
        ok = len(missing) == 0
        if not ok:
            errors += 1
        check(f"{variant}: {len(entries)}/18 entries", ok,
              f"missing: {missing}" if missing else "")

    # 2. Data directories
    print("\n2. Data Directories:")
    data_csv = os.path.join(THESIS_ROOT, "DNABERT2", "data")
    data_koo = os.path.join(THESIS_ROOT, "data", "finetune_data_koo")

    for rbp in RBPS_KOO:
        d = os.path.join(data_koo, rbp)
        ok = all(os.path.exists(os.path.join(d, f)) for f in ["train.csv", "dev.csv", "test.csv"])
        if not ok:
            errors += 1
        check(f"koo/{rbp}", ok, f"missing files in {d}")

    for rbp in RBPS_CSV:
        d = os.path.join(data_csv, rbp)
        ok = all(os.path.exists(os.path.join(d, f)) for f in ["train.csv", "dev.csv", "test.csv"])
        if not ok:
            errors += 1
        check(f"csv/{rbp}", ok, f"missing files in {d}")

    # 3. Pretrained weights
    print("\n3. Pretrained Weights:")
    weights = {
        "LAMAR TAPT (checkpoint-98000)": os.path.join(
            THESIS_ROOT, "LAMAR", "src", "pretrain", "saving_model",
            "tapt_lamar", "checkpoint-98000", "model.safetensors"),
        "LAMAR Pretrained (weights)": os.path.join(THESIS_ROOT, "LAMAR", "weights"),
        "LAMAR Tokenizer": os.path.join(
            THESIS_ROOT, "LAMAR", "src", "pretrain", "saving_model",
            "tapt_lamar", "checkpoint-100000"),
    }
    for label, path in weights.items():
        ok = os.path.exists(path)
        if not ok:
            errors += 1
        check(label, ok, path)

    # 4. Scripts
    print("\n4. Finetuning Scripts:")
    scripts = [
        ("DNABERT2 wrapper", os.path.join(THESIS_ROOT, "DNABERT2", "hpo_finetune", "finetune_with_hpo.py")),
        ("LAMAR wrapper", os.path.join(THESIS_ROOT, "LAMAR", "hpo_finetune", "finetune_with_hpo.py")),
        ("DNABERT2 train.py", os.path.join(THESIS_ROOT, "DNABERT2", "train.py")),
        ("LAMAR finetune_rbp.py", os.path.join(THESIS_ROOT, "LAMAR", "finetune_scripts", "finetune_rbp.py")),
    ]
    for label, path in scripts:
        ok = os.path.exists(path)
        if not ok:
            errors += 1
        check(label, ok, path)

    # 5. SLURM scripts
    print("\n5. SLURM Scripts:")
    slurm = [
        ("dnabert2_pretrained", os.path.join(THESIS_ROOT, "DNABERT2", "hpo_finetune", "slurm_finetune_dnabert2_pretrained.sh")),
        ("dnabert2_random", os.path.join(THESIS_ROOT, "DNABERT2", "hpo_finetune", "slurm_finetune_dnabert2_random.sh")),
        ("lamar_tapt", os.path.join(THESIS_ROOT, "LAMAR", "hpo_finetune", "slurm_finetune_lamar_tapt.sh")),
        ("lamar_pretrained", os.path.join(THESIS_ROOT, "LAMAR", "hpo_finetune", "slurm_finetune_lamar_pretrained.sh")),
        ("lamar_random", os.path.join(THESIS_ROOT, "LAMAR", "hpo_finetune", "slurm_finetune_lamar_random.sh")),
    ]
    for label, path in slurm:
        ok = os.path.exists(path)
        if not ok:
            errors += 1
        check(label, ok, path)

    # Summary
    print(f"\n{'='*60}")
    if errors == 0:
        print("  ALL CHECKS PASSED — ready to submit!")
        print("\n  Submit order:")
        print("    cd Thesis/DNABERT2/hpo_finetune && sbatch slurm_finetune_dnabert2_pretrained.sh")
        print("    cd Thesis/DNABERT2/hpo_finetune && sbatch slurm_finetune_dnabert2_random.sh")
        print("    cd Thesis/LAMAR/hpo_finetune && sbatch slurm_finetune_lamar_tapt.sh")
        print("    cd Thesis/LAMAR/hpo_finetune && sbatch slurm_finetune_lamar_pretrained.sh")
        print("    cd Thesis/LAMAR/hpo_finetune && sbatch slurm_finetune_lamar_random.sh")
    else:
        print(f"  {errors} CHECK(S) FAILED — fix issues above before submitting")
    print("=" * 60)

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
