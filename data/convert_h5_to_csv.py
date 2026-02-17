"""
Convert eclip_koo h5 files to CSV format (sequence, label) for finetuning.

Reads one-hot encoded sequences from h5 files and converts them to
DNA sequence strings with binary labels.

Input:  Thesis/data/eclip_koo/*.h5
Output: Thesis/data/finetune_data_koo/<RBP>/train.csv, dev.csv, test.csv
"""

import os
import glob
import h5py
import numpy as np

H5_DIR = "/home/fr/fr_fr/fr_ml642/Thesis/data/eclip_koo"
OUT_DIR = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/data/finetune_data_koo"

BASES = ["A", "C", "G", "T"]

# Map h5 split names to CSV filenames
SPLIT_MAP = {
    "train": "train.csv",
    "valid": "dev.csv",
    "test":  "test.csv",
}


def onehot_to_dna(onehot_array):
    """Convert one-hot array (N, channels, seq_len) → list of DNA strings.
    
    Uses first 4 channels as ACGT.
    """
    # Transpose to (N, seq_len, channels)
    arr = np.transpose(onehot_array, (0, 2, 1))
    sequences = []
    for seq in arr:
        dna = "".join(BASES[np.argmax(pos[:4])] for pos in seq)
        sequences.append(dna)
    return sequences


def convert_h5(h5_path, out_dir):
    """Convert a single h5 file to train/dev/test CSVs."""
    rbp_name = os.path.basename(h5_path).replace(".h5", "")
    rbp_dir = os.path.join(out_dir, rbp_name)
    os.makedirs(rbp_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        for split, csv_name in SPLIT_MAP.items():
            X = f[f"X_{split}"][()]
            Y = f[f"Y_{split}"][()]

            sequences = onehot_to_dna(X)
            labels = Y.flatten().astype(int)

            csv_path = os.path.join(rbp_dir, csv_name)
            with open(csv_path, "w") as out:
                out.write("sequence,label\n")
                for seq, lab in zip(sequences, labels):
                    out.write(f"{seq},{lab}\n")

            print(f"  {csv_name}: {len(sequences)} samples, seq_len={len(sequences[0])}")

    return rbp_name


def main():
    h5_files = sorted(glob.glob(os.path.join(H5_DIR, "*.h5")))
    print(f"Found {len(h5_files)} h5 files in {H5_DIR}")
    print(f"Output directory: {OUT_DIR}\n")

    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    rbp_names = []
    for h5_path in h5_files:
        rbp_name = os.path.basename(h5_path).replace(".h5", "")
        rbp_dir = os.path.join(OUT_DIR, rbp_name)
        # Skip if already converted
        if os.path.exists(os.path.join(rbp_dir, "test.csv")):
            print(f"Skipping {rbp_name} — already converted")
            rbp_names.append(rbp_name)
            continue
        print(f"\nConverting: {os.path.basename(h5_path)}")
        rbp = convert_h5(h5_path, OUT_DIR)
        rbp_names.append(rbp)

    print(f"\n{'='*60}")
    print(f"Conversion complete: {len(rbp_names)} RBPs")
    print(f"RBPs: {rbp_names}")
    print(f"Output: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
