#!/usr/bin/env python3
"""
Create train/dev/test CSV splits from diff-cells FASTA pairs.

Input FASTA naming convention:
  <task_name>.positives.fa
  <task_name>.negatives.fa

Output directory layout:
  <output_root>/<task_name>/{train.csv,dev.csv,test.csv}

CSV schema:
  sequence,label
where label is 1 for positives and 0 for negatives.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stratified CSV splits from diff-cells FASTA files")
    parser.add_argument(
        "--input_root",
        type=str,
        default="/home/fr/fr_fr/fr_ml642/Thesis/data/diff_cells_data",
        help="Directory containing .positives.fa / .negatives.fa files",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/fr/fr_fr/fr_ml642/Thesis/data/diff_cells_data/splits_csv",
        help="Output root where per-task split folders are created",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Total holdout fraction for dev+test",
    )
    parser.add_argument(
        "--dev_fraction_of_holdout",
        type=float,
        default=0.5,
        help="Within holdout, fraction assigned to dev (test gets the remainder)",
    )
    parser.add_argument(
        "--min_samples_per_class",
        type=int,
        default=10,
        help="Skip tasks with fewer than this many positive or negative samples",
    )
    return parser.parse_args()


def _read_fasta_sequences(path: Path) -> List[str]:
    seqs: List[str] = []
    curr: List[str] = []
    with path.open("r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if curr:
                    seqs.append("".join(curr).upper())
                    curr = []
                continue
            curr.append(line)
    if curr:
        seqs.append("".join(curr).upper())
    return seqs


def _task_stem_from_fasta_name(name: str) -> str:
    # Expects names like <stem>.positives.fa / <stem>.negatives.fa.
    if name.endswith(".positives.fa"):
        return name[: -len(".positives.fa")]
    if name.endswith(".negatives.fa"):
        return name[: -len(".negatives.fa")]
    raise ValueError(f"Unsupported FASTA name: {name}")


def _discover_pairs(input_root: Path) -> Dict[str, Tuple[Path, Path]]:
    positives: Dict[str, Path] = {}
    negatives: Dict[str, Path] = {}

    for fp in sorted(input_root.glob("*.fa")):
        if fp.name.endswith(".positives.fa"):
            positives[_task_stem_from_fasta_name(fp.name)] = fp
        elif fp.name.endswith(".negatives.fa"):
            negatives[_task_stem_from_fasta_name(fp.name)] = fp

    paired: Dict[str, Tuple[Path, Path]] = {}
    for stem, pos_fp in positives.items():
        neg_fp = negatives.get(stem)
        if neg_fp is not None:
            paired[stem] = (pos_fp, neg_fp)
    return paired


def _make_dataframe(pos: Sequence[str], neg: Sequence[str]) -> pd.DataFrame:
    seqs = list(pos) + list(neg)
    labels = np.concatenate(
        [np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)]
    )
    df = pd.DataFrame({"sequence": seqs, "label": labels})
    df["sequence"] = df["sequence"].astype(str).str.replace("U", "T").str.replace("u", "t")
    df = df.drop_duplicates(subset=["sequence", "label"]).reset_index(drop=True)
    return df


def _split_df(
    df: pd.DataFrame,
    seed: int,
    test_size: float,
    dev_fraction_of_holdout: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = df[["sequence"]]
    y = df["label"]

    x_train, x_hold, y_train, y_hold = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    x_dev, x_test, y_dev, y_test = train_test_split(
        x_hold,
        y_hold,
        test_size=(1.0 - dev_fraction_of_holdout),
        random_state=seed,
        stratify=y_hold,
    )

    train_df = pd.DataFrame({"sequence": x_train["sequence"], "label": y_train}).reset_index(drop=True)
    dev_df = pd.DataFrame({"sequence": x_dev["sequence"], "label": y_dev}).reset_index(drop=True)
    test_df = pd.DataFrame({"sequence": x_test["sequence"], "label": y_test}).reset_index(drop=True)
    return train_df, dev_df, test_df


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")

    pairs = _discover_pairs(input_root)
    if not pairs:
        raise RuntimeError(f"No positive/negative FASTA pairs found in {input_root}")

    kept = 0
    skipped = 0

    print(f"[diff-cells] Found {len(pairs)} FASTA task pairs")

    for task_name, (pos_fp, neg_fp) in sorted(pairs.items()):
        pos = _read_fasta_sequences(pos_fp)
        neg = _read_fasta_sequences(neg_fp)

        if len(pos) < args.min_samples_per_class or len(neg) < args.min_samples_per_class:
            print(
                f"  [SKIP] {task_name}: pos={len(pos)}, neg={len(neg)} < min={args.min_samples_per_class}"
            )
            skipped += 1
            continue

        df = _make_dataframe(pos, neg)
        if df["label"].nunique() < 2:
            print(f"  [SKIP] {task_name}: only one class after cleanup")
            skipped += 1
            continue

        try:
            train_df, dev_df, test_df = _split_df(
                df=df,
                seed=args.seed,
                test_size=args.test_size,
                dev_fraction_of_holdout=args.dev_fraction_of_holdout,
            )
        except ValueError as exc:
            print(f"  [SKIP] {task_name}: split failed ({exc})")
            skipped += 1
            continue

        out_dir = output_root / task_name
        out_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(out_dir / "train.csv", index=False)
        dev_df.to_csv(out_dir / "dev.csv", index=False)
        test_df.to_csv(out_dir / "test.csv", index=False)

        print(
            f"  [OK] {task_name}: train={len(train_df)} dev={len(dev_df)} test={len(test_df)} "
            f"(pos={int(df['label'].sum())}, neg={int((1 - df['label']).sum())})"
        )
        kept += 1

    print(f"[done] wrote {kept} tasks to {output_root} | skipped={skipped}")


if __name__ == "__main__":
    main()
