"""
Preprocess eval_db2.txt into 512-length non-overlapping windows.

Input:  plain-text file — one or more DNA sequences (raw nucleotides),
        lines may be wrapped at any column width and blank lines / headers
        starting with '>' are skipped.

Output: JSON array matching the training data format:
        [{"sequence": "ACGT...", "seq_id": "eval_db2:<idx>", "seq_len": 512}, ...]

Usage:
    python preprocess_val_db2.py \
        --input   /path/to/eval_db2.txt \
        --output  /path/to/val_db2_512.json \
        --window  512 \
        --max_n   0.1          # skip windows with >10% N bases
        --max_samples 0        # 0 = keep all windows

    # quick test
    python preprocess_val_db2.py --input eval_db2.txt --output /tmp/val.json --test
"""

import argparse
import json
import re
import sys


VALID_BASES = re.compile(r"^[ACGTNacgtn]+$")


def load_sequence(path: str) -> str:
    """Read a plain-text DNA file (FASTA or raw) and return one continuous string."""
    bases = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            # keep only valid nucleotide characters
            clean = re.sub(r"[^ACGTNacgtn]", "", line).upper()
            if clean:
                bases.append(clean)
    return "".join(bases)


def slide_windows(sequence: str, window: int, stride: int) -> list[dict]:
    """Slice sequence into fixed-length windows."""
    records = []
    total = len(sequence)
    idx = 0
    pos = 0
    while pos + window <= total:
        seq_win = sequence[pos : pos + window]
        records.append(
            {
                "sequence": seq_win,
                "seq_id": f"eval_db2:{idx}",
                "seq_len": window,
            }
        )
        idx += 1
        pos += stride
    return records


def filter_n(records: list[dict], max_n_frac: float) -> list[dict]:
    """Remove windows that contain more than max_n_frac N bases."""
    if max_n_frac >= 1.0:
        return records
    kept = []
    for r in records:
        n_count = r["sequence"].count("N")
        if n_count / len(r["sequence"]) <= max_n_frac:
            kept.append(r)
    return kept


def main():
    p = argparse.ArgumentParser(description="Preprocess eval_db2.txt → JSON windows")
    p.add_argument("--input",       required=True,  help="Path to eval_db2.txt")
    p.add_argument("--output",      required=True,  help="Output JSON path")
    p.add_argument("--window",      type=int, default=512, help="Window length (default 512)")
    p.add_argument("--stride",      type=int, default=0,
                   help="Stride between windows. 0 = non-overlapping (stride = window)")
    p.add_argument("--max_n",       type=float, default=0.1,
                   help="Max fraction of N bases per window (default 0.1)")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Hard cap on output samples. 0 = keep all")
    p.add_argument("--test",        action="store_true",
                   help="Print stats and first sample, do not write output file")
    args = p.parse_args()

    stride = args.stride if args.stride > 0 else args.window

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"[preprocess_val_db2] loading: {args.input}")
    seq = load_sequence(args.input)
    print(f"  total nucleotides : {len(seq):,}")

    # ── Window ────────────────────────────────────────────────────────────
    records = slide_windows(seq, args.window, stride)
    print(f"  windows (stride={stride}) : {len(records):,}")

    # ── Filter N ──────────────────────────────────────────────────────────
    before = len(records)
    records = filter_n(records, args.max_n)
    print(f"  after N-filter ({args.max_n*100:.0f}%) : {len(records):,} "
          f"(dropped {before - len(records):,})")

    # ── Cap ───────────────────────────────────────────────────────────────
    if args.max_samples > 0 and len(records) > args.max_samples:
        records = records[: args.max_samples]
        print(f"  capped at max_samples : {len(records):,}")

    if not records:
        print("ERROR: no records produced. Check your input file.", file=sys.stderr)
        sys.exit(1)

    # ── Report ────────────────────────────────────────────────────────────
    print(f"  final record count : {len(records):,}")
    print(f"  sample: {records[0]['seq_id']}  seq[:40]={records[0]['sequence'][:40]}")

    if args.test:
        print("\n[test mode] not writing output file.")
        return

    # ── Write ─────────────────────────────────────────────────────────────
    with open(args.output, "w") as fh:
        json.dump(records, fh)
    print(f"  written to: {args.output}")


if __name__ == "__main__":
    main()
