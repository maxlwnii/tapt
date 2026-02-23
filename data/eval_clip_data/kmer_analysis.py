"""kmer_analysis.py

Compute dinucleotide and trinucleotide frequencies for FASTA files.

Usage:
  python kmer_analysis.py --input PATH --output-dir OUTDIR

If PATH is a directory, all .fa/.fasta files inside will be processed.
"""
import argparse
import os
from collections import Counter
import itertools
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def iter_fasta(path):
    """Yield sequences (uppercased, T for U) from a fasta file."""
    seq = []
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                if seq:
                    yield ''.join(seq).upper().replace('U', 'T')
                    seq = []
            else:
                seq.append(line.strip())
        if seq:
            yield ''.join(seq).upper().replace('U', 'T')


def generate_kmers(k):
    bases = ['A', 'C', 'G', 'T']
    return [''.join(p) for p in itertools.product(bases, repeat=k)]


def count_kmers_in_sequence(seq, k):
    counts = Counter()
    n = len(seq)
    for i in range(n - k + 1):
        kmer = seq[i:i + k]
        # skip kmers with ambiguous bases
        if all(ch in 'ACGT' for ch in kmer):
            counts[kmer] += 1
    return counts


def analyze_file(path, outdir):
    dinucs = Counter()
    trinucs = Counter()
    total_di = 0
    total_tri = 0

    for seq in iter_fasta(path):
        c2 = count_kmers_in_sequence(seq, 2)
        c3 = count_kmers_in_sequence(seq, 3)
        dinucs.update(c2)
        trinucs.update(c3)
        total_di += sum(c2.values())
        total_tri += sum(c3.values())

    # ensure all possible kmers present
    all_di = generate_kmers(2)
    all_tri = generate_kmers(3)
    for k in all_di:
        dinucs.setdefault(k, 0)
    for k in all_tri:
        trinucs.setdefault(k, 0)

    # prepare output
    basename = os.path.splitext(os.path.basename(path))[0]
    txt_path = os.path.join(outdir, f"{basename}_kmer_freqs.txt")
    os.makedirs(outdir, exist_ok=True)
    with open(txt_path, 'w') as out:
        out.write(f"File: {path}\n")
        out.write(f"Total dinuc k-mers counted: {total_di}\n")
        out.write(f"Total trinuc k-mers counted: {total_tri}\n\n")
        out.write("Dinucleotide\tCount\tFrequency\n")
        for k in sorted(all_di):
            cnt = dinucs[k]
            freq = (cnt / total_di) if total_di > 0 else 0.0
            out.write(f"{k}\t{cnt}\t{freq:.6f}\n")

        out.write("\nTrinucleotide\tCount\tFrequency\n")
        for k in sorted(all_tri):
            cnt = trinucs[k]
            freq = (cnt / total_tri) if total_tri > 0 else 0.0
            out.write(f"{k}\t{cnt}\t{freq:.6f}\n")

    # plots
    # Dinucleotide bar plot (order by descending frequency left->right)
    di_items = sorted(((k, (dinucs[k] / total_di) if total_di > 0 else 0.0) for k in all_di), key=lambda x: x[1], reverse=True)
    di_keys = [k for k, _ in di_items]
    di_vals = [v for _, v in di_items]

    if HAS_MPL:
        plt.figure(figsize=(8, 4))
        plt.bar(di_keys, di_vals, color='tab:blue')
        plt.ylabel('Frequency')
        plt.xlabel('Dinucleotide')
        plt.title(f'Dinucleotide frequencies: {basename}')
        plt.tight_layout()
        di_png = os.path.join(outdir, f"{basename}_dinuc_freq.png")
        plt.savefig(di_png, dpi=200)
        plt.close()
    else:
        di_png = None

    # Trinucleotide bar plot (order by descending frequency left->right)
    tri_items = sorted(((k, (trinucs[k] / total_tri) if total_tri > 0 else 0.0) for k in all_tri), key=lambda x: x[1], reverse=True)
    tri_keys = [k for k, _ in tri_items]
    tri_vals = [v for _, v in tri_items]

    if HAS_MPL:
        plt.figure(figsize=(12, 4))
        plt.bar(tri_keys, tri_vals, color='tab:green')
        plt.ylabel('Frequency')
        plt.xlabel('Trinucleotide')
        plt.title(f'Trinucleotide frequencies: {basename}')
        plt.xticks(rotation=90, fontsize=6)
        plt.tight_layout()
        tri_png = os.path.join(outdir, f"{basename}_trinuc_freq.png")
        plt.savefig(tri_png, dpi=200)
        plt.close()
    else:
        tri_png = None

    return {'txt': txt_path, 'dinuc_png': di_png, 'trinuc_png': tri_png}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='FASTA file or directory')
    parser.add_argument('--output-dir', '-o', required=True, help='Directory to write results')
    args = parser.parse_args()

    inputs = []
    if os.path.isdir(args.input):
        for fname in os.listdir(args.input):
            if fname.endswith('.fa') or fname.endswith('.fasta'):
                inputs.append(os.path.join(args.input, fname))
    elif os.path.isfile(args.input):
        inputs.append(args.input)
    else:
        raise SystemExit(f'Input {args.input} is not a file or directory')

    results = {}
    for path in inputs:
        print(f'Analyzing {path}...')
        out = analyze_file(path, args.output_dir)
        results[path] = out
        print(f'Wrote: {out["txt"]}, plots: {out["dinuc_png"]}, {out["trinuc_png"]}')


if __name__ == '__main__':
    main()
