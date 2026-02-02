import os
import glob
from collections import Counter

def parse_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        seq = ''
        for line in f:
            if line.startswith('>'):
                if seq:
                    sequences.append(seq.upper())
                    seq = ''
            else:
                seq += line.strip()
        if seq:
            sequences.append(seq.upper())
    return sequences

def compute_gc_au(sequences):
    total_bases = 0
    gc_count = 0
    au_count = 0
    for seq in sequences:
        counts = Counter(seq)
        total_bases += len(seq)
        gc_count += counts.get('G', 0) + counts.get('C', 0)
        au_count += counts.get('A', 0) + counts.get('U', 0)
    if total_bases == 0:
        return 0, 0, 0, 0
    gc_percent = (gc_count / total_bases) * 100
    au_percent = (au_count / total_bases) * 100
    return gc_percent, au_percent, len(sequences), total_bases

# Directory paths
data_dir = '/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/data/clip_training_data'
output_dir = '/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/data/eval_clip_data'

# Find all .fa files
fa_files = glob.glob(os.path.join(data_dir, '*.fa'))

# Dictionary to hold results
results = {}

for file_path in fa_files:
    filename = os.path.basename(file_path)
    # Parse filename: e.g., GTF2F1_K562_IDR.positives.fa
    parts = filename.replace('.fa', '').split('.')
    rbp_cell = parts[0]  # GTF2F1_K562_IDR
    label = parts[1]  # positives or negatives
    
    sequences = parse_fasta(file_path)
    gc, au, num_seq, total_bases = compute_gc_au(sequences)
    
    if rbp_cell not in results:
        results[rbp_cell] = {}
    results[rbp_cell][label] = {
        'GC%': gc,
        'AU%': au,
        'Num_sequences': num_seq,
        'Total_bases': total_bases
    }

# Now, compute differences
diff_results = []
for rbp, data in results.items():
    if 'positives' in data and 'negatives' in data:
        pos_gc = data['positives']['GC%']
        neg_gc = data['negatives']['GC%']
        pos_au = data['positives']['AU%']
        neg_au = data['negatives']['AU%']
        diff_gc = pos_gc - neg_gc
        diff_au = pos_au - neg_au
        diff_results.append({
            'RBP': rbp,
            'GC_pos': pos_gc,
            'GC_neg': neg_gc,
            'GC_diff': diff_gc,
            'AU_pos': pos_au,
            'AU_neg': neg_au,
            'AU_diff': diff_au
        })

# Output to CSV
import csv
csv_file = os.path.join(output_dir, 'clip_analysis_differences.csv')
with open(csv_file, 'w', newline='') as f:
    fieldnames = ['RBP', 'GC_pos', 'GC_neg', 'GC_diff', 'AU_pos', 'AU_neg', 'AU_diff']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in diff_results:
        writer.writerow(row)

# Output to TXT
txt_file = os.path.join(output_dir, 'clip_analysis_differences.txt')
with open(txt_file, 'w') as f:
    f.write("CLIP Data Analysis: GC and AU Content Differences\n")
    f.write("=" * 60 + "\n")
    f.write(f"{'RBP':<20} {'GC_pos':<8} {'GC_neg':<8} {'GC_diff':<8} {'AU_pos':<8} {'AU_neg':<8} {'AU_diff':<8}\n")
    f.write("-" * 60 + "\n")
    for row in diff_results:
        f.write(f"{row['RBP']:<20} {row['GC_pos']:<8.2f} {row['GC_neg']:<8.2f} {row['GC_diff']:<8.2f} {row['AU_pos']:<8.2f} {row['AU_neg']:<8.2f} {row['AU_diff']:<8.2f}\n")

print("Analysis complete. Files saved to", output_dir)