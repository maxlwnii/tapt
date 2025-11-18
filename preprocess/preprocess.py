"""
This script preprocesses eCLIP data.
It reads a fasta file containg verified CLIP peaks and a BED file with eCLIP peaks.
It generates fixed-length sequences based on the following strategy:
1) For sequences shorter than max_len, take the entire sequence (later padded)
2) For sequences longer than max_len: Sample overlapping windows of length max_len with stride 
3) For very long sequences, it samples sqrt(#b inding_sites) often random regions of max_len size
"""

import numpy as np
from Bio import SeqIO
from collections import defaultdict
import argparse
import json


def read_fasta_file(fasta_file):
    """Reads a fasta file and returns a dictionary of sequences.
    
    >>> fa_path = '../doctest/test.fa' # Assumes 'test.fa' exists in the same directory
    >>> result = read_fasta_file(fa_path)
    
    # Sort for stable output
    >>> result_sorted = sorted(result, key=lambda x: x['chrom'])
    >>> for item in result_sorted:
    ...     print(item)
    {'chrom': 'chr1', 'seq_id': 'chr1:100-110', 'start': 100, 'end': 110, 'sequence': 'AGCAGTCGAT', 'seq_len': 10}
    {'chrom': 'chr2', 'seq_id': 'chr2:200-205', 'start': 200, 'end': 205, 'sequence': 'ACGTG', 'seq_len': 5}
    {'chrom': 'chrX', 'seq_id': 'chrX:50-55', 'start': 50, 'end': 55, 'sequence': 'CCGGT', 'seq_len': 5}
    """
    sequences = list()
    for record in SeqIO.parse(fasta_file, "fasta"):
        try:
            chrom, rest = record.id.split(':')
            if not is_standard_chromosome(chrom):
                continue

            start, end = rest.split('-')
            sequence = str(record.seq).upper()

        except ValueError:
            continue

        sequences.append(
            {
                'chrom': chrom,
                'seq_id': record.id,
                'start': int(start),
                'end': int(end),
                'sequence': sequence,
                'seq_len': len(sequence)

            }
        )
    
    return sequences


def is_standard_chromosome(chrom):
    """Checks if a chromosome is a standard chromosome (1-22, X, Y).
    >>> is_standard_chromosome('chr1')
    True
    >>> is_standard_chromosome('chrX')
    True
    >>> is_standard_chromosome('chrM')
    False
    >>> is_standard_chromosome('chrUn_gl000220')
    False
    """
    standard_chroms = {f'chr{i}' for i in range(1, 23)}.union({'chrX', 'chrY'})
    return chrom in standard_chroms


def read_bed_file(bed_file, filter_random=True):
    """
    Reads a BED file and returns a dictionary of peaks.
    Args:
        bed_file: Path to the BED file.
        filter_random: If True, filters out random/unplaced chromosomes.

    >>> import json
    >>> bed_path = '../doctest/test.bed'
    >>> result = read_bed_file(bed_path)
    >>> print(json.dumps(result, sort_keys=True, indent=2))
    {
      "chr1": [
        {
          "end": 1100,
          "num_binding_sites": 5,
          "start": 1000
        },
        {
          "end": 2100,
          "num_binding_sites": 10,
          "start": 2000
        }
      ],
      "chrY": [
        {
          "end": 600,
          "num_binding_sites": 2,
          "start": 500
        }
      ]
    }
    """
    peaks = defaultdict(list)
    with open(bed_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            
            parts = line.strip().split('\t')
            chrom = parts[0]
            if not is_standard_chromosome(chrom) and filter_random:
                continue

            start = int(parts[1])
            end = int(parts[2])
            num_binding_sites = int(parts[4])
            peaks[chrom].append({
                'start': start,
                'end': end,
                'num_binding_sites': num_binding_sites
            })
    return peaks


def read_eclip_bed_file(bed_file, filter_random=True):
    """
    Reads an eCLIP BED file and returns a dictionary of peaks.
    Args:
        bed_file: Path to the BED file.
        filter_random: If True, filters out random/unplaced chromosomes.

    >>> import json
    >>> bed_path = '../doctest/test_eclip.bed'
    >>> result = read_eclip_bed_file(bed_path)
    >>> print(json.dumps(result, sort_keys=True, indent=2))
    {
      "chr1": [
        {
          "end": 105,
          "start": 102
        },
        {
          "end": 550,
          "start": 500
        },
        {
          "end": 103,
          "start": 102
        }
      ],
      "chrX": [
        {
          "end": 54,
          "start": 51
        }
      ]
    }
    """
    peaks = defaultdict(list)
    with open(bed_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            
            parts = line.strip().split()
            chrom = parts[0]
            if not is_standard_chromosome(chrom) and filter_random:
                continue

            start = int(parts[1])
            end = int(parts[2])
            peaks[chrom].append({
                'start': start,
                'end': end
            })

    return peaks


def combine_eclip_fasta(sequences, eclip_peaks):
    """
    Combines sequences from fasta with eCLIP peaks.
    Args:
        sequences: List of sequences from fasta.
        eclip_peaks: Dictionary of eCLIP peaks.
    Returns:
        Dictionary with sequence info and list of eCLIP peak coordinates for each sequence.
        
    >>> sequences = read_fasta_file('../doctest/test.fa')
    >>> eclip_peaks = read_eclip_bed_file('../doctest/test_eclip.bed')
    
    >>> combined_result = combine_eclip_fasta(sequences, eclip_peaks)
    >>> print(json.dumps(combined_result, indent=2, sort_keys=True))
    {
      "chr1:100-110": {
        "peaks": [
          {
            "genomic_end": 105,
            "genomic_start": 102,
            "peak_end": 5,
            "peak_start": 2,
            "seq_len": 3,
            "sequence": "CAG"
          },
          {
            "genomic_end": 103,
            "genomic_start": 102,
            "peak_end": 3,
            "peak_start": 2,
            "seq_len": 1,
            "sequence": "C"
          }
        ],
        "sequence_info": {
          "chrom": "chr1",
          "end": 110,
          "seq_id": "chr1:100-110",
          "seq_len": 10,
          "sequence": "AGCAGTCGAT",
          "start": 100
        }
      },
      "chr2:200-205": {
        "peaks": [],
        "sequence_info": {
          "chrom": "chr2",
          "end": 205,
          "seq_id": "chr2:200-205",
          "seq_len": 5,
          "sequence": "ACGTG",
          "start": 200
        }
      },
      "chrX:50-55": {
        "peaks": [
          {
            "genomic_end": 54,
            "genomic_start": 51,
            "peak_end": 4,
            "peak_start": 1,
            "seq_len": 3,
            "sequence": "CGG"
          }
        ],
        "sequence_info": {
          "chrom": "chrX",
          "end": 55,
          "seq_id": "chrX:50-55",
          "seq_len": 5,
          "sequence": "CCGGT",
          "start": 50
        }
      }
    }
    """
    
    combined = {}
    for seq in sequences:
        seq_id = seq['seq_id']
        seq_start = seq['start']
        seq_end = seq['end']
        chrom = seq['chrom']
        
        # Initialize entry for every sequence
        combined[seq_id] = {
            "sequence_info": seq,
            "peaks": []
        }
        
        # Check if chromosome has any peaks
        if chrom not in eclip_peaks:
            continue
            
        for peak in eclip_peaks[chrom]:
            if peak['start'] < seq_start or peak['end'] > seq_end:
                continue
            combined_start = peak['start'] - seq_start
            combined_end = peak['end'] - seq_start
            
            combined[seq_id]["peaks"].append({
                'peak_start': combined_start,
                'peak_end': combined_end,
                'sequence': seq['sequence'][combined_start : combined_end],
                'seq_len': peak['end'] - peak['start'],
                'genomic_start': peak['start'],
                'genomic_end': peak['end'],
            })

    return combined


def sliding_window(seq_id, seq, eclip_peaks_local, max_len=512, stride=256):
    """
    Sequences longer than max_length but not considered very long are split into overlapping windows
    using a pre-defined fix-size stride.
    
    Args:
        seq_id: Sequence identifier
        seq: Full sequence string
        eclip_peaks_local: List of eCLIP peaks with coordinates relative to the full sequence
        max_len: Maximum window length
        stride: Step size for sliding window
        
    Returns:
        List of windows, each with eCLIP peaks mapped to window-local coordinates
    """
    seq_len = len(seq)
    windows = []
    
    for start in range(0, seq_len - max_len + 1, stride):
        end = start + max_len
        
        # Map eCLIP peaks to this window's local coordinates
        peaks_in_window = []
        for peak in eclip_peaks_local:
            peak_start = peak['peak_start']
            peak_end = peak['peak_end']
            
            # Check if peak overlaps with window [start, end)
            if peak_end > start and peak_start < end:
                # Convert to window-local coordinates
                local_start = max(0, peak_start - start)
                local_end = min(max_len, peak_end - start)
                
                peaks_in_window.append({
                    'peak_start': local_start,
                    'peak_end': local_end,
                    'genomic_start': peak['genomic_start'],
                    'genomic_end': peak['genomic_end']
                })
        
        windows.append({
            'sequence': seq[start:end],
            'seq_id': f"{seq_id}_window_{start}_{end}",
            'start': start,
            'end': end,
            'method': 'sliding_window',
            'seq_len': max_len,
            'eclip_regions': peaks_in_window  # Window-local coordinates
        })
    
    # Handle last window if not aligned
    if windows and windows[-1]['end'] < seq_len:
        last_start = seq_len - max_len
        if last_start > windows[-1]['start']:  # Avoid duplicate
            peaks_in_window = []
            for peak in eclip_peaks_local:
                peak_start = peak['peak_start']
                peak_end = peak['peak_end']
                
                if peak_end > last_start and peak_start < seq_len:
                    local_start = max(0, peak_start - last_start)
                    local_end = min(max_len, peak_end - last_start)
                    
                    peaks_in_window.append({
                        'peak_start': local_start,
                        'peak_end': local_end,
                        'genomic_start': peak['genomic_start'],
                        'genomic_end': peak['genomic_end']
                    })
            
            windows.append({
                'sequence': seq[last_start:seq_len],
                'seq_id': f"{seq_id}_window_{last_start}_{seq_len}",
                'start': last_start,
                'end': seq_len,
                'method': 'sliding_window',
                'seq_len': max_len,
                'eclip_regions': peaks_in_window
            })

    return windows


def weighted_sampling(combined, seq, max_len=512, min_samples=1):
    """
    Very long sequences are processed by:
    1. Sampling eCLIP_binding windows centered on each peak
    2. Additional uniform random sampling for diversity
    """
    windows = []
    seq_id = seq['seq_id']
    seq_length = seq['seq_len']
    full_sequence = seq['sequence']
    peaks_list = combined.get(seq_id, {}).get('peaks', [])
    eCLIP_binding = len(peaks_list)
    
    # Sanity check - in your case, always > 0
    assert eCLIP_binding > 0, f"No eCLIP peaks found for {seq_id}"
    
    # PART 1: Sample around each peak
    for _ in range(eCLIP_binding):
        peak = np.random.choice(peaks_list)
        peak_start = peak['peak_start']
        peak_end = peak['peak_end']
        center = (peak_start + peak_end) // 2
        ideal_start = center - (max_len // 2)
        window_start = max(0, min(ideal_start, seq_length - max_len))
        window_end = min(window_start + max_len, seq_length)
        
        window_id = f"{seq_id}_weighted_{window_start}_{window_end}"
        if window_id in [w['seq_id'] for w in windows]:
            continue
        
        # Map peaks in this window
        peaks_in_window = []
        for p in peaks_list:
            if p['peak_end'] > window_start and p['peak_start'] < window_end:
                local_start = max(0, p['peak_start'] - window_start)
                local_end = min(max_len, p['peak_end'] - window_start)
                peaks_in_window.append({
                    'peak_start': local_start,
                    'peak_end': local_end,
                    'genomic_start': p['genomic_start'],
                    'genomic_end': p['genomic_end']
                })
        
        windows.append({
            'sequence': full_sequence[window_start:window_end],
            'seq_id': window_id,
            'start': window_start,
            'end': window_end,
            'method': 'weighted_peak_centered',
            'seq_len': len(full_sequence[window_start:window_end]),
            'eclip_regions': peaks_in_window,
            'num_peaks_in_window': len(peaks_in_window)
        })
    
    # PART 2: Additional uniform sampling for diversity
    num_uniform = max(min_samples, int(np.sqrt(seq_length / max_len)))
    
    for _ in range(num_uniform):
        max_start = max(0, seq_length - max_len)
        window_start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        window_end = min(window_start + max_len, seq_length)
        
        window_id = f"{seq_id}_uniform_{window_start}_{window_end}"
        if window_id in [w['seq_id'] for w in windows]:
            continue
        
        peaks_in_window = []
        for p in peaks_list:
            if p['peak_end'] > window_start and p['peak_start'] < window_end:
                local_start = max(0, p['peak_start'] - window_start)
                local_end = min(max_len, p['peak_end'] - window_start)
                peaks_in_window.append({
                    'peak_start': local_start,
                    'peak_end': local_end,
                    'genomic_start': p['genomic_start'],
                    'genomic_end': p['genomic_end']
                })
        
        windows.append({
            'sequence': full_sequence[window_start:window_end],
            'seq_id': window_id,
            'start': window_start,
            'end': window_end,
            'method': 'weighted_uniform',
            'seq_len': len(full_sequence[window_start:window_end]),
            'eclip_regions': peaks_in_window,
            'num_peaks_in_window': len(peaks_in_window)
        })
    
    return windows


def preprocess_sequences(sequences, peaks_dict, combined, max_len=512, stride=256):
    """Preprocess sequences based on their lengths."""
    processed_data = []
    stats = {
        'short_padded': 0,
        'medium_windowed': 0,
        'long_sampled': 0,
        'total_windows': 0
    }

    for seq in sequences:
        seq_len = seq['seq_len']
        seq_id = seq['seq_id']
        full_sequence = seq['sequence']
        num_binding_sites = 0
        
        if seq['chrom'] in peaks_dict:
            for peak in peaks_dict[seq['chrom']]:
                if peak['start'] >= seq['start'] and peak['end'] <= seq['end']:
                    num_binding_sites += peak['num_binding_sites']
        
        seq['num_binding_sites'] = num_binding_sites
        
        # Get eCLIP peaks for this sequence (already in local coordinates from combine_eclip_fasta)
        eclip_peaks_local = combined.get(seq_id, {}).get('peaks', [])

        if seq_len <= max_len:
            # Short sequences: entire sequence (peaks are already in correct coordinates)
            peaks_in_window = [
                {
                    'peak_start': p['peak_start'],
                    'peak_end': p['peak_end'],
                    'genomic_start': p['genomic_start'],
                    'genomic_end': p['genomic_end']
                }
                for p in eclip_peaks_local
            ]
            
            processed_data.append({
                'sequence': full_sequence,
                'seq_id': seq_id,
                'start': 0,
                'end': seq_len,
                'method': 'full_sequence',
                'seq_len': seq_len,
                'eclip_regions': peaks_in_window
            })
            stats['short_padded'] += 1
            stats['total_windows'] += 1

        elif seq_len <= 12 * max_len:
            # Medium sequences: sliding window
            windows = sliding_window(seq_id, full_sequence, eclip_peaks_local, max_len, stride)
            processed_data.extend(windows)
            stats['medium_windowed'] += 1
            stats['total_windows'] += len(windows)

        else:
            # Long sequences: weighted sampling
            windows = weighted_sampling(combined, seq, max_len)
            processed_data.extend(windows)
            stats['long_sampled'] += 1
            stats['total_windows'] += len(windows)

    return processed_data, stats


def save_processed_data(processed_data, output_prefix):
    """Save processed sequences and metadata."""
    
    # Save sequences as text file (one per line) for easy loading
    with open(f'{output_prefix}_sequences.txt', 'w') as f:
        for item in processed_data:
            f.write(item['sequence'] + '\n')
    
    # Save metadata as JSON (convert numpy types to Python types)
    metadata = []
    for item in processed_data:
        meta_dict = {}
        for k, v in item.items():
               # Convert numpy types to Python native types
            if hasattr(v, 'item'):  # numpy scalar
                meta_dict[k] = v.item()
            elif isinstance(v, (np.integer, np.int64, np.int32)):
                meta_dict[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                meta_dict[k] = float(v)
            else:
                meta_dict[k] = v
        metadata.append(meta_dict)
    
    with open(f'{output_prefix}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved {len(processed_data)} processed sequences to:")
    print(f"  - {output_prefix}_sequences.txt")
    print(f"  - {output_prefix}_metadata.json")


def merge_overlapping_peaks(eclip_peaks_dict, min_overlap=1):
    """
    Merge overlapping eCLIP peaks within each chromosome.
    
    Args:
        eclip_peaks_dict: Dictionary of eCLIP peaks by chromosome.
        min_overlap: Minimum overlap to consider peaks as overlapping (in base pairs).
        
    Returns:
        Dictionary with merged peaks.
    """
    merged_peaks = defaultdict(list)
    
    for chrom, peaks in eclip_peaks_dict.items():
        # Sort peaks by start position
        sorted_peaks = sorted(peaks, key=lambda x: x['start'])
        current_merged = None
        
        for peak in sorted_peaks:
            if current_merged is None:
                current_merged = {
                    'start': peak['start'],
                    'end': peak['end']
                }
            else:
                # Check if there is an overlap
                if peak['start'] <= current_merged['end'] + min_overlap:
                    # Merge by extending the end
                    current_merged['end'] = max(current_merged['end'], peak['end'])
                else:
                    # No overlap, push current merged and start new
                    merged_peaks[chrom].append(current_merged)
                    current_merged = {
                        'start': peak['start'],
                        'end': peak['end']
                    }
        
        # Add the last merged peak
        if current_merged is not None:
            merged_peaks[chrom].append(current_merged)
    
    return merged_peaks


def print_merge_statistics(original_peaks, merged_peaks):
    """
    Print statistics about peak merging.
    
    Args:
        original_peaks: Original eCLIP peaks dictionary.
        merged_peaks: Merged eCLIP peaks dictionary.
    """
    total_original = sum(len(peaks) for peaks in original_peaks.values())
    total_merged = sum(len(peaks) for peaks in merged_peaks.values())
    
    print(f"Merged {total_original} original peaks into {total_merged} merged peaks.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess eCLIP data.")
    parser.add_argument('--fasta', type=str, required=True, help='Path to the fasta file with verified CLIP peaks.')
    parser.add_argument('--bed', type=str, required=True, help='Path to the BED file with eCLIP peaks.')
    parser.add_argument('--eclip', type=str, required=True, help='Path to the eCLIP BED file.')
    parser.add_argument('--output', type=str, required=False, help='Path to the output JSON file.')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum length of sequences.')
    parser.add_argument('--stride', type=int, default=256, help='Stride for sliding window.')
    parser.add_argument('--merge_peaks', action='store_true', help='Merge overlapping eCLIP peaks.')
    
    args = parser.parse_args()

    print("=" * 80)
    print("CLIP Peak Sequence Preprocessing")
    print("=" * 80)
    
    # Read input files
    print("\n[1/5] Reading FASTA file...")
    sequences = read_fasta_file(args.fasta)
    
    print("\n[2/5] Reading BED file...")
    peaks_dict = read_bed_file(args.bed, filter_random=True)
    
    print("\n[3/5] Reading eCLIP BED file...")
    eclip_peaks_dict = read_eclip_bed_file(args.eclip, filter_random=True)
    
    # Merge overlapping peaks if requested
    if args.merge_peaks:
        print("\n[3.5/5] Merging overlapping eCLIP peaks...")
        eclip_peaks_merged = merge_overlapping_peaks(eclip_peaks_dict, min_overlap=1)
        print_merge_statistics(eclip_peaks_dict, eclip_peaks_merged)
        eclip_peaks_dict = eclip_peaks_merged
    
    combined = combine_eclip_fasta(sequences, eclip_peaks_dict)
    
    # Preprocess
    print(f"\n[4/5] Processing sequences (max_len={args.max_len}, stride={args.stride})...")
    processed_data, stats = preprocess_sequences(
        sequences, peaks_dict, combined,
        max_len=args.max_len, 
        stride=args.stride
    )
    
    # Save results
    print("\n[5/5] Saving processed data...")
    save_processed_data(processed_data, args.output)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("PREPROCESSING STATISTICS")
    print("=" * 80)
    print(f"Short sequences (padded):          {stats['short_padded']:>8}")
    print(f"Medium sequences (windowed):       {stats['medium_windowed']:>8}")
    print(f"Long sequences (peak-sampled):     {stats['long_sampled']:>8}")
    print(f"{'─' * 50}")
    print(f"Total input sequences:             {len(sequences):>8}")
    print(f"Total output windows:              {stats['total_windows']:>8}")
    print(f"Expansion factor:                  {stats['total_windows']/len(sequences):>8.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()