import csv
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def parse_fasta_file(filepath, label):
    """
    Parse a FASTA file and return sequences with their labels.
    The given .fa files always contain some description lines starting with '>' followed by the
    sequence on the next tow lines.
    Args:
        filepath: Path to the FASTA file
        label: Numerical label for all sequences in this file (0 or 1)

    Returns:
        List of tuples (sequence, label)
    """
    sequences = []
    current_sequence = ""

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this is a header line (starts with >)
        if line.startswith('>'):
            if current_sequence:
                sequences.append((current_sequence.replace('U', 'T'), label))
                current_sequence = ""
            i += 1

            # Read the sequence 
            while i < len(lines) and not lines[i].strip().startswith('>'):
                current_sequence += lines[i].strip()
                i += 1
        else:
            i += 1
    if current_sequence:
        sequences.append((current_sequence.replace('U', 'T'), label))

    return sequences


def create_csv_files():
    """
    Create train.csv, dev.csv, and test.csv from the FASTA files.
    """
    # Parse negative sequences (label = 0)
    negative_sequences = parse_fasta_file('data/negatives.fa', 0)
    print(f"Found {len(negative_sequences)} negative sequences")

    # Parse positive sequences (label = 1)
    positive_sequences = parse_fasta_file('data/positives.fa', 1)
    print(f"Found {len(positive_sequences)} positive sequences")

    all_sequences = negative_sequences + positive_sequences
    print(f"Total sequences: {len(all_sequences)}")
    df = pd.DataFrame(all_sequences, columns=['sequence', 'label'])
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    print(f"Train set: {len(train_df)} sequences")
    print(f"Dev set: {len(dev_df)} sequences")
    print(f"Test set: {len(test_df)} sequences")

    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    dev_df.to_csv(os.path.join(output_dir, 'dev.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print("\nSample from train.csv:")
    print(train_df.head())

    print(f"\nLabel distribution in train set:")
    print(train_df['label'].value_counts())
    
create_csv_files()