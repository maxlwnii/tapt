import os
import csv
from datasets import load_from_disk

def convert_dataset_to_csv(rbp_name, dataset_dir, output_dir):
    """Convert HuggingFace dataset to CSV format for DNABERT2."""
    
    dataset = load_from_disk(dataset_dir)
    
    # Convert each split
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            # DNABERT2 expects dev.csv for validation
            csv_name = 'dev.csv' if split == 'validation' else f'{split}.csv'
            
            output_path = os.path.join(output_dir, rbp_name, csv_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['sequence', 'label'])  # header
                
                for seq, label in zip(dataset[split]['seq'], dataset[split]['label']):
                    seq = seq.replace('U', 'T').replace('u', 't')
                    writer.writerow([seq, label])
            
            print(f"Saved {split} ({len(dataset[split])} samples) to {output_path}")

def main():
    dataset_base = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/data/finetune_data"
    output_base = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/DNABERT_2_project/data"
    
    rbps = [
        "GTF2F1_K562_IDR",
        "HNRNPL_K562_IDR", 
        "HNRNPM_HepG2_IDR",
        "ILF3_HepG2_IDR",
        "KHSRP_K562_IDR",
        "MATR3_K562_IDR",
        "PTBP1_HepG2_IDR",
        "QKI_K562_IDR"
    ]
    
    for rbp in rbps:
        dataset_dir = os.path.join(dataset_base, rbp)
        convert_dataset_to_csv(rbp, dataset_dir, output_dir=output_base)

if __name__ == "__main__":
    main()