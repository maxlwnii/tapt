import json
import os
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def train_val_split(metadata_path):
    with open(metadata_path) as f:
        data = json.load(f)

    train_data, val_data = train_test_split(
        data, test_size=0.02, random_state=RANDOM_SEED, shuffle=True
    )

    print(f"Total records: {len(data)}")
    print(f"Training records: {len(train_data)}")
    print(f"Validation records: {len(val_data)}")

    base, _ = os.path.splitext(metadata_path)
    train_path = f"{base}_train.json"
    val_path = f"{base}_val.json"

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)

    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=2)

if __name__ == "__main__":
    input_path = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/preprocess/preprocessed_data_1024_metadata.json"
    train_val_split(input_path)