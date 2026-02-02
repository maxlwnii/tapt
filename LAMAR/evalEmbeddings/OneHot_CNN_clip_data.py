import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import glob
import os
import random
from sklearn.model_selection import train_test_split
from Bio import SeqIO

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

# Helper to check GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found for TensorFlow, utilizing CPU/System Ram only.")

# Configuration
data_dir = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/data/clip_training_data"
output_dir = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings/results_clip_data"
model_save_dir = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings/models_clip_data_onehot"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

batch_size = 256
max_length = 512  # Same as LAMAR script

def one_hot_encode(seq, max_len=512):
    """
    One-hot encode DNA sequence.
    Returns shape (max_len, 4) for A, C, G, T
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Initialize with zeros
    one_hot = np.zeros((max_len, 4), dtype=np.float32)
    
    # Truncate or pad sequence
    seq = seq[:max_len].upper()
    
    for i, nucleotide in enumerate(seq):
        if nucleotide in mapping:
            one_hot[i, mapping[nucleotide]] = 1.0
    
    return one_hot

def load_data(rbp_name):
    """Load positive and negative sequences from FASTA files."""
    pos_file = os.path.join(data_dir, f"{rbp_name}.positives.fa")
    neg_file = os.path.join(data_dir, f"{rbp_name}.negatives.fa")
    
    seqs = []
    labels = []
    
    # Load Positives (Label 1)
    if os.path.exists(pos_file):
        for record in SeqIO.parse(pos_file, "fasta"):
            seq = str(record.seq).upper().replace("U", "T")
            seqs.append(seq)
            labels.append(1)
            
    # Load Negatives (Label 0)
    if os.path.exists(neg_file):
        for record in SeqIO.parse(neg_file, "fasta"):
            seq = str(record.seq).upper().replace("U", "T")
            seqs.append(seq)
            labels.append(0)
            
    return np.array(seqs), np.array(labels)

def get_baseline_cnn_onehot(input_shape, output_shape=1):
    """
    Baseline CNN for one-hot sequences.
    Same architecture as in OneHot_CNN.py
    """
    initializer = tf.keras.initializers.HeUniform(seed=42)
    input_layer = keras.Input(shape=input_shape)
    
    # Conv Layer 1: 64 filters, kernel 7
    nn = keras.layers.Conv1D(filters=64,
                             kernel_size=7,
                             padding='same',
                             kernel_initializer=initializer)(input_layer)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # Conv Layer 2: 96 filters, kernel 5
    nn = keras.layers.Conv1D(filters=96,
                             kernel_size=5,
                             padding='same',
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # Conv Layer 3: 128 filters, kernel 5
    nn = keras.layers.Conv1D(filters=128,
                             kernel_size=5,
                             padding='same',
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(2)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # Flatten
    nn = keras.layers.Flatten()(nn)

    # Dense Layer: 256 units
    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    # Output Layer
    logits = keras.layers.Dense(output_shape)(nn)
    output = keras.layers.Activation('sigmoid')(logits)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    return model

# Function to check if RBP is already completed
def is_rbp_completed(rbp):
    """Check if all 5 repetitions exist for this RBP."""
    for rep in range(5):
        checkpoint_path = os.path.join(model_save_dir, f"OneHot_{rbp}_rep{rep}.h5")
        if not os.path.exists(checkpoint_path):
            return False
    return True

# Identify RBPs from file names
files = glob.glob(os.path.join(data_dir, "*.positives.fa"))
rbps = [os.path.basename(f).replace(".positives.fa", "") for f in files]
print(f"Found {len(rbps)} RBPs: {rbps}")

results_data = []

# Metrics
auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)

for rbp in rbps:
    # Skip if already completed
    if is_rbp_completed(rbp):
        print(f"\nSkipping {rbp} - already completed")
        continue
        
    print(f"\n{'='*60}")
    print(f"Processing {rbp}")
    print(f"{'='*60}")
    
    # Load Data
    X, y = load_data(rbp)
    if len(X) == 0:
        print(f"Skipping {rbp}, no data found.")
        continue
        
    # Split Data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1111, random_state=42, stratify=y_train_val
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # One-hot encode sequences
    print("One-hot encoding sequences...")
    X_train_onehot = np.array([one_hot_encode(seq, max_length) for seq in tqdm(X_train, desc="Train")])
    X_val_onehot = np.array([one_hot_encode(seq, max_length) for seq in tqdm(X_val, desc="Val")])
    X_test_onehot = np.array([one_hot_encode(seq, max_length) for seq in tqdm(X_test, desc="Test")])
    
    print(f"One-hot shape: {X_train_onehot.shape}")
    
    # Create TF Datasets
    with tf.device("CPU"):
        dataset_train = tf.data.Dataset.from_tensor_slices(
            (X_train_onehot, y_train)
        ).shuffle(256*4).batch(batch_size)
        
        dataset_val = tf.data.Dataset.from_tensor_slices(
            (X_val_onehot, y_val)
        ).shuffle(256*4).batch(batch_size)
        
        dataset_test = tf.data.Dataset.from_tensor_slices(
            (X_test_onehot, y_test)
        ).batch(batch_size)
    
    input_shape = X_train_onehot.shape[1:]  # (max_length, 4)
    
    acc_scores = []
    auroc_scores = []
    aupr_scores = []
    
    # 5 Repeats
    for rep in range(5):
        print(f"\n--- Repeat {rep+1}/5 ---")
        
        checkpoint_path = os.path.join(model_save_dir, f"OneHot_{rbp}_rep{rep}.h5")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, 
                monitor='val_loss', 
                save_best_only=True, 
                mode='min', 
                save_freq='epoch'
            )
        ]
        
        cnn = get_baseline_cnn_onehot(input_shape, 1)
        cnn.compile(
            loss=loss,
            metrics=['accuracy', auroc, aupr],
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        )
        
        cnn.fit(
            dataset_train,
            validation_data=dataset_val,
            epochs=100,
            verbose=2,
            callbacks=callbacks
        )
        
        _, acc, roc, pr = cnn.evaluate(dataset_test, verbose=0)
        acc_scores.append(acc)
        auroc_scores.append(roc)
        aupr_scores.append(pr)
        
        print(f"Rep {rep+1}: Acc={acc:.4f}, AUROC={roc:.4f}, AUPR={pr:.4f}")
    
    # Aggregate results
    mean_acc = np.mean(acc_scores)
    mean_auroc = np.mean(auroc_scores)
    mean_aupr = np.mean(aupr_scores)
    
    print(f"\nFinal Result for {rbp}:")
    print(f"  Accuracy: {mean_acc:.4f}")
    print(f"  AUROC: {mean_auroc:.4f}")
    print(f"  AUPR: {mean_aupr:.4f}")
    
    results_data.append({
        'TF': rbp,
        'Accuracy': mean_acc,
        'AUROC': mean_auroc,
        'AUPR': mean_aupr,
        'Model': 'OneHot_CNN'
    })
    
    # Save intermediate results
    df = pd.DataFrame(results_data)
    csv_path = os.path.join(output_dir, "OneHot_CNN_clip_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved intermediate results to {csv_path}")

# Final save
df = pd.DataFrame(results_data)
csv_path = os.path.join(output_dir, "OneHot_CNN_clip_results.csv")
df.to_csv(csv_path, index=False)

print("\n" + "="*60)
print("All Done!")
print("="*60)
print(f"\nFinal results saved to {csv_path}")
print("\nSummary Statistics:")
print(df[['Accuracy', 'AUROC', 'AUPR']].describe())
