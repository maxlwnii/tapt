import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import glob
import os
import random
from sklearn.model_selection import train_test_split

# Set TensorFlow logging level BEFORE import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

# Force TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')
print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow configured to use CPU only (efficient for small CNN)")

# Import PyTorch (will have access to GPU)
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
from Bio import SeqIO

# Verify GPU availability for PyTorch (DNABERT2)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available for PyTorch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("WARNING: CUDA not available for PyTorch! DNABERT2 will use CPU (slow)")

# Configuration
data_dir = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/data/clip_training_data_uhl"
output_dir = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/DNABERT2/evalEmbeddings/results/results_clip_data_uhl"
# Use workspace to avoid home directory quota limits
model_save_dir = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/DNABERT2/models/models_clip_data_uhl"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

model_max_length = 512
extraction_batch_size = 32
downstream_batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DNABERT2 model from HuggingFace
model_name = "zhihan1996/DNABERT-2-117M"

def load_data(rbp_name):
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

def chip_cnn(input_shape, output_shape):
    initializer = tf.keras.initializers.HeUniform(seed=42)
    input = keras.Input(shape=input_shape)

    # Batchnorm and dimension reduction
    nn = keras.layers.BatchNormalization()(input)
    nn = keras.layers.Conv1D(filters=512, kernel_size=1, kernel_initializer=initializer)(nn)
    
    # First conv layer
    nn = keras.layers.Conv1D(filters=64, kernel_size=7, padding='same', kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # Second conv layer
    nn = keras.layers.Conv1D(filters=96, kernel_size=5, padding='same', kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # Third conv layer
    nn = keras.layers.Conv1D(filters=128, kernel_size=5, padding='same', kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(2)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # Dense layer
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    # Output layer
    logits = keras.layers.Dense(output_shape)(nn)
    output = keras.layers.Activation('sigmoid')(logits)
    
    model = keras.Model(inputs=input, outputs=output)
    return model


def get_dnabert2_model():
    """Load DNABERT2 model from HuggingFace."""
    print(f"Loading DNABERT2 model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = BertConfig.from_pretrained(
        model_name,
        cache_dir=None,
    )
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config)
    model.to(device)
    model.eval()
    print(f"DNABERT2 model loaded successfully")
    return model, tokenizer

# Identify RBPs from file names
files = glob.glob(os.path.join(data_dir, "*.positives.fa"))
rbps = [os.path.basename(f).replace(".positives.fa", "") for f in files]
print(f"Found RBPs: {rbps}")

# Function to check if RBP is already completed
def is_rbp_completed(rbp):
    """Check if RBP results exist in the CSV file."""
    csv_path = os.path.join(output_dir, "DNABERT2_results_acc_padded.csv")
    if not os.path.exists(csv_path):
        return False
    
    try:
        df = pd.read_csv(csv_path)
        return rbp in df['TF'].values
    except:
        return False

print(f"\n{'='*20}\nProcessing DNABERT2\n{'='*20}")

dnabert2_model, tokenizer = get_dnabert2_model()

results_data = []
        
for rbp in rbps:
    # Skip if already completed
    if is_rbp_completed(rbp):
        print(f"Skipping {rbp} - already completed")
        continue
        
    print(f"Processing {rbp}")
    
    # Load Data
    X, y = load_data(rbp)
    if len(X) == 0:
        print(f"Skipping {rbp}, no data found.")
        continue
        
    # Split Data
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42, stratify=y_train_val) # 0.1111 of 0.9 is ~ 0.1
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    splits = {
        'train': (X_train, y_train),
        'valid': (X_val, y_val),
        'test': (X_test, y_test)
    }
    
    dataset_tf = {}
    emb_shape = None
    
    # Extract Embeddings using DNABERT2
    # Extract Embeddings using DNABERT2
    for split_name, (seqs, labels) in splits.items():
        total_embed = []
        with torch.no_grad():
            for i in tqdm(range(0, len(seqs), extraction_batch_size), desc=f"Embed {split_name}"):
                batch_seqs = seqs[i:i+extraction_batch_size].tolist()
                
                # 1. Get the full encoding (ids AND mask)
                encoded_input = tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=model_max_length
                ).to(device) # .to(device) works on the whole dictionary in recent Transformers versions
                
                # 2. Pass the full dictionary to the model using ** (unpacking)
                outputs = dnabert2_model(**encoded_input)
                
                # 3. Use the hidden states (usually the first element of the output tuple)
                hidden_states = outputs[0]  # Shape: [batch, seq_len, 768]
                
                # 4. Perform pooling (using the mask to ignore padding)
                mask = encoded_input['attention_mask']
                mask_expanded = mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            
                total_embed.extend(embeddings.cpu().numpy())
        
        # Convert to TF Dataset
        dataset_tf[split_name] = tf.data.Dataset.from_tensor_slices(
            (total_embed, labels)
        ).shuffle(256*4).batch(downstream_batch_size)
        
        if split_name == 'train':
            emb_shape = total_embed[0].shape
            
    # Train CNN (using Dense layers since we have mean-pooled embeddings)
    print(f"Training CNN on ({emb_shape}) embeddings...")
    
    # For mean-pooled embeddings (768,), we need a simple MLP instead of CNN
    def simple_mlp(input_shape, output_shape):
        initializer = tf.keras.initializers.HeUniform(seed=42)
        input = keras.Input(shape=input_shape)
        
        nn = keras.layers.BatchNormalization()(input)
        nn = keras.layers.Dense(512, kernel_initializer=initializer)(nn)
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(0.3)(nn)
        
        nn = keras.layers.Dense(256, kernel_initializer=initializer)(nn)
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(0.3)(nn)
        
        nn = keras.layers.Dense(128, kernel_initializer=initializer)(nn)
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(0.5)(nn)
        
        logits = keras.layers.Dense(output_shape)(nn)
        output = keras.layers.Activation('sigmoid')(logits)
        
        model = keras.Model(inputs=input, outputs=output)
        return model
    
    auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
    
    acc_scores = []
    auroc_scores = []
    aupr_scores = []
    
    # 5 Repeats
    for rep in range(5):
        mlp = simple_mlp(emb_shape, 1)
        mlp.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0),
            metrics=['accuracy', auroc, aupr],
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        )
        
        checkpoint_path = os.path.join(model_save_dir, f"DNABERT2_{rbp}_rep{rep}.h5")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        ]
        
        mlp.fit(
            dataset_tf['train'],
            validation_data=dataset_tf['valid'],
            epochs=50,
            verbose=0,
            callbacks=callbacks
        )
        
        _, acc, roc, pr = mlp.evaluate(dataset_tf['test'], verbose=0)
        acc_scores.append(acc)
        auroc_scores.append(roc)
        aupr_scores.append(pr)
        
    # Aggregate results
    mean_acc = np.mean(acc_scores)
    mean_auroc = np.mean(auroc_scores)
    mean_aupr = np.mean(aupr_scores)
    
    print(f"Result: Val Acc={mean_acc:.4f}, AUROC={mean_auroc:.4f}, AUPR={mean_aupr:.4f}")
    
    results_data.append({
        'TF': rbp,
        'Accuracy': mean_acc,
        'AUROC': mean_auroc,
        'AUPR': mean_aupr,
        'Model': 'DNABERT2'
    })
    
    # Save results incrementally
    df = pd.DataFrame(results_data)
    csv_path = os.path.join(output_dir, "DNABERT2_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

print("Done.")