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
from transformers import AutoTokenizer, AutoConfig, EsmConfig
from LAMAR.modeling_nucESM2 import EsmForMaskedLM
from safetensors.torch import load_file
from Bio import SeqIO

# Verify GPU availability for PyTorch (LAMAR)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available for PyTorch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("WARNING: CUDA not available for PyTorch! LAMAR will use CPU (slow)")

# Configuration
data_dir = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/data/clip_training_data_uhl"
output_dir = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings/results/results_clip_data_uhl"
# Use workspace to avoid home directory quota limits
model_save_dir = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/models/models_clip_data_uhl"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

model_max_length = 256
extraction_batch_size = 32
downstream_batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use tokenizer from TAPT checkpoint (same vocab as Pretrained)
tokenizer_path = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/src/pretrain/saving_model/tapt_256_early_stopping_20/checkpoint-293000"
config_path = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/config/config_150M.json"

# Model Variants (Skip Pretrained - already completed)
variants = {
    "TAPT": "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/src/pretrain/saving_model/tapt_256_early_stopping_20/checkpoint-293000/model.safetensors",
    "Random": None
}

target_layers = [11]  # Evaluate only final layer for speed

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


def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)

def get_lamar_model(weights_path):
    """Load LAMAR model with proper config and weight loading."""
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length)
    
    # Load or create config
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        config = AutoConfig.from_pretrained(config_path)
    else:
        print(f"Config not found at {config_path}, creating from scratch")
        from transformers import EsmConfig
        config = EsmConfig(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            mask_token_id=tokenizer.mask_token_id,
            token_dropout=False,
            positional_embedding_type="rotary",
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=12,
            num_hidden_layers=12,
            problem_type="single_label_classification",
            num_labels=2
        )
    
    # Override key parameters to ensure consistency
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id
    config.mask_token_id = tokenizer.mask_token_id
    config.token_dropout = False
    config.positional_embedding_type = "rotary"
    
    model = EsmForMaskedLM(config)
    
    if weights_path:
        print(f"Loading weights from {weights_path}")
        if weights_path.endswith('.safetensors'):
            state_dict = load_file(weights_path)
        else:
            # Handle files without .safetensors extension (check if it's actually safetensors format)
            try:
                state_dict = load_file(weights_path)
            except Exception:
                state_dict = torch.load(weights_path, map_location="cpu")
            
        # Clean weight mapping logic
        weight_dict = {}
        for k, v in state_dict.items():
            if k.startswith("esm.lm_head"):
                new_k = k.replace("esm.", '', 1)
            elif k.startswith("lm_head"):
                new_k = k
            elif k.startswith("esm."):
                new_k = k
            else:
                new_k = "esm." + k
            weight_dict[new_k] = v
            
        result = model.load_state_dict(weight_dict, strict=False)
        print(f"Loaded weights: {result}")
    else:
        print("Initialized with random weights using WOLF strategy")
        model.apply(init_weights)
        
    model.to(device)
    model.eval()
    return model, tokenizer

# Identify RBPs from file names
files = glob.glob(os.path.join(data_dir, "*.positives.fa"))
rbps = [os.path.basename(f).replace(".positives.fa", "") for f in files]
print(f"Found RBPs: {rbps}")

# Function to check if RBP is already completed for a variant/layer
def is_rbp_completed(variant_name, layer, rbp):
    """Check if RBP results exist in the CSV file."""
    csv_path = os.path.join(output_dir, f"LAMAR_{variant_name}_L{layer}_results.csv")
    if not os.path.exists(csv_path):
        return False
    
    try:
        df = pd.read_csv(csv_path)
        return rbp in df['TF'].values
    except:
        return False

for variant_name, weights_path in variants.items():
    print(f"\n{'='*20}\nProcessing Variant: {variant_name}\n{'='*20}")
    
    lamar_model, tokenizer = get_lamar_model(weights_path)

    for layer in target_layers:
        print(f"\n--- Layer {layer} ---")
        
        results_data = []
        
        for rbp in rbps:
            # Skip if already completed
            if is_rbp_completed(variant_name, layer, rbp):
                print(f"Skipping {rbp} - already completed for {variant_name} Layer {layer}")
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
            
            # Extract Embeddings
            for split_name, (seqs, labels) in splits.items():
                total_embed = []
                with torch.no_grad():
                    for i in tqdm(range(0, len(seqs), extraction_batch_size), desc=f"Embed {split_name}"):
                        batch_seqs = seqs[i:i+extraction_batch_size].tolist()
                        tokens = tokenizer(
                            batch_seqs,
                            return_tensors="pt",
                            padding='max_length',
                            truncation=True,
                            max_length=model_max_length
                        ).to(device)
                        
                        outputs = lamar_model.esm(**tokens, output_hidden_states=True)
                        layer_embeddings = outputs.hidden_states[layer + 1]
                        total_embed.extend(layer_embeddings.cpu().numpy())
                
                # Convert to TF Dataset
                dataset_tf[split_name] = tf.data.Dataset.from_tensor_slices(
                    (total_embed, labels)
                ).shuffle(256*4).batch(downstream_batch_size)
                
                if split_name == 'train':
                    emb_shape = total_embed[0].shape
            
            # Train CNN
            print(f"Training CNN on ({emb_shape}) embeddings...")
            
            auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
            aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
            
            acc_scores = []
            auroc_scores = []
            aupr_scores = []
            
            # 5 Repeats
            for rep in range(5):
                cnn = chip_cnn(emb_shape, 1)
                cnn.compile(
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0),
                    metrics=['accuracy', auroc, aupr],
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
                )
                
                checkpoint_path = os.path.join(model_save_dir, f"{variant_name}_L{layer}_{rbp}_rep{rep}.h5")
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
                    # tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', save_freq='epoch')  # Disabled to avoid disk quota issues
                ]
                
                cnn.fit(
                    dataset_tf['train'],
                    validation_data=dataset_tf['valid'],
                    epochs=50, # Reduced from 100 for speed, usually converges fast
                    verbose=0,
                    callbacks=callbacks
                )
                
                _, acc, roc, pr = cnn.evaluate(dataset_tf['test'], verbose=0)
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
                'Model': f"LAMAR_{variant_name}_L{layer}",
                'Layer': layer,
                'Variant': variant_name
            })
            
        # Save results per variant/layer
        df = pd.DataFrame(results_data)
        csv_path = os.path.join(output_dir, f"LAMAR_256_{variant_name}_L{layer}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")

print("Done.")