import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import h5py
import glob
import os

# Set TensorFlow logging level BEFORE import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

# Force TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')
print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow configured to use CPU only (efficient for small MLP)")

# Import PyTorch (will have access to GPU)
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

# Verify GPU availability for PyTorch (DNABERT2)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available for PyTorch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("WARNING: CUDA not available for PyTorch! DNABERT2 will use CPU (slow)")

# Configuration
model_max_length = 512
extraction_batch_size = 16
downstream_batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DNABERT2 will use: {device}")

# Paths
data_dir = '/home/fr/fr_fr/fr_ml642/Thesis/data/eclip_koo'  # KOO h5 files
output_dir = '/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/DNABERT2/evalEmbeddings/results/results_koo_data'
model_save_dir = '/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/DNABERT2/models/models_koo_data'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

# DNABERT2 model from HuggingFace
model_name = "zhihan1996/DNABERT-2-117M"

file_list = glob.glob(os.path.join(data_dir, '*.h5'))
print(f"Found {len(file_list)} h5 files")

# MLP for mean-pooled embeddings (768,)
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

# Callbacks and metrics
earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6
)
auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)

# Load DNABERT2 model
print(f'Loading DNABERT2 model: {model_name}')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = BertConfig.from_pretrained(model_name, trust_remote_code=True)
dnabert2_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config)
dnabert2_model.to(device)
dnabert2_model.eval()
print("DNABERT2 model loaded successfully")

# Function to check if TF is already completed
def is_tf_completed(tf_name):
    """Check if TF results exist in the CSV file."""
    csv_path = os.path.join(output_dir, "DNABERT2_koo_results.csv")
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        return tf_name in df['TF'].values
    except:
        return False

# Results storage
test_aupr = []
test_auroc = []
test_accuracy = []
tf_list = []
model_list = []

# Process each TF file
for file in file_list:
    tf_name = os.path.basename(file).replace('_200_eclip.h5', '').replace('.h5', '')
    
    # Skip if already completed
    if is_tf_completed(tf_name):
        print(f"Skipping {tf_name} - already completed")
        continue
    
    print(f'\nProcessing {tf_name} with DNABERT2')
    
    try:
        data = h5py.File(file, 'r')
    except Exception as e:
        print(f"Error loading {file}: {e}")
        continue
    
    dataset = {}
    embeddings_dict = {}

    # Create dataset for train, valid, test
    for label in ['test', 'valid', 'train']:
        # Convert one-hot to sequences
        sequence = data['X_' + label][()]
        sequence = np.transpose(sequence, (0, 2, 1))
        
        # Convert one-hot (first 4 channels) to DNA strings
        seq_strings = []
        for seq in sequence:
            bases = ['A', 'C', 'G', 'T']
            dna = ''.join([bases[np.argmax(pos[:4])] for pos in seq])
            seq_strings.append(dna)
        
        target = data['Y_' + label][()]
        actual_seq_len = len(seq_strings[0]) if seq_strings else model_max_length
        print(f'Extracting embeddings for {label} set ({len(seq_strings)} sequences, length={actual_seq_len})')
        
        # Extract DNABERT2 embeddings with mean pooling
        total_embed = []
        with torch.no_grad():
            for i in tqdm(range(0, len(seq_strings), extraction_batch_size), desc=f"Embed {label}"):
                batch_seqs = seq_strings[i:i + extraction_batch_size]
                encoded_input = tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=model_max_length
                ).to(device)
                
                outputs = dnabert2_model(**encoded_input)
                hidden_states = outputs[0]
                
                # Masked Mean Pooling
                mask = encoded_input['attention_mask']
                mask_expanded = mask.unsqueeze(-1).expand(hidden_states.size()).float()
                embeddings = torch.sum(hidden_states * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
                
                total_embed.extend(embeddings.cpu().numpy())
                
                # Store embeddings
                embeddings_dict[label] = total_embed
        
        # Create TF dataset
        with tf.device("CPU"):
            dataset[label] = tf.data.Dataset.from_tensor_slices(
                (total_embed, target)
            ).shuffle(256 * 4).batch(downstream_batch_size)
    
    # Get embedding shape
    emb_shape = embeddings_dict['train'][0].shape
    print(f'Embedding shape: {emb_shape}')
    
    print(f'Training MLP for {tf_name} with DNABERT2 embeddings')
    
    # 5 repeats for robustness
    for rep in range(5):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        print(f'Creating MLP with input shape: {emb_shape}')
        mlp_model = simple_mlp(emb_shape, 1)
        mlp_model.compile(
            loss=loss,
            metrics=['accuracy', auroc, aupr],
            optimizer=optimizer
        )
        
        result = mlp_model.fit(
            dataset['train'],
            validation_data=dataset['valid'],
            epochs=100,
            verbose=0,
            callbacks=[earlyStopping_callback, reduce_lr]
        )
        
        _, acc, roc, pr = mlp_model.evaluate(dataset['test'], verbose=0)
        tf_list.append(tf_name)
        test_accuracy.append(acc)
        test_auroc.append(roc)
        test_aupr.append(pr)
        model_list.append('DNABERT2')
        
        print(f'  Rep {rep}: Acc={acc:.4f}, AUROC={roc:.4f}, AUPR={pr:.4f}')
    
    data.close()
    
    # Save results incrementally
    df = pd.DataFrame({
        'TF': tf_list,
        'Accuracy': test_accuracy,
        'AUROC': test_auroc,
        'AUPR': test_aupr,
        'Model': model_list
    })
    df.to_csv(os.path.join(output_dir, 'DNABERT2_koo_results.csv'), index=False)
    print(f'Results saved incrementally')

# Final summary
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
df = pd.DataFrame({
    'TF': tf_list,
    'Accuracy': test_accuracy,
    'AUROC': test_auroc,
    'AUPR': test_aupr,
    'Model': model_list
})
print(df.groupby('TF')[['Accuracy', 'AUROC', 'AUPR']].mean())
print("\nOverall mean:")
print(df[['Accuracy', 'AUROC', 'AUPR']].mean())

print("\nDone!")
