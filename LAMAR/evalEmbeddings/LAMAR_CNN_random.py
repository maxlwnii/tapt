import pandas as pd
import numpy as np
from tqdm import tqdm
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
print(f"TensorFlow configured to use CPU only (efficient for small CNN)")

# Import PyTorch (will have access to GPU)
import torch
from transformers import AutoTokenizer, AutoConfig
from LAMAR.modeling_nucESM2 import EsmForMaskedLM

# Verify GPU availability for PyTorch (LAMAR)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available for PyTorch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("WARNING: CUDA not available for PyTorch! LAMAR will use CPU (slow)")

# LAMAR configuration
model_max_length = 512  # Adjust based on your sequence length
extraction_batch_size = 16  # Reduced for memory efficiency during embedding extraction
downstream_batch_size = 256 # Batch size for CNN training (matches NT paper)
datalen = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"LAMAR will use: {device}")
file_list = glob.glob('/home/fr/fr_fr/fr_ml642/Thesis/eclip/*.h5')

test_aupr = []
test_auroc = []
test_accuracy = []
tf_list = []


def chip_cnn(input_shape, output_shape):
    initializer = tf.keras.initializers.HeUniform(seed=42)
    input = keras.Input(shape=input_shape)

    # add batchnorm and dimension reduction
    nn = keras.layers.BatchNormalization()(input)
    nn = keras.layers.Conv1D(filters=512, kernel_size=1, kernel_initializer=initializer)(nn)
    # first conv layer
    nn = keras.layers.Conv1D(
        filters=64, kernel_size=7, padding='same', kernel_initializer=initializer
    )(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # Second conv layer
    nn = keras.layers.Conv1D(
        filters=96, kernel_size=5, padding='same', kernel_initializer=initializer
    )(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # third conv layer
    nn = keras.layers.Conv1D(
        filters=128, kernel_size=5, padding='same', kernel_initializer=initializer
    )(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(2)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # dense layer
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


earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6
)
auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)


# Random initialization helper

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


# Load LAMAR model config and randomly initialize weights
print('Loading LAMAR model with random initialization')
tokenizer = AutoTokenizer.from_pretrained(
    "/home/fr/fr_fr/fr_ml642/Thesis/pretrain/saving_model/tapt_lamar/checkpoint-100000/",
    model_max_length=model_max_length,
)
config = AutoConfig.from_pretrained(
    "/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/config/config_150M.json",
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    mask_token_id=tokenizer.mask_token_id,
    token_dropout=False,
    positional_embedding_type="rotary",
    hidden_size=768,
    intermediate_size=3072,
    num_attention_heads=12,
    num_hidden_layers=12,
)
lamar_model = EsmForMaskedLM(config)
lamar_model.apply(init_weights)
lamar_model.to(device)
lamar_model.eval()

# Extract embeddings from different layers
for layer in (11, 5):  # Layer 11 (last) and layer 5 (middle)
    test_aupr = []
    test_auroc = []
    test_accuracy = []
    tf_list = []
    model_list = []

    # per TF, load onehot data and generate embedding for train/valid/test set.
    for file in file_list:
        tf_name = file.split('/')[-1][:-12]
        data = h5py.File(file, 'r')
        print(f'Processing {tf_name} with LAMAR Layer {layer} (random init)')
        dataset = {}
        embeddings_dict = {}

        # create dataset for train,valid,test.
        for label in ['test', 'valid', 'train']:
            # Convert one-hot to sequences
            sequence = data['X_' + label][()]
            sequence = np.transpose(sequence, (0, 2, 1))
            # Convert one-hot (first 4 channels) to DNA strings
            seq_strings = []
            for seq in sequence:
                bases = ['A', 'C', 'G', 'T']
                # Only use the first 4 channels for the argmax to avoid picking extra features
                dna = ''.join([bases[np.argmax(pos[:4])] for pos in seq])
                seq_strings.append(dna)

            target = data['Y_' + label][()]

            # Get actual sequence length from first sequence
            actual_seq_len = len(seq_strings[0]) if seq_strings else model_max_length
            print(f'Extracting embeddings for {label} set ({len(seq_strings)} sequences, length={actual_seq_len})')

            # Pass through model for embeddings
            total_embed = []
            with torch.no_grad():
                for i in tqdm(range(0, len(seq_strings), extraction_batch_size)):
                    batch_seqs = seq_strings[i:i+extraction_batch_size]
                    # Tokenize
                    tokens = tokenizer(
                        batch_seqs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=model_max_length,
                    ).to(device)

                    # Get embeddings from specified layer
                    outputs = lamar_model.esm(
                        **tokens,
                        output_hidden_states=True,
                    )
                    layer_embeddings = outputs.hidden_states[layer + 1]
                    batch_embeddings = layer_embeddings.cpu().numpy()
                    total_embed.extend(batch_embeddings)

            # Store embeddings for this split
            embeddings_dict[label] = total_embed

            # create TF dataset from embeddings
            with tf.device("CPU"):
                dataset[label] = (
                    tf.data.Dataset.from_tensor_slices((total_embed, data['Y_' + label][()]))
                    .shuffle(256 * 4)
                    .batch(downstream_batch_size)
                )

        # Get actual input length from first embedding (use train set)
        actual_input_len = embeddings_dict['train'][0].shape[0]
        print(f'Embedding shape: ({actual_input_len}, 768)')

        print(f'Training CNN for {tf_name} with LAMAR Layer {layer} embeddings (random init)')
        for i in range(5):  # 5 repeats for robustness
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            os.makedirs('/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/models', exist_ok=True)
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                f'/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/models/random_layer{layer}_{tf_name}_rep{i}.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                save_freq='epoch',
            )
            # Input shape: actual_seq_len (from data) x LAMAR hidden_size (768)
            print(f'Creating CNN with input shape: ({actual_input_len}, 768)')
            cnn_model = chip_cnn((actual_input_len, 768), 1)
            cnn_model.compile(loss=loss, metrics=['accuracy', auroc, aupr], optimizer=optimizer)

            result = cnn_model.fit(
                dataset['train'],
                validation_data=dataset['valid'],
                epochs=100,
                verbose=2,
                callbacks=[earlyStopping_callback, reduce_lr, checkpoint],
            )
            _, acc, roc, pr = cnn_model.evaluate(dataset['test'])
            tf_list.append(tf_name)
            test_accuracy.append(acc)
            test_auroc.append(roc)
            test_aupr.append(pr)
            model_list.append(f'LAMAR Layer{layer} CNN (random)')

    df = pd.DataFrame(
        list(zip(tf_list, test_accuracy, test_auroc, test_aupr, model_list)),
        columns=['TF', 'Accuracy', 'AUROC', 'AUPR', 'Model'],
    )
    os.makedirs('/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/results', exist_ok=True)
    df.to_csv(
        f'/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/results/LAMAR_layer{layer}_random_perf.csv'
    )
    print(f'Results saved for layer {layer} (random init)')
    print(df.groupby('Model')[['Accuracy', 'AUROC', 'AUPR']].mean())
