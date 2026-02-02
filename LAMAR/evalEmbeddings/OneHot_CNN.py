import pandas as pd
import numpy as np
import sys
import h5py
import glob
import os

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
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found for TensorFlow, utilizing CPU/System Ram only.")

# Configuration
batch_size = 256
file_list = glob.glob('/home/fr/fr_fr/fr_ml642/Thesis/eclip/*.h5')

test_aupr = []
test_auroc = []
test_accuracy = []
tf_list = []
model_list = []

def get_baseline_cnn_onehot(input_shape, output_shape=1):
    """
    Implements the Baseline CNN from the Methods section:
    "For one-hot sequences, batch-norm and the convolution with kernel 1 were not employed."
    
    Structure:
    3. Conv1D (64, size 7, same) -> BN -> ReLU -> MaxPool(4) -> Dropout(0.2)
    4. Conv1D (96, size 5, same) -> BN -> ReLU -> MaxPool(4) -> Dropout(0.2)
    5. Conv1D (128, size 5, same) -> BN -> ReLU -> MaxPool(2) -> Dropout(0.2)
    6. Flatten
    7. Dense (256) -> BN -> ReLU -> Dropout(0.5)
    8. Output (1) -> Sigmoid
    """
    initializer = tf.keras.initializers.HeUniform(seed=42)
    input_layer = keras.Input(shape=input_shape)
    
    # Layer 3: Conv 64
    nn = keras.layers.Conv1D(filters=64,
                             kernel_size=7,
                             padding='same',
                             kernel_initializer=initializer)(input_layer)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # Layer 4: Conv 96
    nn = keras.layers.Conv1D(filters=96,
                             kernel_size=5,
                             padding='same',
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # Layer 5: Conv 128
    nn = keras.layers.Conv1D(filters=128,
                             kernel_size=5,
                             padding='same',
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(2)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # Layer 6: Flatten
    nn = keras.layers.Flatten()(nn)

    # Layer 7: Dense 256
    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    # Layer 8: Output
    logits = keras.layers.Dense(output_shape)(nn)
    output = keras.layers.Activation('sigmoid')(logits)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    return model

# Callbacks
earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2,
            patience=5, min_lr=1e-6)
auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)

# Main loop
for file in file_list:
    # Extract TF name
    filename = file.split('/')[-1]
    tf_name = filename[:-3] if filename.endswith(".h5") else filename
    
    print(f'Processing {tf_name} with Baseline One-Hot CNN')
    
    data = h5py.File(file, 'r')
    dataset = {}
    
    # helper to prepare data
    def load_data(label):
        # Load data. One-hot should be (N, L, 4) for Conv1D. 
        # Checking previous files, data might be (N, C, L) or (N, L, C).
        # We'll read it and check shape.
        key_x = f'X_{label}'
        key_y = f'Y_{label}'
        
        # Try uppercase keys first, falling back if needed (based on previous exploration keys were uppercase)
        if key_x not in data:
            print(f"Warning: {key_x} not found, trying lowercase.")
            key_x = f'x_{label}'
            key_y = f'y_{label}'
            
        x = data[key_x][()] # Returns numpy array
        y = data[key_y][()]
        
        # Handle shape: if (N, 4, 200) or similar (channels first), transpose to (N, 200, 4)
        if x.shape[1] == 4 or x.shape[1] == 9: 
             # likely channels first
             x = np.transpose(x, (0, 2, 1))
        
        # Keep only first 4 channels (A, C, G, T)
        x = x[:, :, :4]
        x = x.astype(np.float32)
        
        return x, y

    x_train, y_train = load_data('train')
    x_valid, y_valid = load_data('valid')
    x_test, y_test = load_data('test')

    print(f"Data shape for train: {x_train.shape}")

    # Create TF datasets
    with tf.device("CPU"):
        dataset['train'] = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(256*4).batch(batch_size)
        dataset['valid'] = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).shuffle(256*4).batch(batch_size)
        dataset['test'] = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    input_shape = x_train.shape[1:]
    
    for i in range(5):  # 5 repeats
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Save path
        save_path = f'/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/models_onehot/{tf_name}_rep{i}.h5'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                        save_path,
                                        monitor='val_loss',
                                        save_best_only=True,
                                        mode = 'min',
                                        save_freq='epoch',)
        
        cnn_model = get_baseline_cnn_onehot(input_shape, 1)
        cnn_model.compile(loss = loss,
                    metrics=['accuracy',auroc,aupr],
                    optimizer=optimizer)
        
        result = cnn_model.fit(dataset['train'],
            validation_data=dataset['valid'],
            epochs=100,
            verbose=2,
            callbacks=[earlyStopping_callback,reduce_lr,checkpoint]
        )
        _, acc, roc, pr = cnn_model.evaluate(dataset['test'])
        
        tf_list.append(tf_name)
        test_accuracy.append(acc)
        test_auroc.append(roc)
        test_aupr.append(pr)
        model_list.append('One-Hot CNN')
        print(f"Finished {tf_name} rep {i} - AUROC: {roc:.4f}")

# Save results
df = pd.DataFrame(list(zip(tf_list, test_accuracy, test_auroc, test_aupr, model_list)),
           columns =['TF','Accuracy','AUROC','AUPR','Model'])
output_dir = '/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/results'
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, 'OneHot_perf.csv'))

print(f'Results saved to OneHot_perf.csv')
print(df.groupby('Model')[['Accuracy', 'AUROC', 'AUPR']].mean())
