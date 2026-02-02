import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
import os
import sys

# Configuration
output_dir = '/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/results/'
os.makedirs(output_dir, exist_ok=True)
model_name = 'OneHot_CNN'

file_list = glob.glob('/home/fr/fr_fr/fr_ml642/Thesis/eclip/*.h5')
print('Number of tasks detected : %d'%(len(file_list)))

test_aupr = []
test_auroc = []
test_accuracy = []
tf_list = []
model_list = []

def chip_cnn(input_shape,output_shape):
    initializer = tf.keras.initializers.HeUniform()
    input = keras.Input(shape=input_shape)
    
    # In representation_perf.py, the projection layer (512) was calculated but overwritten.
    # We implement the effective architecture (Input -> Conv64).
    
    #first conv layer
    nn = keras.layers.Conv1D(filters=64,
                             kernel_size=7,
                             padding = 'same',
                             kernel_initializer=initializer)(input)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    #Second conv layer
    nn = keras.layers.Conv1D(filters=96,
                             kernel_size=5,
                             padding = 'same',
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    #third conv layer
    nn = keras.layers.Conv1D(filters=128,
                             kernel_size=5,
                             padding = 'same',
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(2)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    #dense layer
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    #Output layer
    logits = keras.layers.Dense(output_shape)(nn)
    output = keras.layers.Activation('sigmoid')(logits)
    
    model = keras.Model(inputs=input, outputs=output)
    return model

earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2,
            patience=5, min_lr=1e-6)
auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)

#loop through TFs
for file in file_list:
    tf_name = file.split('/')[-1][:-3] # Remove .h5

    #load dataset into TF dataset form
    data = h5py.File(file,'r')
    
    # Helper to load and slice data (N, L, 9) -> (N, L, 4)
    # Note: LAMAR script implied transposition might be needed depending on file version,
    # but based on keys check, it seems we have X_train.
    # We will assume (N, L, 9) or (N, 9, L). 
    # Let's check shape of first batch in loop or assume standard from representation_perf.py context which didn't transpose but LAMAR did.
    # LAMAR script: sequence = data['X_'+label][()] -> sequence = np.transpose(sequence,(0,2,1))
    # This implies stored as (N, Channel, Length).
    # Tensorflow Conv1D expects (N, Length, Channel).
    # So we MUST transpose if stored as (N, C, L).
    
    def prepare_data(key_x, key_y):
        x = data[key_x][()]
        y = data[key_y][()]
        # Transpose (N, C, L) -> (N, L, C)
        x = np.transpose(x, (0, 2, 1))
        # Slice to first 4 channels
        x = x[:, :, :4]
        return x, y

    x_train, y_train = prepare_data('X_train', 'Y_train')
    x_valid, y_valid = prepare_data('X_valid', 'Y_valid')
    x_test, y_test = prepare_data('X_test', 'Y_test')

    with tf.device("CPU"):
        trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(256*4).batch(256)
        validset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).shuffle(256*4).batch(256)
        testset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(256*4).batch(256)
    
    #model compile and training
    print('########Training ',tf_name, '########')
    print(f'Input shape: {x_train.shape[1:]}')
    
    for i in range(5):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        checkpoint_path = os.path.join('/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/models_onehot_rep', f'{tf_name}_rep{i}.h5')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                        checkpoint_path,
                                        monitor='val_loss',
                                        save_best_only=True,
                                        mode = 'min',
                                        save_freq='epoch',)
        
        model = chip_cnn((x_train.shape[1:]),1)
        model.compile(loss = loss,
                    metrics=['accuracy',auroc,aupr],
                    optimizer=optimizer)
        
        result = model.fit(trainset,
            batch_size=256,
            validation_data=validset,
            epochs=100,
            verbose=2,
            callbacks=[earlyStopping_callback,reduce_lr,checkpoint]
        )
        _, acc, roc, pr = model.evaluate(testset)
        tf_list.append(tf_name)
        test_accuracy.append(acc)
        test_auroc.append(roc)
        test_aupr.append(pr)
        model_list.append(model_name)

### collect result into csv
df = pd.DataFrame(list(zip(tf_list, test_accuracy, test_auroc, test_aupr,model_list)),
               columns =['TF','Accuracy','AUROC','AUPR','Model'])

df.to_csv(os.path.join(output_dir, 'OneHot_Rep_perf.csv'))
print("Done.")
