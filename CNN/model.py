import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf
from math import floor
import keras


CACHE_DIR = os.path.join("data_cache", "CNN")
seed = 2025
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Configurazione GPU
def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"GPUs found: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu}")
                                
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found, using CPU")

setup_gpu()

@keras.saving.register_keras_serializable()
class HandmadeAccuracy(keras.metrics.Metric):
    def __init__(self, name="accuracy_handmade", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int64)

        matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)

        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return {'tp_hm':self.correct, 'total':self.total}

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)



def calculate_confusion_matrix(y_true, y_pred, num_classes):
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    
    cm = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=num_classes,
        dtype=tf.float32
    )
    return cm

class ConfusionMatrixLogger(keras.callbacks.Callback):
    def __init__(self, validation_dataset, num_classes):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.num_classes = num_classes

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nCalculating confusion matrix for epoch {epoch + 1}")
        
        all_y_true = []
        all_y_pred = []

        for x_batch, y_batch in self.validation_dataset:
            y_pred_batch = self.model.predict_on_batch(x_batch)
            
            all_y_true.append(y_batch)
            all_y_pred.append(y_pred_batch)

        y_true = tf.concat(all_y_true, axis=0)
        y_pred = tf.concat(all_y_pred, axis=0)
        
        cm = calculate_confusion_matrix(y_true, y_pred, self.num_classes)
        logs['val_confusion_matrix'] = cm.numpy().astype(int)
        
        print(f"Validation Confusion Matrix:\n{cm.numpy().astype(int)}")


def create_model(w_h, n_classes):
    initializer = tf.keras.initializers.GlorotNormal(seed=42)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(w_h[1], w_h[0], 3)))
    model.add(tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu', kernel_initializer=initializer))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer=initializer))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', kernel_initializer=initializer))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer=initializer))
    return model

def create_model_off(w_h, n_classes):
    initializer = tf.keras.initializers.GlorotNormal(seed=2025)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(w_h[1], w_h[0], 3))
    ])
    lb = min(w_h)
    while floor(lb / 2) >= 1:
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer=initializer))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.MaxPool2D((2, 2)))
        lb = lb / 2
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer=initializer))
    return model

def train(train_df, valid_df, patience, cp_path, w_h, n_classes, class_weight_dict, checkpoint_freq=5):
    """
    Train a CNN model using the provided training and validation datasets.
    Args:
        train_dataset (DataFrame): The training dataset.
        valid_dataset (DataFrame): The validation dataset.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        cp_path (str): Subdirectory to save model checkpoints.
        w_h (tuple): Width and height for the model input.
        n_classes (int): Number of output classes.
        checkpoint_freq (int): Frequency (in epochs) to save model checkpoints.
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    cp_dir = os.path.join(CACHE_DIR, cp_path)
    if not os.path.exists(cp_dir):
        os.makedirs(cp_dir)
    model = create_model(w_h, n_classes)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(),
        metrics=['accuracy', HandmadeAccuracy()]
    )
    print(valid_df)
    es_cb = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, mode="max", min_delta=0.001, restore_best_weights=True)
    cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(cp_dir, 'recovery_weights.weights.h5'), save_weights_only=True, save_freq=checkpoint_freq)
    logger = tf.keras.callbacks.CSVLogger(os.path.join(CACHE_DIR, str(n_classes) + '_training_log.csv'), append=True)
    cm_cb = ConfusionMatrixLogger(validation_dataset=valid_df, num_classes=n_classes)
    print("training")

    history = model.fit(train_df, 
                        epochs=2, 
                        validation_data=valid_df, 
                        callbacks=[es_cb, cp_cb, cm_cb, logger],
                        class_weight=class_weight_dict
                        )
    
    print(history.history)
    model.save(os.path.join(CACHE_DIR, str(n_classes) + '_final_model.h5'))


