import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf
from math import floor


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


class ReportCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_dataset, output_path="CNN_metrics.csv"):
        super().__init__()
        self.valid_dataset = valid_dataset
        self.output_path = output_path
        self.reports = [] 

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.valid_dataset, verbose=0)
        y_true = np.concatenate([y for x, y in self.valid_dataset], axis=0)
        y_pred = np.argmax(preds, axis=1)

        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)


        flat_report = {"epoch": epoch+1}
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                for m_name, value in metrics.items():
                    flat_report[f"{label}_{m_name}"] = value
            else:
                flat_report[label] = metrics

        self.reports.append(flat_report)

        mode = 'a' if epoch > 0 else 'w'
        header = False if epoch > 0 else True
        
        df = pd.DataFrame([flat_report])
        df.to_csv(self.output_path, mode=mode, header=header, index=False)
        
def create_model_off(w_h, n_classes):
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

def create_model(w_h, n_classes):
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
        metrics=['accuracy'],
    )
    es_cb = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, mode="max", min_delta=0.001, restore_best_weights=True)
    cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(cp_dir, 'recovery_weights.weights.h5'), save_weights_only=True, save_freq=checkpoint_freq)
    log_cb = ReportCallback(valid_df, output_path=os.path.join(CACHE_DIR, str(n_classes) + '_CNN_metrics.csv'))
    print("training")

    history = model.fit(train_df, 
                        epochs=50, 
                        validation_data=valid_df, 
                        callbacks=[es_cb, cp_cb, log_cb],
                        class_weight=class_weight_dict
                        )
    print(history.history)
    model.save(os.path.join(CACHE_DIR, str(n_classes) + '_final_model.h5'))


