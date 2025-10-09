import os
import numpy as np
import tensorflow as tf
import pandas as pd
from math import floor
import csv
import random
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

seed = 2025
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
rng = np.random.default_rng(seed = seed)

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

CNN_CACHE_DIR = os.path.join("data_cache", "CNN")
if not os.path.exists(CNN_CACHE_DIR):
    print(f"Creating CNN data cache directory at {CNN_CACHE_DIR}")
    os.makedirs(CNN_CACHE_DIR)

def get_split(data_dir, classes, split_perc, h, w, batch_size=32):  
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=split_perc['val'], 
        subset="training", 
        seed=2025,
        class_names=classes,
        image_size=(h, w),
        batch_size=batch_size,
        follow_links=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=split_perc['val'],
        subset="validation",
        seed=2025,
        class_names=classes,
        image_size=(h, w),
        batch_size=batch_size,
        follow_links=True
    )
    class_names = pd.DataFrame({'label': train_ds.class_names, 'index': range(len(train_ds.class_names))})
    class_names.to_csv(os.path.join(CNN_CACHE_DIR, 'label_to_index.csv'), index=False)
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return {'train': train_ds, 'val': val_ds}
