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

def load_and_preprocess_image(img_path, sizes):
    """
    Load and return preprocessed image.
    Args:
        img_path (str): path to the image on disk.
        sizes (dict): dictionary with keys 'h', 'w' for height and width.
    Returns (Tensor): preprocessed image
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [sizes['h'], sizes['w']])
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def get_split(data_dir, classes, split_perc, h, w, batch_size=16):
    rng = random.Random(seed)
    train_dict = {}
    val_dict = {}
    label_to_index_dict = {}

    # retrieve filepaths for all classes and split in train and val set
    for i, c in enumerate(classes):
        img_list = os.listdir(os.path.join(data_dir, c))
        img_list = [os.path.join(data_dir, c, img_path) for img_path in img_list]
        rng.shuffle(img_list)
        index = floor(len(img_list)*(split_perc['train']))
        train_list = img_list[:index]
        val_list = img_list[index:]
        if c == 'other':
            train_list = [os.path.realpath(img_path) for img_path in train_list]
            val_list = [os.path.realpath(img_path) for img_path in val_list]
        train_dict[i] = train_list
        val_dict[i] = val_list
        label_to_index_dict[c] = i

    # save label mapping
    label_df = pd.DataFrame(label_to_index_dict.items(), columns=['label', 'index'])
    label_df.to_csv(os.path.join(CNN_CACHE_DIR, 'label_to_index.csv'), index=False)

    # create train and val dataframes
    train_df = pd.DataFrame(train_dict.items(), columns=['label', 'filenames'])
    train_df = train_df.explode('filenames').reset_index(drop=True)

    val_df = pd.DataFrame(val_dict.items(), columns=['label', 'filenames'])
    val_df = val_df.explode('filenames').reset_index(drop=True)

    train_df.to_csv(os.path.join(CNN_CACHE_DIR, 'train_split.csv'), index=False)
    # from dataframe to tensors with image loading and preprocessing
    train_tensor = tf.data.Dataset.from_tensor_slices((train_df["filenames"].values, train_df["label"].values))
    train_tensor = train_tensor.map(lambda x, y: (load_and_preprocess_image(x, {'h': h, 'w': w}), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_tensor = train_tensor.shuffle(buffer_size=len(train_df), seed=2025)
    
    val_tensor = tf.data.Dataset.from_tensor_slices((val_df["filenames"].values, val_df["label"].values))
    val_tensor = val_tensor.map(lambda x, y: (load_and_preprocess_image(x, {'h': h, 'w': w}), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_tensor = train_tensor.batch(batch_size)
    val_tensor   = val_tensor.batch(batch_size)

    train_tensor = train_tensor.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_tensor   = val_tensor.prefetch(buffer_size=tf.data.AUTOTUNE)

    return {'train': train_tensor, 'val': val_tensor}

def get_test(data_dir, mapping_dir, classes, h, w, batch_size=16):
    rng = random.Random(seed)
    test_dict = {}
    mapping_df = pd.read_csv(mapping_dir)
    mapping_dict = dict(zip(mapping_df['label'], mapping_df['index']))

    # retrieve filepaths for all classes and split in test set
    for c in classes:
        if c in mapping_dict.keys():
            img_list = os.listdir(os.path.join(data_dir, c))
            img_list = [os.path.join(data_dir, c, img_path) for img_path in img_list]
            rng.shuffle(img_list)
            if c == 'other':
                img_list = [os.path.realpath(img_path) for img_path in img_list]
            i = mapping_dict[c]
            test_dict[i] = img_list
        else:
            continue

    # create test dataframes
    test_df = pd.DataFrame(test_dict.items(), columns=['label', 'filenames'])
    test_df = test_df.explode('filenames').reset_index(drop=True)

    test_df.to_csv(os.path.join(CNN_CACHE_DIR, 'test_split.csv'), index=False)

    # from dataframe to tensors with image loading and preprocessing
    test_tensor = tf.data.Dataset.from_tensor_slices((test_df["filenames"].values, test_df["label"].values))
    test_tensor = test_tensor.map(lambda x, y: (load_and_preprocess_image(x, {'h': h, 'w': w}), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    test_tensor = test_tensor.batch(batch_size)
    test_tensor = test_tensor.prefetch(buffer_size=tf.data.AUTOTUNE)

    return test_tensor

def get_split_off(data_dir, classes, split_perc, h, w, batch_size=16):  

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
    print(type(train_ds))
    class_names = pd.DataFrame({'label': train_ds.class_names, 'index': range(len(train_ds.class_names))})
    class_names.to_csv(os.path.join(CNN_CACHE_DIR, 'label_to_index.csv'), index=False)
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return {'train': train_ds, 'val': val_ds}
