import os
import numpy as np
import tensorflow as tf
import pandas as pd
from math import floor
import pickle

np.random.seed(2025)
tf.random.set_seed(2025)
rng = np.random.default_rng(seed = 2025)

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

def preprocess_image(file_path, height=164, width=397):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [height, width])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def load_dataset(img_list, height=164, width=397, batch_size=32, shuffle=True):
    labels = [os.path.basename(os.path.dirname(f)) for f in img_list]

    ds = tf.data.Dataset.from_tensor_slices((img_list, labels))

    def _process(file_path, label):
        image = preprocess_image(file_path, height, width)
        return image, label   # <-- meglio restituire (immagine, etichetta) per Keras

    ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_list))

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def get_split(data_dir, classes, split_perc, h, w):
    """
    Get the training, validation and test sets from the dataset.
    Args:
        data_dir (str): Path to the full dataset.
        classes (list): List of class names to be considered.
        split_perc (dict): Dictionary containing the percentage split for train, val, and test.
        h (int): Desired height of the output images.
        w (int): Desired width of the output images.
    Returns:
        dict: Dictionary containing training, validation and test sets.
    """
    sets = {}
    tot_files = []

    # List of all files in the dataset
    print("Listing all image files path...")
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if os.path.exists(cls_dir):
            curr_files = [os.path.join(cls, f) for f in os.listdir(cls_dir) if f.endswith('.png')]
            tot_files.extend(curr_files)
        else:
            print(f"Directory {cls_dir} does not exist. Skipping class {cls}.")
    perm = rng.permutation(len(tot_files))
    dir_list = [os.path.join(data_dir, f) for f in tot_files]

    print("Creating and caching data split...")
    for split in split_perc.keys():
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'.")
        
        # Get all files for the split based on the split percentage
        split_len = floor(len(tot_files) * split_perc[split])
        files_path = [dir_list[i] for i in perm[:split_len]]
        perm = perm[split_len:]

        # Apply loading and preprocessing to each element of files list
        print(f"Processing {split} set with {len(files_path)} images...")
        print(f"lunghezza files_path: {len(files_path)}")
        print(f"numero classi: {len(classes)}")

        split_ds = load_dataset(files_path, height=h, width=w)

        # Insert set into dictionary
        sets[split] = split_ds
        print(f"{split} set processed and added to the dictionary.")
    return sets
        
