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


def load_and_preprocess_image(img_list, height=164, width=397):
    """
    Load and preprocess the image from the given path.
    The image preprocessing includes resizing and normalization.
    Args:
        img_list (list): List of image file paths.
        height (int): Desired height of the output image.
        width (int): Desired width of the output image.
    
    Returns:
        numpy.ndarray: Preprocessed image as a numpy array.
    """
    images = []
    for f in img_list:
        label = os.path.basename(os.path.dirname(f))
        image = tf.io.read_file(f)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [height, width])
        image = tf.cast(image, tf.float32) / 255.0
        images.append({"label": label, "image": image.numpy()})
    return pd.DataFrame(images)

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
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if os.path.exists(cls_dir):
            files = [os.path.join(cls, f) for f in os.listdir(cls_dir) if f.endswith('.png')]
            tot_files.extend(files)
        else:
            print(f"Directory {cls_dir} does not exist. Skipping class {cls}.")
    perm = rng.permutation(len(tot_files))
    dir_list = [os.path.join(data_dir, f) for f in tot_files]

    if not os.path.exists(os.path.join(CNN_CACHE_DIR, "split_data.pkl")):
        print("Creating and caching data split...")
        for split in split_perc.keys():
            if split not in ['train', 'val', 'test']:
                raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'.")
            
            # Get all files for the split based on the split percentage
            split_len = floor(len(tot_files) * split_perc[split])
            files = [dir_list[i] for i in perm[:split_len]]
            perm = perm[split_len:]

            # Apply loading and preprocessing to each element of files list
            split_ds = load_and_preprocess_image(files, height=h, width=w)

            # Insert set into dictionary
            sets[split] = split_ds
        with open(os.path.join(CNN_CACHE_DIR, "split_data.pkl"), 'wb') as handle:
            pickle.dump(sets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading split data from cache...")
        with open(os.path.join(CNN_CACHE_DIR, "split_data.pkl"), 'rb') as handle:
            sets = pickle.load(handle)
        
    return sets
        
