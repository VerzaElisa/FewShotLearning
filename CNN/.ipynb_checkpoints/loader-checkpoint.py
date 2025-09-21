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

def preprocess_image(file_path, height=164, width=397):
    """
    Load and preprocess an image from a file path.
    Args:
        file_path (str): Path to the image file.
        height (int): Desired height of the output image.
        width (int): Desired width of the output image.
    Returns:
        tf.Tensor: A preprocessed image tensor.
    """
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [height, width])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def load_dataset(img_list, height, width, is_train, batch_size=32):
    """
    From a list of image paths, create a dataset with 
    preprocessed and batched images alongside its labels.
    Args:
        img_list (list): List of image file paths.
        height (int): Desired height of the output images.
        width (int): Desired width of the output images.
        is_train (bool): Indicates if the dataset is for training.
        batch_size (int): Size of the batches of data.
    Returns:
        tf.data.Dataset: A TensorFlow Dataset object containing the preprocessed and batched images.
        dict: A dictionary containing class weights if is_train is True.
    """
    print(f'Loading dataset with {len(img_list)} images...')
    class_weight_dict = None

    labels = [os.path.basename(os.path.dirname(f)) for f in img_list]
    unique_labels = sorted(set(labels))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    labels = [label_to_index[label] for label in labels]
    unique_labels_numeric = np.array([label_to_index[label] for label in unique_labels])
    if is_train:
        with open(os.path.join(CNN_CACHE_DIR, str(len(unique_labels))+'_label_to_index.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['label', 'index'])
            for label, index in label_to_index.items():
                writer.writerow([label, index])
        
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels_numeric, y=labels)
        class_weight_dict = dict(zip(unique_labels_numeric, class_weights))

    ds = tf.data.Dataset.from_tensor_slices((img_list, labels))  

    ds = ds.map(lambda file_path, label: (preprocess_image(file_path, height, width), label), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=1000, seed=seed)

    if batch_size is not None:
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, class_weight_dict 

def get_img_list(data_dir, classes):
    tot_files = []
    print("Listing all image files path...")
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if os.path.exists(cls_dir):
            curr_files = [os.path.join(cls, f) for f in os.listdir(cls_dir) if f.endswith('.png')]
            print(f"Found {len(curr_files)} images for class {cls}.")
            tot_files.extend(curr_files)
        else:
            print(f"Directory {cls_dir} does not exist. Skipping class {cls}.")
    perm = rng.permutation(len(tot_files))
    dir_list = [os.path.join(data_dir, f) for f in tot_files]
    return dir_list, perm, tot_files

def get_split(data_dir, classes, split_perc, h, w, batch_size=32):  
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

    # List of all files in the dataset
    dir_list, perm, tot_files = get_img_list(data_dir, classes)
    cw = None
    print("Creating and caching data split...")
    for split in split_perc.keys():
        train = True if split == 'train' else False
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'.")
        
        # Get all files for the split based on the split percentage
        n_elem = floor(len(tot_files) * split_perc[split])
        split_len = n_elem if n_elem > 0 else 1
        files_path = [dir_list[i] for i in perm[:split_len]]
        perm = perm[split_len:]

        # Apply loading and preprocessing to each element of files list
        print(f"Processing {split} set with {len(files_path)} images...")
        if batch_size is not None:
            bs = batch_size if len(files_path) >= batch_size else len(files_path)
        else:
            bs = None
        split_ds, class_weight_dict = load_dataset(files_path, height=h, width=w, batch_size=bs, is_train=train)

        # Insert set into dictionary
        sets[split] = split_ds
        print(f"{split} set processed and added to the dictionary.")
        cw = class_weight_dict if train else cw
    return sets, cw
        
