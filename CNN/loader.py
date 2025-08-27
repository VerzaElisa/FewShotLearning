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

def _bytes_feature(value):
    """
    Convert a value to a type compatible with tf.train.Example.
    Args:
        value: A value to be converted (string, bytes, or tf.Tensor).
    Returns:
        tf.train.Feature: Returns a bytes_list.
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """
    Convert a value to a type compatible with tf.train.Example.
    Args:
        value: A value to be converted.
        
    Returns:
        tf.train.Feature: Returns an int64_list.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_tfrecord(img_list, class_names, output_path):
    """
    Write images and labels to a TFRecord file.
    Args:
        img_list (list): List of image file paths.
        class_names (list): List of class names corresponding to labels.
        output_path (str): Path to save the TFRecord file.
    """
    if not os.path.exists(output_path):
        with tf.io.TFRecordWriter(output_path) as writer:
            for f in img_list:
                # Read image
                image = tf.io.read_file(f)

                # Extract label from the folder name, assigning an integer based on class_names list index
                label_name = os.path.basename(os.path.dirname(f))
                label = class_names.index(label_name)

                # Create Example
                feature = {
                    "image_raw": _bytes_feature(image),
                    "label": _int64_feature(label),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        print(f"TFRecord saved in {output_path}")


def _parse_function(proto, height=164, width=397):
    """
    Parse a single TFRecord example. Applies image decoding and preprocessing.
    Args:
        proto: A serialized TFRecord example.
        height (int): Desired height of the output image.
        width (int): Desired width of the output image.
    Returns:
        tuple: A tuple containing the preprocessed image and its label."""
    # Data schema
    keys_to_features = {
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(proto, keys_to_features)

    # Image processing
    image = tf.image.decode_png(parsed["image_raw"], channels=3)
    image = tf.image.resize(image, [height, width])
    image = tf.cast(image, tf.float32) / 255.0

    return image, parsed["label"]

def load_tfrecord_dataset(tfrecord_path, batch_size=32, height=164, width=397, shuffle=True):
    """
    Load a TFRecord dataset and apply preprocessing and batching.
    Args:
        tfrecord_path (str): Path to the TFRecord file.
        batch_size (int): Size of the batches.
        height (int): Desired height of the output images.
        width (int): Desired width of the output images.
        shuffle (bool): Whether to shuffle the dataset.
    Returns:
        tf.data.Dataset: A TensorFlow dataset ready for training or evaluation.
    """
    ds = tf.data.TFRecordDataset(tfrecord_path)

    ds = ds.map(lambda x: _parse_function(x, height, width), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

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

        dataset_path = os.path.join(CNN_CACHE_DIR, f"{split}_data.tfrecord")
        print(f"lunghezza files_path: {len(files_path)}")
        print(f"numero classi: {len(classes)}")
        write_tfrecord(files_path, classes, output_path=dataset_path)
        split_ds = load_tfrecord_dataset(dataset_path, height=h, width=w)

        # Insert set into dictionary
        sets[split] = split_ds
        print(f"{split} set processed and added to the dictionary.")
    return sets
        
