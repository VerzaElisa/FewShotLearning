import os
import numpy as np
import tensorflow as tf
import pandas as pd

np.random.seed(2025)
tf.random.set_seed(2025)

CNN_CACHE_DIR = os.path.join("data_cache", "CNN")
if not os.path.exists(CNN_CACHE_DIR):
    print(f"Creating CNN data cache directory at {CNN_CACHE_DIR}")
    os.makedirs(CNN_CACHE_DIR)


def load_and_preprocess_image(img_path, height=164, width=397):
    """
    Load and preprocess the image from the given path.
    The image preprocessing includes resizing and normalization.
    Args:
        img_path (str): Path to the image file.
        height (int): Desired height of the output image.
        width (int): Desired width of the output image.
    
    Returns:
        numpy.ndarray: Preprocessed image as a numpy array.
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [height, width])
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()

def get_set_files(data_dir, classes):
    """
    Get the files for each class in the dataset.
    Args:
        data_dir (str): Path to the full dataset.
        classes (list): List of class names.
    
    Returns:
        DataFrame: DataFrame containing class names and their corresponding file paths.
    """
    data = {}
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if os.path.exists(cls_dir):
            files = [f for f in os.listdir(cls_dir) if f.endswith('.png')]
            data[cls] = files
        else:
            print(f"Directory {cls_dir} does not exist. Skipping class {cls}.")
    
    df = pd.DataFrame({
        "key": list(data.keys()),
        "values": list(data.values())
    })
    df = df.explode("values", ignore_index=True)
    df.rename(columns={"key": "label", "values": "file"}, inplace=True)
    df['file'] = df['file'].apply(lambda x: os.path.join(data_dir, x))

    return df


def get_sets(data_dir, dataset_split, h, w):
    """
    Get the training, validation and test sets from the dataset.
    Args:
        data_dir (str): Path to the full dataset.
        dataset_split (DataFrame): Dataframe containing class names with their corresponding set.
                                   The DataFrame should have columns 'set' and 'file'.
        h (int): Desired height of the output images.
        w (int): Desired width of the output images.
    Returns:
        dict: Dictionary containing training, validation and test sets.
    """
    sets = {}
    for split in dataset_split['set'].unique():
        ds_df_dir = os.path.join(CNN_CACHE_DIR, f"ds_{split}.pkl")

        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'.")
        
        if not os.path.exists(ds_df_dir):
            set_ds = get_set_files(data_dir, dataset_split.unique())
            set_ds['file'] = set_ds['file'].apply(lambda x: os.path.join(data_dir, x))
            set_ds['file'] = set_ds['file'].apply(lambda x: load_and_preprocess_image(x, height=h, width=w))
            set_ds.to_pickle(ds_df_dir)
            print(f"Data saved to {ds_df_dir}")
        else:
            set_ds = pd.read_pickle(ds_df_dir)
            print(f"Data loaded from {ds_df_dir}")

        sets[split] = set_ds['file']
        
