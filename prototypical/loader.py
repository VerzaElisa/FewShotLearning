import os
import numpy as np
import tensorflow as tf
import pandas as pd

rng = np.random.default_rng(seed = 2025)

DATA_CACHE_DIR = os.path.join("data_cache")
if not os.path.exists(DATA_CACHE_DIR):
    print(f"Creating data cache directory at {DATA_CACHE_DIR}")
    os.makedirs(DATA_CACHE_DIR)


class DataLoader(object):
    def __init__(self, data, classes, n_way, n_support, n_query, img_dims={'h': 164, 'w': 397, 'c': 3}):
        self.data = data
        self.n_way = n_way
        self.classes = classes
        self.n_support = n_support
        self.n_query = n_query
        self.h = img_dims['h']
        self.w = img_dims['w']
        self.c = img_dims['c']

    def get_next_episode(self):

        # Randomly select classes for the episode, the intersection of episodes's classes may be not empty
        episode_cls = rng.permutation(self.classes)[:self.n_way]
        episode_cls_list = [cls.split('/')[-1] for cls in episode_cls]
        curr_files = self.data[self.data['label'].isin(episode_cls_list)]
        print(f'episode classes: {episode_cls_list}, file len {curr_files.shape}')

        support = curr_files.groupby("label", group_keys=False).apply(lambda x: x.sample(n=self.n_support))
        query = curr_files.drop(support.index).groupby("label", group_keys=False).apply(lambda x: x.sample(n=self.n_query))
        print(f'train support len {support.shape[0]}, train query len {query.shape[0]}')

        label_mapping = {label: idx for idx, label in enumerate(episode_cls_list)}
        support["label"] = support["label"].map(label_mapping)
        query["label"] = query["label"].map(label_mapping)
 
        support["file"] = support["file"].apply(lambda x, height=self.h, width=self.w: load_and_preprocess_image(x, height=height, width=width))
        query["file"] = query["file"].apply(lambda x, height=self.h, width=self.w: load_and_preprocess_image(x, height=height, width=width))

        return support.reset_index(drop=True), query.reset_index(drop=True), label_mapping
    
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


def configuration(config, split, classes):
    """
    Configuration for the DataLoader.
    Args:
        config (dict): input dict with model workflow params.
        split (str): 'train'|'val'|'test'
        classes (list): list of classes path.
    Returns:
        n_way (int): number of classes per episode.
        n_support (int): number of support examples per class.
        n_query (int): number of query examples per class.
        class_names (list): list of all class names.
        records (DataFrame): dataframe of filepaths and labels.
    """
    # n_way (number of classes per episode)
    if split in ['val', 'test']:
        n_way = config['data.test_way']
    else:
        n_way = config['data.train_way']

    # n_support (number of support examples per class)
    if split in ['val', 'test']:
        n_support = config['data.test_support']
    else:
        n_support = config['data.train_support']

    # n_query (number of query examples per class)
    if split in ['val', 'test']:
        n_query = config['data.test_query']
    else:
        n_query = config['data.train_query']
    
    # Create records DataFrame with complete file paths and labels
    records = pd.DataFrame(columns=["file", "label"])
    for class_name in classes:
        class_files = [os.path.join(class_name, f) for f in os.listdir(class_name) if f.endswith('.png')]
        class_labels = [os.path.basename(class_name)] * len(class_files)
        class_records = pd.DataFrame({"file": class_files, "label": class_labels})
        records = pd.concat([records, class_records], ignore_index=True)
    return n_way, n_support, n_query, records

def load(data_dirs, config, splits):
    """
    Load dataset.

    Args:
        data_dirs (DataFrame): containing classes with their corresponding set.
                               The DataFrame must have two columns: 'split' and 'class'.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as tf.Dataset
    """
    
    ret = {}
    for split in splits:
        print(f"Loading data for split: {split}")

        # Selection of classes for the split and classes path building
        split_classes = data_dirs[data_dirs['split'] == split]['class']
        split_classes = split_classes.apply(lambda x: os.path.join(config['data.dataset'], x)).tolist()

        # Create records DataFrame with complete file paths and labels
        n_way, n_support, n_query, records = configuration(config, split, split_classes)
        print(f"Found classes: {split_classes}")

        # Load and preprocess images, updating the 'file' column to contain image arrays
        w, h, _ = list(map(int, config['model.x_dim'].split(',')))
        #records['file'] = records['file'].apply(lambda x, height=h, width=w: load_and_preprocess_image(x, height=height, width=width))
   
        # Create DataLoader instance
        print(f"Creating DataLoader for split: {split} with {records.shape} samples.")
        data_loader = DataLoader(records,
                                 classes=split_classes,
                                 n_way=n_way,
                                 n_support=n_support,
                                 n_query=n_query)

        ret[split] = data_loader
    return ret
