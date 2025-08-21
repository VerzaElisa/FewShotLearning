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
    def __init__(self, data, n_classes, n_way, n_support, n_query):
        self.data = data
        self.n_way = n_way
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query

    def get_next_episode(self):
        h, w, c = 164, 397, 3
        support = np.zeros([self.n_way, self.n_support, h, w, c], dtype=np.float32)
        query = np.zeros([self.n_way, self.n_query, h, w, c], dtype=np.float32)

        # Randomly select classes for the episode
        classes_ep = rng.permutation(self.n_classes)[:self.n_way]

        for i, i_class in enumerate(classes_ep):
            # Randomly select support and query examples from the class (for each class in the episode)
            curr_subset = self.data[i_class]
            n_examples = curr_subset.shape[0]

            selected = rng.permutation(n_examples)[:self.n_support + self.n_query]

            # TODO: gli esempi si support e query non sono abbastanza (per classi con 2 esempi) togli la print risolto il problema
            print(f"selezionati {len(curr_subset.loc[selected[:self.n_support], 'file'].tolist())} support "
                  f"e {len(curr_subset.loc[selected[self.n_support:], 'file'].tolist())} query per classe "
                  f"{curr_subset['label'].unique()}")
            
            support[i] = np.stack(curr_subset.loc[selected[:self.n_support], 'file'].tolist())
            query[i] = np.stack(curr_subset.loc[selected[self.n_support:], 'file'].tolist())

        return support, query
    
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

def from_df_to_dict(data):
    """
    Create a dictionary from data DataFrame with keys as class indexes
    and values as DataFrame of class name and numpy array of the file.
    """
    data_dict = {}
    for i, class_name in enumerate(data['label'].unique()):
        class_data = data[data['label'] == class_name].copy()
        class_data['class_index'] = i
        class_data = class_data.reset_index(drop=True)
        class_data.index.name = "sample_id"
        data_dict[i] = class_data
    return data_dict

def configuration(config, split, data_dir):
    """
    Configuration for the DataLoader.
    Args:
        config (dict): input dict with model workflow params.
        split (str): 'train'|'val'|'test'
        data_dir (str): path of the directory with data.
    Returns:
        n_way (int): number of classes per episode.
        n_support (int): number of support examples per class.
        n_query (int): number of query examples per class.
        class_names (list): list of all class names.
        records (list): list of dictionaries with filepath and label of each sample.
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

    # Get all class names
    class_names = next(os.walk(data_dir))[1]
    
    records = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.endswith('.png'):
                fpath = os.path.join(class_dir, fname)
                records.append({"file": fpath, "label": class_name})
    return n_way, n_support, n_query, class_names, records

def load(data_dirs, config, splits):
    """
    Load dataset.

    Args:
        data_dirs (str): path of the directory with 'splits' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as tf.Dataset
    """
    
    ret = {}
    for split in splits:
        
        print(f"Loading data for split: {split}")
        n_way, n_support, n_query, class_names, records = configuration(config, split, data_dirs[split])
        ds_df_dir = os.path.join(DATA_CACHE_DIR, f"ds_{split}.pkl")
        print(f"Found classes: {class_names}")

        # DataFrame creation and serialization, if already serialized: loading
        if not os.path.exists(ds_df_dir):
            data = pd.DataFrame(records)
            w, h, c = list(map(int, config['model.x_dim'].split(',')))
            data['file'] = data['file'].apply(lambda x: load_and_preprocess_image(x, height=h, width=w))
            data.to_pickle(ds_df_dir)
            print(f"Data saved to {ds_df_dir}")
            
        else:
            data = pd.read_pickle(ds_df_dir)
            print(f"Data loaded from {ds_df_dir}")
        data_dict = from_df_to_dict(data)

        # Create DataLoader instance
        data_loader = DataLoader(data_dict,
                                 n_classes=len(class_names),
                                 n_way=n_way,
                                 n_support=n_support,
                                 n_query=n_query)

        ret[split] = data_loader
    return ret
