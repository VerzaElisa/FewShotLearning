import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd

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

    #TODO - probabilmente non funziona, devi vedere dove e come viene usato
    def get_next_episode(self):
        h, w, c = 164, 397, 3
        support = np.zeros([self.n_way, self.n_support, h, w, c], dtype=np.float32)
        query = np.zeros([self.n_way, self.n_query, h, w, c], dtype=np.float32)

        # Randomly select classes for the episode
        classes_ep = np.random.permutation(self.n_classes)[:self.n_way]

        for i, i_class in enumerate(classes_ep):
            # Randomly select support and query examples from the class (for each class in the episode)
            curr_subset = self.data[i_class]
            n_examples = curr_subset.shape[0]
            curr_subset.to_csv('curr_subset.csv')

            selected = np.random.permutation(n_examples)[:self.n_support + self.n_query]
            support[i] = np.stack(curr_subset.loc[selected[:self.n_support], 'file'].tolist())
            query[i] = np.stack(curr_subset.loc[selected[self.n_support:], 'file'].tolist())

        return support, query
    
def load_and_preprocess_image(img_path, height=164, width=397):
    print(f"preprocessing classe: {img_path.split('/')[-2]}", end='\r')
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [height, width])
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()

def from_df_to_dict(data):
    """
    Create a dictionary from data DataFrame with keys as class indexes
    and values as DataFrame of class name and tensor of the file.
    """
    data_dict = {}
    for i, class_name in enumerate(data['label'].unique()):
        class_data = data[data['label'] == class_name]
        class_data['class_index'] = i
        class_data = class_data.reset_index(drop=True)
        class_data.index.name = "sample_id"
        data_dict[i] = class_data
    return data_dict

def load(data_dir, config, splits):
    """
    Load omniglot dataset.

    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as tf.Dataset

    """
    ds_df_dir = os.path.join(DATA_CACHE_DIR, "ds.pkl")
    ret = {}
    for split in splits:
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

        # Creazione del dataframe con dataset preprocessato e serializzazione, se già serializzato: caricamento
        if not os.path.exists(ds_df_dir):
            data = pd.DataFrame(records)
            data['file'] = data['file'].apply(lambda x: load_and_preprocess_image(x))
            data.to_pickle(ds_df_dir)
            print(f"Data saved to {ds_df_dir}")
            
        else:
            data = pd.read_pickle(ds_df_dir)
            print(f"Data loaded from {ds_df_dir}")
        data_dict = from_df_to_dict(data)
        print(data_dict[1].info())
        data_loader = DataLoader(data_dict,
                                 n_classes=len(class_names),
                                 n_way=n_way,
                                 n_support=n_support,
                                 n_query=n_query)

        ret[split] = data_loader
    return ret
