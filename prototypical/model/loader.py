import os
import glob
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
import random

seed = 2025
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
rng = np.random.default_rng(seed = seed)

class DataLoader(object):
    def __init__(self, data, n_classes, n_way, n_support, n_query, size):
        self.data = data
        self.n_way = n_way
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query
        self.h = size['h']
        self.w = size['w']
        self.c = size['c']

    def get_next_episode(self):
        support = np.zeros([self.n_way, self.n_support, self.h, self.w, self.c], dtype=np.float32)
        query = np.zeros([self.n_way, self.n_query, self.h, self.w, self.c], dtype=np.float32)
        classes_ep = np.random.permutation(self.n_classes)[:self.n_way]

        for i, i_class in enumerate(classes_ep):
            selected = np.random.permutation(self.n_support + self.n_query)
            support[i] = self.data[i_class, selected[:self.n_support]]
            query[i] = self.data[i_class, selected[self.n_support:]]

        return support, query


def class_names_to_paths(data_dir, class_names):
    """
    Return full paths to the directories containing classes of images.

    Args:
        data_dir (str): directory with dataset
        class_names (list): names of the classes in format species

    Returns (list, list): list of paths to the classes
    """
    d = []
    for class_name in class_names:
        image_dir = os.path.join(data_dir, 'data', class_name)
        d.append(image_dir)
    return d


def get_class_images_paths(dir_paths):
    """
    Return class names, paths to the corresponding images from
    the path of the classes' directories.

    Args:
        dir_paths (list): list of the class directories

    Returns (list, list): list of class names, list of lists of paths to
    the images.

    """
    classes, img_paths= [], []
    count_list = [len(os.listdir(dir_path)) for dir_path in dir_paths]
    min_count = min(count_list)

    for dir_path in dir_paths:
        class_images = sorted(glob.glob(os.path.join(dir_path, '*.png')))
        np.random.shuffle(class_images)
        class_images = class_images[:min_count]
        classes.append(dir_path)
        img_paths.append(class_images)
    return classes, img_paths


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

def get_config_info(config, split):
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
    size_str = config['model.x_dim'].split(',')
    size = {'h': int(size_str[0]), 'w': int(size_str[1]), 'c': int(size_str[2])}
    return {'n_way': n_way, 'n_support': n_support, 'n_query': n_query, 'size': size}

def load(data_dir, config, splits):
    """
    Load mammals calls dataset.

    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as tf.Dataset

    """
    split_dir = os.path.join(data_dir, 'splits', config['data.split'])
    ret = {}
    for split in splits:
        config_dict = get_config_info(config, split)

        # Get all class names
        class_names = []
        with open(os.path.join(split_dir, f"{split}.txt"), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))

        # Get class names, images paths per each class
        class_paths = class_names_to_paths(data_dir,
                                            class_names)
        classes, img_paths = get_class_images_paths(
            class_paths)
        sizes = config_dict['size']
        data = np.zeros([len(classes), len(img_paths[0]), sizes['h'], sizes['w'], sizes['c']])
        for i_class in range(len(classes)):
            for i_img in range(len(img_paths[i_class])):
                data[i_class, i_img, :, :, :] = load_and_preprocess_image(
                    img_paths[i_class][i_img], config_dict['size'])

        data_loader = DataLoader(data,
                                 n_classes=len(classes),
                                 n_way=config_dict['n_way'],
                                 n_support=config_dict['n_support'],
                                 n_query=config_dict['n_query'],
                                 size=sizes)

        ret[split] = data_loader
    print(f"Loaded {len(splits)} splits with {len(classes)} classes each.")
    return ret

def embedding(support, w_h_c, model_dir):
    """
    Compute prototypes for the given support set using the trained model.
    Args:
        support (Tensor): support set of shape [n_support, w, h, c]
        w_h_c (tuple): width, height, channels of the images
        model_dir (str): path to the trained model
    Returns (ndarray): prototypes of shape [embedding_dim,]"""
    w, h, c = w_h_c
    model = tf.keras.models.load_model(model_dir)
    z = []
    for cat in support:
        cat = tf.reshape(cat, [1, w, h, c])
        z.append(model(cat))
    z = tf.concat(z, axis=0)
    print(f"initial shape {z.shape}")
    # Prototypes are means of n_support examples
    z_prototypes = tf.math.reduce_mean(z, axis=0)
    print(f"prototypes shape {z_prototypes.shape}")
    return z_prototypes.numpy()


def get_samples(classes, n_support_dict, w_h_c, model_dir, data_dir):
    """
    Load samples and compute embeddings for each class. After that, save
    the embeddings in a csv file.
    Note: every class can have different number of support samples.
    Args:
        classes (list): list of class names
        n_support_dict (dict): dict with number of support samples per class
        w_h_c (tuple): width, height, channels of the images
        model_dir (str): path to the trained model
        data_dir (str): path to the data directory
    Returns (DataFrame): dataframe with class names and embeddings
    """
    embedding_dict = {}
    for curr_class in classes:

        # Get number of support samples for the current class
        n_support = n_support_dict[curr_class]
        main_dir = os.path.join(data_dir, curr_class)
        files = os.listdir(main_dir)
        selected_files = random.sample(files, n_support)

        # Load support samples for the current class
        class_embeddings = []
        for i_img in range(n_support):
            curr_img = os.path.join(main_dir, selected_files[i_img])
            class_embeddings.append(load_and_preprocess_image(curr_img, w_h_c))
        embedding_dict[curr_class] = class_embeddings
    
    # Compute embeddings and save to csv
    embedding_df = pd.DataFrame(list(embedding_dict.items()), columns=['class', 'embeddings'])
    embedding_df['embeddings'] = embedding_df['embeddings'].apply(lambda x: embedding(x, w_h_c, model_dir))  
    embedding_df.to_csv('proto_embeddings.csv', index=False)
    return embedding_df
