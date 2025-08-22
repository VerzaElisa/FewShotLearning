import os
import numpy as np
import pandas as pd
from math import floor
from model_workflow.train import train

cuda_on = 0
rng = np.random.default_rng(seed = 2025)

def split_classes(data_dir, train_perc, val_perc):
    split = {}
    classes = [d for d in os.listdir(data_dir) if len(os.listdir(os.path.join(data_dir, d))) < 1000 and 
               len(os.listdir(os.path.join(data_dir, d))) > 10 and
               os.path.isdir(os.path.join(data_dir, d))]
    print(f"Total classes found: {len(classes)}")
    len_val = floor(len(classes) * val_perc)
    len_train = floor(len(classes) * train_perc)
    perm = rng.permutation(len_val + len_train)
    split['train'] = [classes[i] for i in perm[:len_train]]
    split['val'] = [classes[i] for i in perm[len_train:]]

    df = pd.DataFrame({"split": list(split.keys()), "class": list(split.values())})
    return df.explode("class", ignore_index=True)

def test_1_shot_1_way():
    config = {
        "data.dataset": "data/mammals_calls",
        "data.train_class_percent": 0.6,
        "data.val_class_percent": 0.4,
        "data.train_way": 3,
        "data.train_support": 5,
        "data.train_query": 5,
        "data.test_way": 3,
        "data.test_support": 5,
        "data.test_query": 5,
        "data.episodes": 10,
        "data.cuda": cuda_on,
        "data.gpu": 0,
        "model.x_dim": "397,164,3",
        "model.z_dim": 64,
        "train.epochs": 2,
        'train.optim_method': "Adam",
        "train.lr": 0.001,
        "train.patience": 5,
        "model.save_path": 'mmd.h5'
    }
    split_df = split_classes(config['data.dataset'], config['data.train_class_percent'], config['data.val_class_percent'])
    print(split_df.head())
    train(config, split_df)
    print("Training completed.")


if __name__ == "__main__":
    test_1_shot_1_way()
