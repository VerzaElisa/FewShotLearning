import os

from model_workflow.train import train

cuda_on = 0


def test_1_shot_1_way():
    config = {
        "data.dataset": "mammals_calls_short",
        "data.split": "mmd_splits",
        "data.train_way": 5,
        "data.train_support": 2,
        "data.train_query": 2,
        "data.test_way": 1,
        "data.test_support": 1,
        "data.test_query": 1,
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
    train(config)
    os.remove('mmd.h5')


if __name__ == "__main__":
    test_1_shot_1_way()
