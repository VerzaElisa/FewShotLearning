"""
Logic for model creation, training launching and actions needed to be
accomplished during training (metrics monitor, model saving etc.)
"""

import os
import time
import numpy as np
import tensorflow as tf

from prototypical.model.prototypical import Prototypical
from prototypical.model.loader import load
from prototypical.train.train_engine import TrainEngine


def train(config):
    np.random.seed(2019)
    tf.random.set_seed(2019)

    # Create folder for model
    save_path = config['model.save_path']
    model_dir = save_path[:save_path.rfind('/')]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    data_dir = f"data/{config['data.dataset']}"
    ret = load(data_dir, config, ['train', 'val'])
    train_loader = ret['train']
    val_loader = ret['val']

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    # Setup training operations
    n_support = config['data.train_support']
    n_query = config['data.train_query']
    w, h, c = list(map(int, config['model.x_dim'].split(',')))
    model = Prototypical(n_support, n_query, w, h, c)
    optimizer = tf.keras.optimizers.Adam(config['train.lr'])

    # Metrics to gather
    train_loss = tf.metrics.Mean(name='train_loss')
    val_loss = tf.metrics.Mean(name='val_loss')
    train_acc = tf.metrics.Mean(name='train_accuracy')
    val_acc = tf.metrics.Mean(name='val_accuracy')
    val_losses = []

    @tf.function
    def loss(model, support, query):
        loss, acc = model(support, query)
        return loss, acc

    @tf.function
    def train_step(model, support, query):
        # Forward & update gradients
        with tf.GradientTape() as tape:
            loss, acc = model(support, query)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        # Log loss and accuracy for step
        train_loss(loss)
        train_acc(acc)

    @tf.function
    def val_step(model, loss_func, support, query):
        loss, acc = loss_func(model, support, query)
        val_loss(loss)
        val_acc(acc)

    # Create empty training engine
    train_engine = TrainEngine()

    # Set hooks on training engine
    def on_start(state):
        print("Training started.")
    train_engine.hooks['on_start'] = on_start

    def on_end(state):
        print("Training ended.")
    train_engine.hooks['on_end'] = on_end

    def on_start_epoch(state):
        print(f"Epoch {state['epoch']} started.")
        train_loss.reset_state()
        val_loss.reset_state()
        train_acc.reset_state()
        val_acc.reset_state()
    train_engine.hooks['on_start_epoch'] = on_start_epoch

    def on_end_epoch(state):
        print(f"Epoch {state['epoch']} ended.")
        epoch = state['epoch']
        template = 'Epoch {}, Loss: {}, Accuracy: {}, ' \
                   'Val Loss: {}, Val Accuracy: {}'
        print(template.format(epoch + 1, train_loss.result(), train_acc.result() * 100,
                            val_loss.result(), val_acc.result() * 100))

        cur_loss = val_loss.result().numpy()
        val_losses.append(cur_loss)

        patience = config['train.patience']

        # Early stopping tracking
        if cur_loss < state['best_val_loss']:
            print("Saving new best model with loss:", cur_loss)
            state['best_val_loss'] = cur_loss
            state['no_improve_epochs'] = 0  # reset patience counter
            model.save(config['model.save_path'])
        else:
            state['no_improve_epochs'] += 1

        if state['no_improve_epochs'] >= patience:
            print(f"No improvement for {patience} epochs. Early stopping triggered.")
            state['early_stopping_triggered'] = True
    train_engine.hooks['on_end_epoch'] = on_end_epoch

    def on_start_episode(state):
        support, query = state['sample']
        train_step(state['model'], support, query)
    train_engine.hooks['on_start_episode'] = on_start_episode

    def on_end_episode(state):
         # Validation
        val_loader = state['val_loader']
        loss_func = state['loss_func']
        for _ in range(config['data.episodes']):
            support, query = val_loader.get_next_episode()
            val_step(state['model'], loss_func, support, query)
    train_engine.hooks['on_end_episode'] = on_end_episode

    time_start = time.time()
    with tf.device(device_name):
        train_engine.train(
            model=model,
            loss_func=loss,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['train.epochs'],
            n_episodes=config['data.episodes'])
    time_end = time.time()

    elapsed = time_end - time_start
    h, minutes = elapsed//3600, elapsed%3600//60
    sec = elapsed-minutes*60
    print(f"Training took: {h} h {minutes} min {sec} sec")
