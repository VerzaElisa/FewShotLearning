import os
import time
import numpy as np
import tensorflow as tf
from model_workflow.train_engine import TrainEngine

from protonet.loader import load
from protonet.prototypical import Prototypical

try:
    # Con CUDA disabilitato, questa lista sarà sempre vuota
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.gpu.set_per_process_memory_growth(True)
        print(f"GPU trovate: {len(gpus)}")
    else:
        print("Utilizzo CPU (CUDA disabilitato)")
except Exception as e:
    print(f"Errore nella configurazione: {e}")


def train(config):
    np.random.seed(2019)
    tf.random.set_seed(2019)
    print('1. Loading data...')
    # Creazione cartella per il modello
    model_dir = os.path.join("data_cache", "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Creazione del dataset diviso in training e validation
    data_dir = {}
    data_dir['train'] = os.path.join("data", config['data.dataset'], "train")
    data_dir['val'] = os.path.join("data", config['data.dataset'], "val")

    ret = load(data_dir, config, ['train', 'val'])
    train_loader = ret['train']
    val_loader = ret['val']

    # Scelta CPU/GPU
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    # Setup training operations
    n_support = config['data.train_support']
    n_query = config['data.train_query']
    w, h, c = list(map(int, config['model.x_dim'].split(',')))

    # Model initialization and optimizer
    model = Prototypical(n_support, n_query, w, h, c)
    optimizer = tf.keras.optimizers.Adam(config['train.lr'])

    # Metrics to gather
    #TODO: generalizzare la scelta delle metriche
    train_loss = tf.metrics.Mean(name='train_loss')
    val_loss = tf.metrics.Mean(name='val_loss')
    train_acc = tf.metrics.Mean(name='train_accuracy')
    val_acc = tf.metrics.Mean(name='val_accuracy')
    val_losses = []

    @tf.function
    def loss(support, query):
        # execute loss and accuracy computation given support and query sets doing the forward pass
        # and return the computed metrics
        loss, acc = model(support, query)
        return loss, acc

    @tf.function
    def train_step(model, support, query):
        with tf.GradientTape() as tape:
            # Forward pass
            loss, acc = model(support, query)

        # Backward pass: returns a list of gradients for each trainable variable
        gradients = tape.gradient(loss, model.trainable_variables)

        # Update model parameters
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Log loss and accuracy for step
        train_loss(loss)
        train_acc(acc)

    @tf.function
    def val_step(loss_func, support, query):
        loss, acc = loss_func(support, query)
        val_loss(loss)
        val_acc(acc)

    # Create empty training engine
    train_engine = TrainEngine()

    # Set hooks on training engine
    # on_start and on_end hooks just print messages
    def on_start(state):
        print("Training started.")
    train_engine.hooks['on_start'] = on_start

    def on_end(state):
        print("Training ended.")
    train_engine.hooks['on_end'] = on_end

    # Reset the state of the losses and accuracies at the start of each epoch
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
        print(
            template.format(epoch + 1, train_loss.result(), train_acc.result() * 100,
                            val_loss.result(),
                            val_acc.result() * 100))

        cur_loss = val_loss.result().numpy()
        if cur_loss < state['best_val_loss']:
            print("Saving new best model with loss: ", cur_loss)
            state['best_val_loss'] = cur_loss
            model.save(os.path.join(model_dir, config['model.save_path']))
        val_losses.append(cur_loss)

        # Early stopping
        patience = config['train.patience']
        if len(val_losses) > patience \
                and max(val_losses[-patience:]) == val_losses[-1]:
            state['early_stopping_triggered'] = True
    train_engine.hooks['on_end_epoch'] = on_end_epoch

    # Execute training step. The tf.function train_step executes 
    # the forward and backward pass and updates the model parameters.
    def on_start_episode(state):
        
        support, query = state['sample']
        train_step(model, support, query)
    train_engine.hooks['on_start_episode'] = on_start_episode

    def on_end_episode(state):
        # Validation
        print("Validating...")
        val_loader = state['val_loader']
        # loss function is initialized when the train method of TrainEngine is called
        loss_func = state['loss_func']
        
        # TODO: alla fine di ogni episodio fa validazione n volte con n numero di episodi,
        # serve farlo? o andrebbe fatta validazione alla fine di ogni epoca?
        for _ in range(config['data.episodes']):
            support, query = val_loader.get_next_episode()
            val_step(loss_func, support, query)
    train_engine.hooks['on_end_episode'] = on_end_episode

    time_start = time.time()
    with tf.device(device_name):
        train_engine.train(
            loss_func=loss,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['train.epochs'],
            n_episodes=config['data.episodes'])
    time_end = time.time()

    elapsed = time_end - time_start
    h, min_val = elapsed//3600, elapsed%3600//60
    sec = elapsed-min_val*60
    print(f"Training took: {h} h {min_val} min {sec} sec")
