import numpy as np


class TrainEngine(object):
    """
    Engine that launches training per epochs and episodes.
    Contains hooks to perform certain actions when necessary.
    """
    def __init__(self):
        self.hooks = {name: lambda state: None
                      for name in ['on_start',
                                   'on_start_epoch',
                                   'on_end_epoch',
                                   'on_start_episode',
                                   'on_end_episode',
                                   'on_end']}

    def train(self, loss_func, train_loader, val_loader, epochs, n_episodes, **kwargs):
        # State of the training procedure
        state = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'loss_func': loss_func,
            'sample': None,
            'epoch': 1,
            'total_episode': 1,
            'epochs': epochs,
            'n_episodes': n_episodes,
            'best_val_loss': np.inf,
            'early_stopping_triggered': False
        }
        # on_start hook just prints the message "Training started."
        self.hooks['on_start'](state)

        for _ in range(state['epochs']):

            # Reset the state of the losses and accuracies
            self.hooks['on_start_epoch'](state)

            for i_episode in range(state['n_episodes']):
                # Get the next episode's data from the train_loader
                print(f"Episode {state['total_episode']} started.")
                support, query = train_loader.get_next_episode()
                state['sample'] = (support, query)

                # Execute the forward and backward pass and update the model parameters
                self.hooks['on_start_episode'](state)
                if i_episode+1 == state['n_episodes']:
                    break
                
                self.hooks['on_end_episode'](state)
                print(f"Episode {state['total_episode']} ended.")
                state['total_episode'] += 1

            self.hooks['on_end_epoch'](state)
            state['epoch'] += 1

            # Early stopping
            if state['early_stopping_triggered']:
                print("Early stopping triggered!")
                break

        self.hooks['on_end'](state)
        print("Training succeed!")
