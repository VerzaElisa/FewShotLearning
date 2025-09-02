import os
from CNN.loader import get_split
from CNN.model import train

CNN_CACHE_DIR = os.path.join("data_cache", "CNN")
MODELS_METRICS_DIR = os.path.join("models_metrics")

def move_data(cache_dir, models_metrics_dir):
    """
    Move cached data from the cache directory to the models_metrics directory.
    Args:
        cache_dir (str): Path to the cache directory.
        models_metrics_dir (str): Path to the models_metrics directory.
    """
    if not os.path.exists(models_metrics_dir):
        os.makedirs(models_metrics_dir)

    for filename in os.listdir(cache_dir):
        #eseguire solo se il file non è una cartella
        if os.path.isdir(os.path.join(cache_dir, filename)):
            continue
        src_path = os.path.join(cache_dir, filename)
        dst_path = os.path.join(models_metrics_dir, filename)
        os.rename(src_path, dst_path)
    print("Moved cached data to models_metrics directory.")

def create_class_list(count_df, starting_index, n_classes):
    """
    Create a list of classes to include in the training.
    Args:
        count_df: DataFrame containing species and their file counts.
        starting_index: Index to start adding new classes from.
        n_classes: Number of classes to add.
    Returns:
        class_list_plus_n: List of classes including the newly added ones.
    """
    count_df = count_df[count_df['file_count'] > 1]
    sorted_df = count_df.sort_values(by='file_count', ascending=False)
    sorted_labels = sorted_df['species'].tolist()
    class_list_plus_n = sorted_labels[:starting_index + n_classes]
    added_classes = sorted_labels[starting_index:starting_index + n_classes]
    print(f'Added classes: {added_classes}')
    return class_list_plus_n

def train_routine(count_df, patience, split_perc, data_dir, w_h, new_classes, to_train, cardinality=None):
    """
    Execute the training routine.
    Args:
        count_df: DataFrame containing species and their file counts.
        patience: Number of epochs with no improvement after which training will be stopped.
        split_perc: Dictionary containing the train/validation split percentages.
        data_dir: Directory containing the images divided into subfolders by class.
        w_h: Tuple containing the width and height to which images will be resized.
        new_classes: Tuple containing the starting index and number of new classes to add.
        to_train: Boolean indicating whether to execute the training.
        cardinality: If specified, filter classes by minimum number of samples.
    Returns:
        n_classes: Total number of classes used in training.
    """

    if cardinality is not None:
        class_list = count_df[count_df['file_count'] >= cardinality]['species'].tolist()
    else:
        class_list = create_class_list(count_df, new_classes[0], new_classes[1])
    n_classes = len(class_list)
    print(f'Total classes found: {n_classes}')
    if to_train:
        split_ds = get_split(data_dir, class_list, split_perc, w_h[0], w_h[1])
        train(split_ds['train'], split_ds['val'], patience=patience, cp_path='checkpoints', w_h = (w_h[0], w_h[1]), n_classes=n_classes)
        move_data(CNN_CACHE_DIR, MODELS_METRICS_DIR)
    return n_classes