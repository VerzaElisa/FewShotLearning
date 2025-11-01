import os
from CNN.loader import get_split, get_test
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
        if os.path.isdir(os.path.join(cache_dir, filename)):
            continue
        src_path = os.path.join(cache_dir, filename)
        dst_path = os.path.join(models_metrics_dir, filename)
        os.rename(src_path, dst_path)
    print("Moved cached data to models_metrics directory.")

def create_class_list(count_df, starting_index, n_classes, from_start=True):
    """
    Create a list of classes to include in the training.
    Args:
        count_df: DataFrame containing species and their file counts.
        starting_index: Index to start adding new classes from.
        n_classes: Number of classes to add.
        from_start: If True, start including classes from the beginning of count_df;
                    otherwise, start from starting_index.
    Returns:
        class_list_plus_n: List of classes including the newly added ones.
    """
    count_df = count_df[count_df['file_count'] > 1]
    sorted_df = count_df.sort_values(by='file_count', ascending=False)
    sorted_labels = sorted_df['species'].tolist()
    first_class = 0 if from_start else starting_index
    class_list_plus_n = sorted_labels[first_class:starting_index + n_classes]
    added_classes = sorted_labels[starting_index:starting_index + n_classes]
    print(f'Added classes: {added_classes}')
    return class_list_plus_n

def train_routine(count_df, patience, split_perc, data_dir, w_h, new_classes, to_train, subfolder, from_start=True, cardinality=None):
    """
    Execute the training routine.
    Args:
        count_df: DataFrame containing species and their file counts.
        patience: Number of epochs with no improvement after which training will be stopped.
        split_perc: Dictionary containing the train/validation split percentages (keys: 'train', 'val', 'test').
        data_dir: Directory containing the images divided into subfolders by class.
        w_h: Tuple containing the width and height to which images will be resized.
        new_classes: Tuple containing the starting index and number of new classes to add.
        to_train: Boolean indicating whether to execute the training.
        cardinality: If specified, filter classes by minimum number of samples.
    Returns:
        n_classes: Total number of classes used in training.
        history: Training history object.
        
    """

    if cardinality is not None:
        class_list = count_df[count_df['file_count'] >= cardinality]['species'].tolist()
    else:
        class_list = create_class_list(count_df, new_classes[0], new_classes[1], from_start=from_start)

    n_classes = len(class_list)
    print(f'Total classes found: {n_classes}')

    tot_files = count_df[count_df['species'].isin(class_list)]['file_count'].sum()
    tot_train_files = int(tot_files * split_perc['train'])
    class_weight_dict = {}
    class_name_to_index = {name: index for index, name in enumerate(class_list)}
    for cls in class_list:
        class_num = class_name_to_index[cls]
        cls_count = count_df[count_df['species'] == cls]['file_count'].values[0]
        cls_train_count = int(cls_count * split_perc['train'])
        # same formula used in sklearn.utils.class_weight.compute_class_weight for 'balanced' mode
        class_weight_dict[class_num] = tot_train_files / (n_classes * cls_train_count)

    split_ds = get_split(data_dir, class_list, split_perc, w_h[1], w_h[0])
    test_tensor = get_test(os.path.join("data", "mammals_calls_test"), 'data_cache/CNN/label_to_index.csv', class_list, 164, 397)
    print(class_list)
    if to_train:
        history = train(split_ds['train'], split_ds['val'], test_tensor, patience=patience, cp_path='checkpoints', w_h=(w_h[0], w_h[1]), n_classes=n_classes, class_weight_dict=class_weight_dict)
        dest_folder = os.path.join(MODELS_METRICS_DIR, subfolder)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        move_data(CNN_CACHE_DIR, dest_folder)
    return n_classes + new_classes[0], history if to_train else None