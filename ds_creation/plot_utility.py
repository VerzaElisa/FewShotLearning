import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import librosa
from sklearn.manifold import TSNE

def mfcc_extractor(row, chunk_size, mfcc_num=50):
    """ 
    Extract MFCC features from an audio file and split into chunks.
    Args:
        row: A row from a DataFrame containing 'audio_files' and 'species'.
        chunk_size (int): Size of each chunk in seconds.
        mfcc_num (int): Number of MFCC features to extract.
    Returns:
        row: The input row with an added 'chunk_list' containing MFCC features for each chunk.
    """
    try:
        signal, sr = librosa.load(row['audio_files'])
    except Exception as e:
        print(f"Error loading audio file {row['audio_files']}: {e}")
        row['chunk_list'] = []
        return row
    chunk_size = chunk_size * sr
    mfcc_chunks = []
    i = 1
    
    for start in range(0, len(signal), sr):
        i += 1
        end = start + chunk_size
        y_chunk = signal[start:end]
        
        if len(y_chunk) < chunk_size:
            break  
        mfcc = librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=mfcc_num)
        mfcc_mean = np.mean(mfcc, axis=1)

        mfcc_chunks.append(mfcc_mean)
    row['chunk_list'] = mfcc_chunks
    return row

def process_audio_files(species_list, audio_dir, models_metrics_dir, mfcc_num=50):
    """
    Process audio files to extract MFCC features and save to CSV.
    Args:
        species_list (list): List of species (subfolder names) to process.
        audio_dir (dict): Directory containing subfolders of audio files.
        models_metrics_dir (str): Directory to save the processed CSV file.
    Returns:
        audio_df_exploded_clean (DataFrame): DataFrame with species and their MFCC chunks.
        mfcc_matrix (ndarray): Numpy array of MFCC features.
    """
    audio_files = {}
    for species in species_list:
        curr_path = os.path.join(audio_dir, species)
        audio_files[species] = [os.path.join(curr_path, f) for f in os.listdir(curr_path) if f.endswith('.wav')]

    audio_df = pd.DataFrame(list(audio_files.items()), columns=['species', 'audio_files'])
    audio_df = audio_df.explode('audio_files').reset_index(drop=True)

    audio_df = audio_df.apply(mfcc_extractor, axis=1, chunk_size=2, mfcc_num=mfcc_num)
    audio_df_exploded = audio_df.explode('chunk_list').reset_index(drop=True)
    audio_df_exploded_clean = audio_df_exploded.dropna(axis=0, subset=['chunk_list'])
    valid_chunks = audio_df_exploded_clean['chunk_list']
    valid_chunks = valid_chunks[valid_chunks.apply(lambda x: isinstance(x, np.ndarray) and len(x) == mfcc_num)]

    mfcc_matrix = np.array(valid_chunks.tolist())
    audio_df_exploded.to_csv(os.path.join(models_metrics_dir, 'audio_data.csv'), index=False)
    return audio_df_exploded_clean, mfcc_matrix

def tsne_calc(audio_df_exploded, mfcc_matrix, models_metrics_dir):    
    tsne = TSNE(n_components=2, random_state=42)
        
    x_transformed = tsne.fit_transform(mfcc_matrix)
    tsne_df = pd.DataFrame(np.column_stack((x_transformed, audio_df_exploded["species"])), columns=['X', 'Y', "Targets"])
    tsne_df.loc[:, "Targets"] = tsne_df.Targets.astype('category')
    tsne_df.to_csv(os.path.join(models_metrics_dir, 'tsne_data.csv'), index=False)
    return tsne_df

def generate_perceptually_uniform_colors(n_colors):
    """
    Generate a list of perceptually uniform colors.
    Args:
        n_colors (int): Number of distinct colors to generate.
    Returns:
        List of colors in RGB format.
    """
    if n_colors <= 10:
        return sns.color_palette("tab10", n_colors)
    elif n_colors <= 20:
        return sns.color_palette("tab20", n_colors)
    else:
        # Per molti colori, combina diverse strategie
        base_colors = sns.color_palette("husl", n_colors)  # HUSL è perceptually uniform
        return base_colors
    
def tsne_plot(tsne_df):
    plt.figure(figsize=(10,8))

    cp = sns.color_palette()
    g = sns.FacetGrid(data=tsne_df, hue='Targets', height=8, palette=cp)
    g.map(plt.scatter, 'X', 'Y').add_legend()
    plt.show()


def get_metrics(cm, label_dict):
    """
    Calculate precision, recall, f1-score, support, tp, fp, fn, tn from confusion matrix.
    Args:
        cm: Confusion matrix as a 2D numpy array.
        label_dict: Dictionary mapping class indices to labels and support.
    Returns:
        metrics_df: DataFrame containing the calculated metrics for each class."""
    num_classes = cm.shape[0]
    true_positives = np.diag(cm)
    false_positives = np.sum(cm, axis=0) - true_positives
    false_negatives = np.sum(cm, axis=1) - true_positives

    support =  np.asarray([label_dict[i]["support"] for i in range(num_classes)], dtype=int)
    true_negatives = np.sum(cm) - (true_positives + false_positives + false_negatives)

    precision = np.divide(true_positives, true_positives + false_positives, out=np.zeros_like(true_positives, dtype=float), where=(true_positives + false_positives) != 0)
    recall = np.divide(true_positives, true_positives + false_negatives, out=np.zeros_like(true_positives, dtype=float), where=(true_positives + false_negatives) != 0)
    f1_score = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision, dtype=float), where=(precision + recall) != 0)
    
    metrics_df = pd.DataFrame({
        'label': [label_dict[i]["label"] for i in range(num_classes)],
        'precision': precision,
        'recall': recall,
        'f1-score': f1_score,
        'support': support,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives,
        'tn': true_negatives
    })

    return metrics_df


def check_training_accuracy(acc_per_epoch_train, acc_per_epoch_val, best_epoch):
    """
    Plot training and validation accuracy over epochs, marking the best epoch.
    Args:
        acc_per_epoch_train: List of training accuracies per epoch.
        acc_per_epoch_val: List of validation accuracies per epoch.
        best_epoch: Epoch number with the best validation accuracy."""
    plt.figure(figsize=(10, 6))
    plt.plot(acc_per_epoch_train, label='Training Accuracy')
    plt.plot(acc_per_epoch_val, label='Validation Accuracy')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def confusion_matrix_plot(cm, labels):
    """
    Plot percentage normalized confusion matrix.
    Args:
        cm: Confusion matrix as a 2D numpy array.
        labels: List of class labels.
    """
    cmn = cm.astype('int') / cm.sum(axis=1)[:, np.newaxis]

    def custom_format(val):
        if val < 0.01:
            return "0"
        else:
            return f"{val:.2f}"
    formatted_annotations = np.vectorize(custom_format)(cmn)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cmn, annot=formatted_annotations, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def metrics_plot_builder(metrics_df):
    """
    Create bar plots for precision, recall, and f1-score per class.
    Args:
        metrics_df: DataFrame containing metrics for each class.
    Returns:
        fig: Matplotlib figure object containing the plots.
    """
    metrics_list = ['precision', 'recall', 'f1-score']
    f = 1
    fig = plt.figure(figsize=(15, 15))
    for metric in metrics_list:
        axs = plt.subplot(2, 2, f)
        axs.bar(metrics_df['label'], metrics_df[metric], color="#87CEEB")
        for i, (metric_value, support) in enumerate(zip(metrics_df[metric], metrics_df['support'])):
            label_pos = metric_value - (metric_value/2) if metric_value > 0 else metric_value + 0.02
            plt.text(i, label_pos, f'n:{int(support)}', ha='center', va='bottom', fontsize=9, rotation=90)
        axs.set_xlabel('Class')
        axs.tick_params(axis='x', rotation=90)
        axs.set_ylabel(metric.capitalize())
        axs.set_title(f'{metric.capitalize()} per Class (ordered by Support - descending)')
        
        f += 1
    fig.tight_layout()
    return fig

def process_metrics(count_df, n_classes, training_path, metrics_dir):
    """
    Process training metrics and generate bar plots for metrics visualization,
    confusion matrix, and t-SNE plot.
    Args:
        count_df: DataFrame containing species and their file counts.
        n_classes: Total number of classes used in training.
        training_path: Path to the training logs and metrics.
        metrics_dir: Directory to save the processed metrics and plots.
    """
    all_classes_df = pd.read_csv(os.path.join(training_path, f'{n_classes}_training_log.csv'))
    label_df = pd.read_csv(os.path.join(training_path, 'label_to_index.csv'))
    label_df = label_df.merge(count_df, left_on='label', right_on='species', how='left', validate='one_to_one', suffixes=('_training', '_total')).drop(columns=['species'])

    best_weights = all_classes_df[all_classes_df['val_accuracy'] == all_classes_df['val_accuracy'].max()]
    best_epoch = best_weights['epoch'].values[0]
    cm = best_weights['val_confusion_matrix']
    cm = cm.values[0]
    cm = cm[2:-2]
    cm_list = cm.split(', ')
    cm_matrix = []
    for r in cm_list:
        r = r[1:-1]
        r = r.split()
        cm_matrix.append([int(i) for i in r])
    cm_matrix = np.array(cm_matrix)
    support = cm_matrix.sum(axis=1)
    label_df['support'] = support
    
    label_dict = label_df.to_dict('index')
    metrics_df = get_metrics(cm_matrix, label_dict)
    metrics_df.to_csv(os.path.join(training_path, f'{n_classes}_metrics.csv'), index=False)
    metrics_plot_builder(metrics_df)
    confusion_matrix_plot(cm_matrix, metrics_df['label'].tolist())

    tsne_df = pd.read_csv(os.path.join(metrics_dir, 'tsne_data.csv'))
    tsne_df = tsne_df[tsne_df['Targets'].isin(label_df['label'].tolist())]
    tsne_plot(tsne_df)

    check_training_accuracy(all_classes_df['accuracy'].tolist(), all_classes_df['val_accuracy'].tolist(), best_epoch)
    
    


