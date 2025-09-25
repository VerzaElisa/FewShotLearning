import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import soundfile as sf
from soundfile import LibsndfileError
from scipy import signal


def chunk_data(data: np.array, sample_rate: int, chunk_size: int, hop: int) -> list:
    """
    Chunk audio data into smaller segments.
    Args:
        data (np.array): Audio data array.
        sample_rate (int): Sample rate of the audio data.
        chunk_size (int): Size of each chunk in seconds.
        hop (int): Hop size in seconds.
    Returns:
        list: List of audio chunks.
    """
    # get number of samples per chunk
    chunk_samples = int(sample_rate * chunk_size)

    # get chunks with specified hop size
    chunks = [data[i:i + chunk_samples] for i in range(0, len(data), int(hop * sample_rate)) if i + chunk_samples <= len(data)]
    return chunks

def save_chunk(plot_struct: dict, data: np.array, output_file: str, cmap='magma'):
    """
    Save a spectrogram chunk as an image file.
    Args:
        plot_struct (dict): Dictionary containing matplotlib figure and axis (keys: 'fig', 'ax').
        data (np.array): Spectrogram data array.
        output_file (str): Path to save the output image file.
        cmap (str): Colormap to use for the spectrogram.
    """
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    ax = plot_struct['ax']
    fig = plot_struct['fig']
    ax.clear()
    ax.axis('off')
    ax.imshow(data, aspect='auto', origin='lower', cmap=cmap)

    fig.savefig(output_file, bbox_inches='tight', transparent=True)
    plt.close(fig)


def preprocessing(full_data: np.array, sample: int, species_dir: dict, file_name: str, sft_config: dict, chunk_config: dict, cmap: str):
    """
    Preprocess audio data to generate and save spectrogram chunks as images and numeric data.
    Args:
        full_data (np.array): Full audio data array.
        sample (int): Sample rate of the audio data.
        species_dir (dict): Directory to save the spectrograms and numeric data (keys: 'spec', 'num').
        file_name (str): Base name for the output files.
        sft_config (dict): Configuration for Short-Time Fourier Transform (keys: 'win', 'hop', 'fs').
        chunk_config (dict): Configuration for chunking the audio data (keys: 'size', 'hop').
        cmap (str): Colormap to use for the spectrogram.
    """
    matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=(10, 4))

    # chunk the data
    chunked_data = chunk_data(full_data, sample, chunk_config['size'], chunk_config['hop'])
    chunk_num = 0
    for data in chunked_data:
        composed_filename = file_name+'-'+str(chunk_num)
        output_file = os.path.join(species_dir['spec'], composed_filename + ".png")

        # STF calculation
        SFT = signal.ShortTimeFFT(sft_config['win'], sft_config['hop'], sft_config['fs'])
        s_x = SFT.stft(data)

        spectrogram = np.abs(s_x)**2

        # log scaling
        log_spectrogram = np.log(spectrogram + 1e-10)
        # saving spectrogram image as PNG
        if not os.path.exists(output_file):
            save_chunk({'fig': fig, 'ax': ax}, log_spectrogram, output_file, cmap)

        chunk_num += 1
    
    plt.close(fig)


def species_spec(dirs, species_list, output_dir, ds_path, stf_config, chunk_config, cmap):
    # definizione della finestra di Hann
    hann_win = signal.windows.hann(stf_config['frame_win'])
    j = 0
    for curr_dir in dirs:
        if curr_dir not in species_list:
            continue
        print(f'Processing directory: {curr_dir}: {j+1}/{len(species_list)}')
        curr_files = os.listdir(os.path.join(ds_path, curr_dir))
        i = 0
        for file in curr_files:
            print(f'Processing {i+1}/{len(curr_files)} files in {curr_dir}', end='\r')
            if file.endswith('.wav'):
                try:
                    x, sr = sf.read(os.path.join(ds_path, curr_dir, file))
                except LibsndfileError:
                    continue
                spec_curr_dir = os.path.join(output_dir['spec'], curr_dir)
                num_curr_dir = os.path.join(output_dir['num'], curr_dir)
                stf_config['win'] = hann_win
                preprocessing(x, sr, {'spec': spec_curr_dir, 'num': num_curr_dir}, file.split('.')[0], stf_config, chunk_config, cmap)
            
            i += 1
        j += 1