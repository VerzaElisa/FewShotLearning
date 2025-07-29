import pandas as pd
import os
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import signal
from pathlib import Path
from scipy.io import wavfile
from ds_download import download_dataset, retrieve_metadata, check_full_audio, link_generator

DOWNLOAD_FILES = False
DOWLOAD_METADATA = False
CHECK_MASTER_TAPES = False

url = 'https://whoicf2.whoi.edu/science/B/whalesounds/fullCuts.cfm'

ds_path = 'dataset'
md_path = 'metadata'
mt_path = 'master_tapes'

if not os.path.exists(ds_path):
    os.makedirs(ds_path)
if not os.path.exists(md_path):
    os.makedirs(md_path)
if not os.path.exists(mt_path):
    os.makedirs(mt_path)

# download dei file ritagliati
if DOWNLOAD_FILES:
    download_dataset(ds_path, url, 'getSpecies', 'pickYear')

# creazione dei metadati con il numero di file per specie
folder_list = []
for folder in os.listdir(ds_path):
    folder_path = Path(ds_path, folder)
    if os.path.isdir(folder_path):
        file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(Path(folder_path, f))])
        folder_list.append({'species': folder, 'file_count': file_count})

species_df = pd.DataFrame(folder_list)
species_df.to_csv(Path(md_path, 'species_count.csv'), index=False)