import os
import pandas as pd
import random

seed = 2025
random.seed(seed)

def phisical_split(ds_dir, split_perc):
    """
    Phisically separate test files from the main dataset directory into a dedicated one.
    
    Parameters:
    ds_dir (str): Path to the dataset directory.
    split_perc (float): Percentage of data to be used for training (between 0 and 1).
    """
    data_dir = ds_dir.split('/')[0]
    test_dir = os.path.join(data_dir, 'mammals_calls_test')
    
    # Create test directory if it doesn't exist
    os.makedirs(test_dir, exist_ok=True)

    for species in os.listdir(ds_dir):
        species_path = os.path.join(ds_dir, species)
        if os.path.isdir(species_path):
            files = os.listdir(species_path)
            num_train = int(len(files) * split_perc)
            # Shuffle files to ensure randomness
            random.shuffle(files)
            test_files = files[num_train:]
            
            # Create species subdirectories in test directory
            os.makedirs(os.path.join(test_dir, species), exist_ok=True)
            
            # Move files to respective directories
            for file in test_files:
                os.rename(os.path.join(species_path, file), os.path.join(test_dir, species, file))
