import os
import numpy as np
from generator import generate
import json

def load_config(path=r'state-vector-calculations\TEPAIConfig.json'):
    """Load TEPAI configuration from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def proceed():
    """
    Checks if 'data/pai_snap0.csv' exists in the relative 'data/' folder.
    Returns False if the file exists, otherwise returns True.
    """
    # Construct the path to the target file
    file_path = os.path.join('data', 'pai_snap0.csv')

    # os.path.isfile returns True if the path points to an existing regular file
    return not os.path.isfile(file_path)

def should_process_params(numQs: int, Δ: float, T: int, N: int, n_snapshot: int, resamples: int) -> bool:
    """
    Checks if a folder with the given parameters exists in the 'data/' directory.
    The folder naming convention is:
        N-{N}-n-{n_snapshot}-r-{resamples}-Δ-{Δ}-T-{T}-q-{numQs}

    Returns False if such a folder exists, otherwise returns True.
    """
    # Build the expected folder name using default string formatting for the float
    folder_name = f"N-{N}-n-{n_snapshot}-r-{resamples}-Δ-{Δ}-T-{T}-q-{numQs}"
    folder_path = os.path.join('data', folder_name)

    # os.path.isdir returns True if the path points to an existing directory
    return not os.path.isdir(folder_path)

def main():
    config = load_config()
    for name, params in config.items():
        print(f"Running generate() for config {name}: {params}")
        numQs=params['numQs']
        Δ=np.pi / 2**params['Δ']
        T=params['T']
        N=params['N']
        n_snapshot=10
        resamples=params['resamples']
        
        if proceed() and should_process_params(numQs, Δ, T, N, n_snapshot, resamples):
            generate(numQs, Δ, T, N, n_snapshot, resamples)
        elif not proceed():
            print("Aborting TE-PAI simulation. 'data/pai_snap0.csv' already exists.")
        elif not should_process_params(numQs, Δ, T, N, n_snapshot, resamples):
            print(f"Aborting TE-PAI simulation. Folder with parameters {params} already exists.")

if __name__ == '__main__':
    main()