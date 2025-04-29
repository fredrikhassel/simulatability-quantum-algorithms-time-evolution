import os
import shutil
import numpy as np

from te_pai.hamil import Hamiltonian
from te_pai.sampling import resample
from te_pai.te_pai import TE_PAI
from te_pai.trotter import Trotter

def cleanup(numQs, Δ, T, N, n_snapshot, resamples):
    # Paths to the data files
    data_dir = "data"
    pai_pattern = "pai_snap"
    lie_folder_path = os.path.join(data_dir, "lie")

    # Create a new folder for organizing data
    output_dir = os.path.join(data_dir, f"N-{N}-n-{n_snapshot}-r-{resamples}-Δ-{Δ}-T-{float(T)}-q-{numQs}")
    os.makedirs(output_dir, exist_ok=True)

    # Move files into the correct folders
    try:
        # Move TE-PAI snapshot files
        pai_files = [os.path.join(data_dir, f"{pai_pattern}{i}.csv") for i in range(n_snapshot + 1)]
        for file in pai_files:
            if os.path.exists(file):
                shutil.move(file, os.path.join(output_dir, os.path.basename(file)))

        # Move Lie-Trotter folder if it exists
        if os.path.exists(lie_folder_path):
            shutil.move(lie_folder_path, os.path.join(output_dir, "lie"))

    except Exception as move_error:
        print(f"Error while moving files: {move_error}")
    finally:
        print(f"All available files have been processed and moved to: {output_dir}")

def generate(numQs, Δ, T, N, n_snapshot, resamples):
    # Define the folder path
    folder_path = './data/lie'
    # Check if the folder exists, and if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print(f"Numqs: {numQs}, Δ: {Δ}, T: {T}, N: {N}, n_snapshot: {n_snapshot}, resamples: {resamples}")
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=numQs)
    # Initialize Hamiltonian and Trotter simulation
    # Assuming a spin chain Hamiltonian constructor
    hamil = Hamiltonian.spin_chain_hamil(numQs, freqs)
    te_pai = TE_PAI(hamil, numQs, Δ, T, N, n_snapshot)
    # Print expected number of gates and overhead
    print("Expected number of gates:", te_pai.expected_num_gates)
    print("Measurement overhead:", te_pai.overhead)
    # Run the TE-PAI simulation 
    te_pai.run_te_pai(resamples)
    
    # Use Lie Trotter to run the simulation
    trotter = Trotter(hamil, numQs, T, N, n_snapshot)
    res = [2 * prob - 1 for prob in trotter.run()]
    
    cleanup(numQs, Δ, T, N, n_snapshot, resamples)

if __name__ == "__main__":
    # Parameters for the example
    numQs = 5  # Number of qubits
    Δ = np.pi / (2**10)  # Delta parameter
    T = 1  # Total evolution time
    N = 1000  # Number of Trotter steps
    n_snapshot = 10  # Number of snapshots
    resamples = 10
    generate(numQs, Δ, T, N, n_snapshot, resamples)