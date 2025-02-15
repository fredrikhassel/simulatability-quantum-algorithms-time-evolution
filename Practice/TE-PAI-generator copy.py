import csv
import sys
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
from memory_profiler import profile

# Add the correct path to the system path
current_dir = os.path.abspath(os.path.dirname(__file__))
local_te_pai_path = os.path.join(current_dir, "te_pai-main", "te_pai")
sys.path.insert(0, local_te_pai_path)

# Import the local classes from the te_pai package
from hamil import Hamiltonian
from trotter import Trotter
from te_pai import TE_PAI
from sampling import resample

@profile
def main():
    # Define the folder path
    folder_path = './data/lie'
    os.makedirs(folder_path, exist_ok=True)

    # Parameters
    numQs = 7  # Number of qubits
    Δ = np.pi / (2**6)  # Delta parameter
    T = 2  # Total evolution time
    N = 50  # Number of Trotter steps
    n_snapshot = 10  # Number of snapshots
    resamples = 100

    # Initialize Hamiltonian
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=numQs)
    hamil = Hamiltonian.spin_chain_hamil(numQs, freqs)
    
    # TE-PAI simulation setup
    te_pai = TE_PAI(hamil, numQs, Δ, T, N, n_snapshot)

    # Print expected number of gates
    print(f"Expected number of gates: {te_pai.expected_num_gates}")
    print(f"Measurement overhead: {te_pai.overhead}")

    # Run TE-PAI and measure memory usage
    data, signs = run_te_pai_memory_debug(te_pai, resamples)

    # Compute mean and standard deviation
    process_te_pai_results(data)

    # Free memory
    del data, signs

    # Run Lie Trotter simulation
    run_trotter_simulation(hamil, numQs, T, N, n_snapshot)

    # Organize data files
    move_output_files(N, n_snapshot, resamples, Δ, T, numQs)

@profile
def run_te_pai_memory_debug(te_pai, resamples):
    """ Run TE-PAI simulation and check memory usage. """
    print("\nRunning TE-PAI simulation...")
    data, signs = te_pai.run_te_pai(resamples)

    # Debug data size
    print(f"Data Type: {type(data)}, Signs Type: {type(signs)}")
    print(f"Total Snapshots: {len(data)}")
    print(f"Total Elements in Data: {sum(len(d) for d in data)}")

    for i, d in enumerate(data[:3]):  # Print first 3 snapshots for inspection
        print(f"Snapshot {i}: shape {np.shape(d)}, dtype {d.dtype if isinstance(d, np.ndarray) else type(d)}")

    return data, signs

@profile
def process_te_pai_results(data):
    """ Compute mean and std deviation while optimizing memory. """
    print("\nProcessing TE-PAI results...")
    
    res = (resample(d) for d in data)  # Use generator instead of list
    means, stds = zip(*((np.mean(y), np.std(y)) for y in res))
    
    print(f"Means of TE-PAI result: {means}")
    print(f"Standard deviations of TE-PAI result: {stds}")

@profile
def run_trotter_simulation(hamil, numQs, T, N, n_snapshot):
    """ Run Trotter simulation and check memory usage. """
    print("\nRunning Lie-Trotter simulation...")

    try:
        trotter = Trotter(hamil, numQs, T, N, n_snapshot)
        results = trotter.run()

        print(f"Trotter results type: {type(results)}")
        print(f"Trotter results length: {len(results)}")

        # Compute mean and std
        res = (2 * prob - 1 for prob in results)  # Use generator
        means, stds = zip(*((np.mean(y), np.std(y)) for y in res))

        print(f"Means of Trotter result: {means}")
    except ValueError as e:
        print(f"Error in generating Lie-Trotter data: {e}")
        print("Skipping Lie-Trotter data generation.")

@profile
def move_output_files(N, n_snapshot, resamples, Δ, T, numQs):
    """ Move output files into organized folders while optimizing memory usage. """
    print("\nOrganizing data files...")

    data_dir = "data"
    pai_pattern = "pai_snap"
    lie_folder_path = os.path.join(data_dir, "lie")

    output_dir = os.path.join(data_dir, f"N-{N}-n-{n_snapshot}-r-{resamples}-Δ-{Δ}-T-{T}-q-{numQs}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        pai_files = (os.path.join(data_dir, f"{pai_pattern}{i}.csv") for i in range(n_snapshot + 1))
        for file in pai_files:
            if os.path.exists(file):
                shutil.move(file, os.path.join(output_dir, os.path.basename(file)))

        if os.path.exists(lie_folder_path):
            shutil.move(lie_folder_path, os.path.join(output_dir, "lie"))

    except Exception as move_error:
        print(f"Error while moving files: {move_error}")
    finally:
        print(f"All files processed and moved to: {output_dir}")

if __name__ == "__main__":
    main()
