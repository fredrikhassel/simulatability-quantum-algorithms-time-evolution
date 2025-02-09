import csv
import sys
import os
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

# Add the correct path to the system path
current_dir = os.path.abspath(os.path.dirname(__file__))
local_te_pai_path = os.path.join(current_dir, "te_pai-main", "te_pai")
sys.path.insert(0, local_te_pai_path)

# Import the local classes from the te_pai package
from hamil import Hamiltonian
from trotter import Trotter
from te_pai import TE_PAI
from sampling import resample
#from te_pai.te_pai import 

#from te_pai.hamil import Hamiltonian
#from te_pai.trotter import Trotter
from sampling import resample

if __name__ == "__main__":
    # Define the folder path
    folder_path = './data/lie'

    # Check if the folder exists, and if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Parameters for the example
    numQs = 7  # Number of qubits
    Δ = np.pi / (2**6)  # Delta parameter
    T = 1  # Total evolution time
    N = 50  # Number of Trotter steps
    n_snapshot = 22  # Number of snapshots
    resamples = 10
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=numQs)
 
    # Initialize Hamiltonian and Trotter simulation
    # Assuming a spin chain Hamiltonian constructor
    hamil = Hamiltonian.spin_chain_hamil(numQs, freqs)
    te_pai = TE_PAI(hamil, numQs, Δ, T, N, n_snapshot)

    # Print expected number of gates and overhead
    print("Expected number of gates:", te_pai.expected_num_gates)
    print("Measurement overhead:", te_pai.overhead)

    # Run the TE-PAI simulation and resample the results
    data, signs = te_pai.run_te_pai(resamples)
    res = [resample(d) for d in data]

    # Processing the signs
    max_length = max(len(sign) for sign in signs)
    # Ensure padding works for all types of arrays in signs
    sign_data = np.array([
        list(arr) + [0] * (max_length - len(arr)) for arr in signs
    ])

    print(np.shape(sign_data))

    # Compute mean and standard deviation for the resampled data
    mean, std = zip(*[(np.mean(y), np.std(y)) for y in res])

    print("Means of TE-PAI result:", mean)
    print("Standard deviations of TE-PAI result:", std)

    # Attempt to use Lie Trotter to run the simulation
    try:
        trotter = Trotter(hamil, numQs, T, N, n_snapshot)
        res = [2 * prob - 1 for prob in trotter.run()]
        mean, std = zip(*[(np.mean(y), np.std(y)) for y in res])
        print("Means of Trotter result:", mean)
    except ValueError as e:
        # Handle the error and notify the user
        print(f"Error in generating Lie-Trotter data: {e}")
        print("Skipping Lie-Trotter data generation.")

    # Paths to the data files
    data_dir = "data"
    pai_pattern = "pai_snap"
    lie_folder_path = os.path.join(data_dir, "lie")

    # Create a new folder for organizing data
    output_dir = os.path.join(data_dir, f"N-{N}-n-{n_snapshot}-r-{resamples}-Δ-{Δ}-T-{T}-q-{numQs}")
    os.makedirs(output_dir, exist_ok=True)

    # Save sign_data as CSV
    sign_data_file_path = os.path.join(output_dir, "sign_data.csv")
    try:
        # Write sign_data to CSV
        with open(sign_data_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(sign_data)
        print(f"sign_data successfully saved to: {sign_data_file_path}")
    except Exception as e:
        print(f"Error while saving sign_data: {e}")

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