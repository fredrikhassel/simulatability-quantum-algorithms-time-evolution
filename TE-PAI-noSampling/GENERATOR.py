import csv
import sys
import os
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

# Import the local TE-PAI classes
from HAMILTONIAN import Hamiltonian
from TROTTER import Trotter
from main import TE_PAI

# Add the correct path to the system path
current_dir = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':

    # Parameters for circuit generation
    numQs       = 7                 # Number of qubits
    Δ           = np.pi / (2**6)    # Delta parameter
    Δ_name      = 'pi_over_'+str(2**6)
    T           = 1                 # Total evolution time
    N           = 50                # Number of Trotter steps
    n_snapshot  = 10                # Number of snapshots
    circuits    = 10               # Number of circuits    
    rng         = np.random.default_rng(0)
    freqs       = rng.uniform(-1, 1, size=numQs)

    # Initialize Hamiltonian and Trotter simulation
    # Assuming a spin chain Hamiltonian constructor
    hamil = Hamiltonian.spin_chain_hamil(numQs, freqs)
    te_pai = TE_PAI(hamil, numQs, Δ, T, N, n_snapshot)

    # Run the TE-PAI simulation and get the circuit details
    _, sign, sign_list, gates_arr = te_pai.run_te_pai(circuits)
    sign = np.prod(sign)
    print(np.shape(sign_list))

    # Ensure directories for saving data exist
    output_dir = os.path.join(current_dir, "data", "circuits")
    os.makedirs(output_dir, exist_ok=True)

    # Filenames for sign_list and gates_arr
    sign_file_path = os.path.join(
        output_dir, f"sign_list-N-{N}-n-{n_snapshot}-c-{circuits}-Δ-{Δ_name}-T-{T}-q-{numQs}.csv"
    )
    gates_file_path = os.path.join(
        output_dir, f"gates_arr-N-{N}-n-{n_snapshot}-c-{circuits}-Δ-{Δ_name}-T-{T}-q-{numQs}.csv"
    )

    # Save sign_list to CSV
    try:
        with open(sign_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Index", "Sign"])
            for i, sign_val in enumerate(sign_list):
                writer.writerow([i, sign_val])
        print(f"Sign list successfully saved to {sign_file_path}")
    except Exception as e:
        print(f"Error saving sign list file: {e}")

    # Save gates_arr to CSV
    try:
        with open(gates_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Index", "Gates_Arr"])
            for i, gate_row in enumerate(gates_arr):
                writer.writerow([i, str(gate_row)])  # Convert each row to string to avoid CSV issues
        print(f"Gates array successfully saved to {gates_file_path}")
    except Exception as e:
        print(f"Error saving gates array file: {e}")