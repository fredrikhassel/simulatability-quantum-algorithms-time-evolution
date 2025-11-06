import json
import os
import numpy as np
import re
import gc
import csv

# Import the local TE-PAI classes
from HAMILTONIAN import Hamiltonian
from main import TE_PAI
from TROTTER import Trotter

def cleanup_temp_files(directory="data", prefix="temp_"):
    """
    Removes all temporary files in the given directory that start with the given prefix.
    By default, this will clean up temp_result_*.npy and temp_gates_*.json.
    """
    try:
        temp_pattern = re.compile(rf"^{prefix}.*\.(npy|json)$")
        removed_files = 0

        for fname in os.listdir(directory):
            if temp_pattern.match(fname):
                full_path = os.path.join(directory, fname)
                os.remove(full_path)
                removed_files += 1

        print(f"[Cleanup] Removed {removed_files} temporary files from {directory}")

    except Exception as e:
        print(f"[Cleanup Error] Failed to remove temporary files: {e}")

def clearDataFolder(folder_path):
    pattern = re.compile(r'^pai_snap\d+\.csv$')
    try:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if pattern.match(filename):
                    file_path = os.path.join(folder_path, filename)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
            print("Cleanup complete.")
        else:
            print(f"Folder not found: {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def trotterSimulation(hamil, N, n_snapshot, c, Δ_name, T, numQs):
    try:
        trotter = Trotter(hamil, N, n_snapshot, c, Δ_name, T, numQs)
        res = [2 * prob - 1 for prob in trotter.run()]
        mean, std = zip(*[(np.mean(y), np.std(y)) for y in res])
        print("Means of Trotter result:", mean)
    except ValueError as e:
        print(f"Error in generating Lie-Trotter data: {e}")
        print("Skipping Lie-Trotter data generation.")

current_dir = os.path.abspath(os.path.dirname(__file__))
def generate(params,n_workers=None):
    # Parameters
    print(params)
    numQs, Δ, T, dT, N, circuit_pool_size, H_name = params
    Δ_name = 'pi_over_' + str(2**Δ)
    Δ = np.pi / (2**Δ)
    finalTimes = np.arange(dT, T + dT, dT)
    n_snapshot = 1

    # Setting up TE-PAI
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=numQs)

    if H_name == "SCH":
        hamil = Hamiltonian.spin_chain_hamil(numQs, freqs)
    elif H_name == "NNN":
        hamil = Hamiltonian.next_nearest_neighbor_hamil(numQs, freqs)
    elif H_name == "2D":
        hamil = Hamiltonian.lattice_2d_hamil(numQs, freqs=freqs)
    else:
        raise ValueError(f"Hamiltonian '{H_name}' not recognized.")

    # Prepping output directory
    if H_name == "NNN":
        output_dir = os.path.join(current_dir, "NNN_data", "circuits")
    elif H_name == "SCH":
        output_dir = os.path.join(current_dir, "data", "circuits")
    elif H_name == "2D":
        output_dir = os.path.join(current_dir, "2D_data", "circuits")

    os.makedirs(output_dir, exist_ok=True)

    folder_name = f"N-{N}-n-{n_snapshot}-p-{circuit_pool_size}-Δ-{Δ_name}-q-{numQs}-dT-{dT}-T-{T}"
    folder_path = os.path.join(output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Running TE-PAI
    te_pai = TE_PAI(hamil, numQs, Δ, dT, N, n_snapshot)

    # Define file paths inside the new folder
    sign_file_path = os.path.join(
        folder_path,
        f"sign_list-N-{N}-n-{n_snapshot}-p-{circuit_pool_size}-Δ-{Δ_name}-T-{T}-q-{numQs}.json"
    )
    gates_file_path = os.path.join(
        folder_path,
        f"gates_arr-N-{N}-n-{n_snapshot}-p-{circuit_pool_size}-Δ-{Δ_name}-T-{T}-q-{numQs}.json"
    )

    def _to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        # Fallback for custom classes (e.g., term objects)
        return repr(obj)

    csv_file_path = os.path.join(
        folder_path,
        f"hamil_meta-N-{N}-n-{n_snapshot}-p-{circuit_pool_size}-Δ-{Δ_name}-T-{T}-q-{numQs}.csv"
    )

    # Safely JSON-encode both payloads so they fit cleanly into two CSV rows
    freqs_json = json.dumps(freqs.tolist())
    terms_json = json.dumps(getattr(hamil, "terms", None), default=_to_serializable)

    with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "value_json"])
        w.writerow(["freqs", freqs_json])
        w.writerow(["hamil.terms", terms_json])

    print(f"[Info] Wrote {csv_file_path}")

    # Execute TE-PAI and cleanup
    te_pai.run_te_pai(
        num_circuits=circuit_pool_size,
        sign_file_path=sign_file_path,
        gates_file_path=gates_file_path,
        overhead=te_pai.overhead,
        verbose=True,
        n_workers=n_workers
    )
    clearDataFolder('./data')


if __name__ == '__main__':
    if True:
        # Example parameters
        numQs = 12
        Δ = 6
        T = 1.0
        dT = 0.1
        N = 1000
        circuit_pool_size = 10
        H_name = "2D"
        # Generate circuits with the specified parameters
        generate((numQs, Δ, T, dT, N, circuit_pool_size, H_name))
    