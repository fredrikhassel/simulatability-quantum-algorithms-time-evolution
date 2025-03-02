import json
import os
import numpy as np
import re
import gc
import h5py

# Import the local TE-PAI classes
from HAMILTONIAN import Hamiltonian
from main import TE_PAI
from TROTTER import Trotter

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

def save_to_hdf5(N, n, c, Δ, T, q, sign_dict, gates_data, folder_path):
    sign_file_path = os.path.join(folder_path, "sign_list.json")
    gates_file_path = os.path.join(folder_path, "gates.h5")

    # Save signs and parameters to JSON
    metadata = {
        "N": N,
        "n": n,
        "c": c,
        "Δ": Δ,
        "T": T,
        "q": q,
        "signs": sign_dict
    }
    
    with open(sign_file_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Sign list successfully saved to {sign_file_path}")

    # Save gates to HDF5
    with h5py.File(gates_file_path, 'w') as hdf_file:
        hdf_file.attrs.update({"N": N, "n": n, "c": c, "Δ": Δ, "T": T, "q": q})
        circuits_group = hdf_file.create_group("circuits")

        for circuit_idx, circuit in gates_data.items():
            circuit_group = circuits_group.create_group(str(circuit_idx))

            for snap_idx, snapshot in circuit.items():
                json_data = json.dumps(snapshot).encode('utf-8')
                circuit_group.create_dataset(str(snap_idx), data=json_data)

    print(f"Gates data successfully saved to {gates_file_path}")

current_dir = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    # Parameters
    numQs = 4
    Δ = np.pi / (2**10)
    Δ_name = 'pi_over_' + str(2**10)
    T = 0.1
    dT = 0.01
    finalTimes = np.arange(dT, T + dT, dT)
    N = 1000
    n_snapshot = 1
    circuit_pool_size = 1000

    # Setting up TE-PAI
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=numQs)
    hamil = Hamiltonian.spin_chain_hamil(numQs, freqs)

    # Prepping output directory
    output_dir = os.path.join(current_dir, "data", "circuits")
    os.makedirs(output_dir, exist_ok=True)
    folder_name = f"N-{N}-n-{n_snapshot}-p-{circuit_pool_size}-Δ-{Δ_name}-q-{numQs}-dT-{dT}-T-{T}"
    folder_path = os.path.join(output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Running TE-PAI
    te_pai = TE_PAI(hamil, numQs, Δ, dT, N, n_snapshot)
    overhead = te_pai.overhead
    _, sign, sign_list, gates_arr = te_pai.run_te_pai(circuit_pool_size)

    print("Overhead: ", overhead)

    # Convert sign list to dictionary format
    sign_data = {"overhead": overhead}
    for i, sign_val in enumerate(sign):
        sign_data[str(i + 1)] = sign_val

    # Convert gates array to nested dictionary format
    gates_data = {}
    for circuit_idx, circuit in enumerate(gates_arr, start=1):
        gates_data[circuit_idx] = {
            snap_idx + 1: [
                {"gate_name": gate[0], "angle": float(gate[1]), "qubits": list(gate[2])}
                for gate in snapshot
            ]
            for snap_idx, snapshot in enumerate(circuit)
        }

    # Save to HDF5 and JSON
    save_to_hdf5(N, n_snapshot, circuit_pool_size, Δ, T, numQs, sign_data, gates_data, folder_path)

    # Explicitly free memory
    del sign_list, gates_arr, sign
    gc.collect()

    clearDataFolder('./data')
