import json
import os
import numpy as np
import re

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
if __name__ == '__main__':
    numQs = 7 # 4
    Δ = np.pi / (2**10)
    Δ_name = 'pi_over_' + str(2**10)
    T = 0.1
    N = 1000 # increase to 10000
    n_snapshot = 20
    circuits = 20
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=numQs)
    
    hamil = Hamiltonian.spin_chain_hamil(numQs, freqs)
    te_pai = TE_PAI(hamil, numQs, Δ, T, N, n_snapshot)
    overhead = te_pai.overhead
    print("Overhead: ", overhead)
    
    _, sign, sign_list, gates_arr = te_pai.run_te_pai(circuits)
    print(sign)
    
    trotterSimulation(hamil, 100, n_snapshot, circuits, Δ_name, T, numQs)
    trotterSimulation(hamil, 1000, n_snapshot, circuits, Δ_name, T, numQs)
    
    output_dir = os.path.join(current_dir, "data", "circuits")
    os.makedirs(output_dir, exist_ok=True)
    
    sign_file_path = os.path.join(output_dir, f"sign_list-N-{N}-n-{n_snapshot}-c-{circuits}-Δ-{Δ_name}-T-{T}-q-{numQs}.json")
    gates_file_path = os.path.join(output_dir, f"gates_arr-N-{N}-n-{n_snapshot}-c-{circuits}-Δ-{Δ_name}-T-{T}-q-{numQs}.json")
    
    sign_data = {"overhead": overhead}
    for i, sign_val in enumerate(sign):
        sign_data[str(i + 1)] = sign_val
    
    try:
        with open(sign_file_path, 'w') as file:
            json.dump(sign_data, file, indent=4)
        print(f"Sign list successfully saved to {sign_file_path}")
    except Exception as e:
        print(f"Error saving sign list file: {e}")
    
    gates_data = {}
    for circuit_idx, circuit in enumerate(gates_arr, start=1):
        circuit_data = {}
        for snap_idx, snapshot in enumerate(circuit, start=1):
            snapshot_data = {}
            for gate_idx, gate in enumerate(snapshot, start=1):
                snapshot_data[str(gate_idx)] = {
                    "gate_name": gate[0],
                    "angle": float(gate[1]),
                    "qubits": list(gate[2])
                }
            circuit_data[str(snap_idx)] = snapshot_data
        gates_data[str(circuit_idx)] = circuit_data
    
    try:
        with open(gates_file_path, 'w') as file:
            json.dump(gates_data, file, indent=4)
        print(f"Gates array successfully saved to {gates_file_path}")
    except Exception as e:
        print(f"Error saving gates array file: {e}")
    
    folder_path = './data'
    clearDataFolder(folder_path)
