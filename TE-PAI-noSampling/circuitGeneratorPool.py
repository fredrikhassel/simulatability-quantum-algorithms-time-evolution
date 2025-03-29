import json
import os
import numpy as np
import re
import gc

# Import the local TE-PAI classes
from HAMILTONIAN import Hamiltonian
from main import TE_PAI
from TROTTER import Trotter

import os
import re

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
if __name__ == '__main__':
    # Parameters
    numQs = 10
    Δ = np.pi / (2**10)
    Δ_name = 'pi_over_' + str(2**10)
    T = 10
    dT = 1
    finalTimes = np.arange(dT,T+dT,dT)
    N = 1000
    n_snapshot = 1
    circuit_pool_size = 100

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

    # Define file paths inside the new folder
    sign_file_path = os.path.join(folder_path, f"sign_list-N-{N}-n-{n_snapshot}-p-{circuit_pool_size}-Δ-{Δ_name}-T-{T}-q-{numQs}.json")
    gates_file_path = os.path.join(folder_path, f"gates_arr-N-{N}-n-{n_snapshot}-p-{circuit_pool_size}-Δ-{Δ_name}-T-{T}-q-{numQs}.json")  

    te_pai.run_te_pai(
        num_circuits=circuit_pool_size,
        sign_file_path=sign_file_path,
        gates_file_path=gates_file_path,
        overhead=te_pai.overhead
    )
    clearDataFolder('./data')

    if False:
        sign_list, gates_arr = te_pai.run_te_pai(circuit_pool_size)
        cleanup_temp_files()

        print("Overhead: ", overhead)   

        # Define file paths inside the new folder
        sign_file_path = os.path.join(folder_path, f"sign_list-N-{N}-n-{n_snapshot}-p-{circuit_pool_size}-Δ-{Δ_name}-T-{T}-q-{numQs}.json")
        gates_file_path = os.path.join(folder_path, f"gates_arr-N-{N}-n-{n_snapshot}-p-{circuit_pool_size}-Δ-{Δ_name}-T-{T}-q-{numQs}.json")    
        
        # Save sign list
        sign_data = {"overhead": overhead}
        for i, sign_val in enumerate(sign_list):
            sign_data[str(i + 1)] = sign_val    
        try:
            with open(sign_file_path, 'w') as file:
                json.dump(sign_data, file, indent=4)
            print(f"Sign list successfully saved to {sign_file_path}")
        except Exception as e:
            print(f"Error saving sign list file: {e}")  
        # Save gates array incrementally
        try:
            with open(gates_file_path, 'w') as file:
                file.write('{\n')  # Start JSON manually
                first_circuit = True    
                for circuit_idx, circuit in enumerate(gates_arr, start=1):
                    if not first_circuit:
                        file.write(',\n')
                    first_circuit = False
                    file.write(f'"{circuit_idx}":{{\n') 
                    first_snapshot = True
                    for snap_idx, snapshot in enumerate(circuit, start=1):
                        if not first_snapshot:
                            file.write(',\n')
                        first_snapshot = False
                        file.write(f'"{snap_idx}":{{')  
                        gate_entries = []
                        for gate_idx, gate in enumerate(snapshot, start=1):

                            if not isinstance(gate, (list, tuple)) or len(gate) != 3:
                                print(f"Malformed gate at snapshot {snap_idx}, gate_idx {gate_idx}: {gate}")

                            gate_entries.append(f'"{gate_idx}":{{"gate_name":"{gate[0]}","angle":{float(gate[1])},"qubits":{list(gate[2])}}}')

                        file.write(",".join(gate_entries))
                        file.write('}')

                    file.write('}') 
                file.write('\n}')
            print(f"Gates array successfully saved to {gates_file_path}")
        except Exception as e:
            print(f"Error saving gates array file: {e}")    
        # Explicitly free memory
        del sign_list, gates_arr
        gc.collect()    
        clearDataFolder('./data')