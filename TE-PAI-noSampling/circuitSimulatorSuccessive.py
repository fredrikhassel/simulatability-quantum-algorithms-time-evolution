import csv
import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
import quimb.tensor as qtn
import quimb as qu
from pyquest import unitaries
from matplotlib import pyplot as plt

def loadData(folder_path):
    """Load JSON data from a folder and organize it by parameter sets."""
    
    # Regular expression pattern to extract the shared parameters from filenames
    pattern = r"(sign_list|gates_arr)-N-(\d+)-n-(\d+)-c-(\d+)-Δ-([\w\-]+)-T-([\d\.]+)-q-(\d+)\.json"
    
    # Dictionary to store matched data by parameter set
    data_storage = defaultdict(lambda: {"sign_list": None, "gates_arr": None})
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return data_storage  # Return empty storage if folder not found
    
    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Match the file name against the pattern
        match = re.match(pattern, filename)
        if match:

            file_type, N, n_snapshot, circuits, delta_name, T, numQs = match.groups()
            
            # Convert parameters to appropriate types
            N = int(N)
            n_snapshot = int(n_snapshot)
            circuits = int(circuits)
            T = float(T) if "." in T else int(T)  # Convert T to float if it contains a dot
            numQs = int(numQs)
            
            # Create a unique key for the dataset
            key = (N, n_snapshot, circuits, delta_name, T, numQs)
            
            # Load the JSON file into a DataFrame
            try:
                df = pd.read_json(file_path, orient="index")
                data_storage[key][file_type] = df  # Store under the appropriate type
            except ValueError as e:
                print(f"Error reading JSON file {filename}: {e}")
    
    # Check for missing pairs and warn
    for key, value in data_storage.items():
        if value["gates_arr"] is None or value["sign_list"] is None:
            print(f"Warning: Incomplete dataset for parameters {key}. Missing files.")
    
    return data_storage

def toArray(data_storage):
    """Parse the data from the data storage dictionary and extract gates, signs, and overhead."""
    parsed_data = {}

    for timestep_params, timestep_data in data_storage.items():
        sign_list = timestep_data["sign_list"]
        gates_arr = timestep_data["gates_arr"]

        if sign_list is not None and gates_arr is not None:
            # Convert sign_list DataFrame to dictionary if it's a DataFrame
            if isinstance(sign_list, pd.DataFrame):
                sign_list_dict = sign_list.to_dict(orient="index")
            else:
                sign_list_dict = sign_list

            # Extract overhead (assuming "overhead" is a key in the sign_list data)
            overhead = sign_list_dict.get("overhead", None)[0]
            if overhead is None:
                print(f"Error: Overhead not found in sign_list for parameters: {timestep_params}")
                raise KeyError(f"Overhead not found in sign_list for parameters: {timestep_params}")

            # Parse sign data (excluding "overhead")
            sign_data = {k: v for k, v in sign_list_dict.items() if k != "overhead"}
            
            # Parse gates array into a structured format¨
            gate_data = [[] for i in range(len(gates_arr))]
            for circuit_idx, circuit in gates_arr.items():
                snapshot_data = []
                for snap_idx, snapshot in circuit.items():
                    gate_list = []
                    for gate_idx, gate in snapshot.items():
                        # Sorting the circuits
                        gate_data[snap_idx-1].append((gate["gate_name"], gate["angle"], gate["qubits"]))

            # Store the parsed data under the parameter key
            parsed_data[timestep_params] = {
                "gates_arr": gate_data,
                "sign_list": sign_data,
                "overhead": overhead  # Ensure overhead is included here
            }

    return parsed_data

def toQuimbMag(circuit_gates, n, gate_name_mapping, circuit):
    """Parse the 1 circuit to a magnetization measurement from quimb."""

    # Iterate over snapshots
    for gate in circuit_gates:
        gate_name = gate[0]
        angle = gate[1]
        qubit_indices = gate[2]
        
        # Translate the gate name if it's in the mapping
        quimb_gate_name = gate_name_mapping[gate_name]
        
        # Apply the gate to the circuit        
        if len(qubit_indices) == 1:
            circuit.apply_gate(gate_id=quimb_gate_name, qubits=[qubit_indices[0]], params=[angle])
        elif len(qubit_indices) == 2:
            circuit.apply_gate(gate_id=quimb_gate_name, qubits=qubit_indices, params=[angle])
        else:
            raise ValueError(f"Unsupported number of qubits for gate {quimb_gate_name}: {len(qubit_indices)}")

    mag = measure(circuit, n)
    return mag

def measure(circuit,q):
    magnetization_operator = unitaries.Z([0]).as_matrix(q)
    state_vector = circuit.psi.to_dense()
    expect = qu.expec(state_vector, magnetization_operator).real
    return (expect+1) / 2

def getMags(gate_data, sign_data, q):
    """
    Parse the arrays into quimb circuits for a single dataset.
    
    Args:
        gate_data: Nested list of gates for the circuits.
        n: Number of qubits.

    Returns:
        magnetizations: A list of the magnetization at the termination of each circuit.
    """

    # Initialize the circuit with the specified number of qubits
    gate_name_mapping = {
        'XX': 'rxx',
        'YY': 'ryy',
        'ZZ': 'rzz',
        'Z': 'rz'
    }

    runs_magnetizations = []

    for i,gates in enumerate(gate_data):
        #print("This circuit has length: ",len(gates))
        sign = sign_data[str(i+1)][0]
        current_circuit = qtn.Circuit(q)
        runs_magnetizations.append(toQuimbMag(gates, q, gate_name_mapping, current_circuit) * int(sign))
        del current_circuit

    runs_magnetizations_mean = np.mean(runs_magnetizations)
    runs_magnetizations_std = np.std(runs_magnetizations)

    print("Average: ",runs_magnetizations_mean)

    return runs_magnetizations_mean, runs_magnetizations_std

def parse(relative_path):
    """
    Parse the data from the specified folder path and return a dictionary containing the parsed results.
    
    Args:
        relative_path: The relative path to the folder containing JSON files.

    Returns:
        parsed_results: A dictionary where:
            - The key is the parameter tuple (N, n_snapshot, circuits, delta_name, T, numQs).
            - The value is another dictionary containing:
                - "magnetizations": The magnetization expected values stored run by run.
                - "overhead": The overhead value extracted from the sign data.
                - "params": The parameter tuple for the dataset.
                - "sign_list": The raw sign data.
    """
    # Extracting data
    folder_path = os.path.abspath(relative_path)
    data_storage = loadData(folder_path)  # Load data from the folder.
    data_storage = toArray(data_storage) # key: timestep_params, value: "gates_arr": gate_data, "sign_list": sign_data, "overhead": overhead

    # Preparing calculations
    #start_time = time.time()
    parsed_results = {}
    
    # Iterate through each timestep in data_storage
    for params, data in data_storage.items():
        N, n_snapshot, circuits, delta_name, T, numQs = params # Extracting parameters
        print("Calculating the data for: ", params)

        # Extract gate_data, sign_data, and overhead from toArray
        gate_data = data["gates_arr"]
        sign_data = data["sign_list"]
        overhead = data["overhead"]

        # Parse gates and signs into quimb circuits
        magnetizations, stds = getMags(gate_data, sign_data, int(numQs))

        # Store the results in the parsed_results dictionary
        parsed_results[params] = {
            "magnetizations": magnetizations, # stored run by run
            "stds": stds,
            "overhead": overhead,
            "params": params,
        }

    return parsed_results


def plot_magnetization(parsed_results, dT):
    """
    Plots magnetization data with standard deviations as error bars.
    
    Args:
        parsed_results: A dictionary where keys are parameter tuples and values contain
                        "magnetizations" and "stds" lists.
    """
    # Extract T values, magnetization means, and stds
    T_values = []
    magnetization_means = []
    magnetization_stds = []
    
    for params, data in parsed_results.items():
        N, n_snapshot, circuits_count, delta_name, T, q = params
        T_values.append(params[4])  # Extract temperature (T) from the parameter tuple
        magnetization_means.append(data["magnetizations"])  # Magnetization mean is already calculated
        magnetization_stds.append(data["stds"])  # Standard deviation is already calculated
    
    # Sort data by T values
    sorted_indices = sorted(range(len(T_values)), key=lambda i: T_values[i])
    T_values = [T_values[i] for i in sorted_indices]
    magnetization_means = [magnetization_means[i] for i in sorted_indices]
    magnetization_stds = [magnetization_stds[i] for i in sorted_indices]

    # Saving the data
    output_dir = os.path.join("TE-PAI-noSampling", "data", "plotting")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    filename = f"N-{N}-n-{n_snapshot}-c-{circuits_count}-Δ-{delta_name}-T-{np.max(T_values)}-q-{q}-dT-{dT}.csv"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["T", "Magnetization", "Std Dev"])
            for T, mag, std in zip(T_values, magnetization_means, magnetization_stds):
                writer.writerow([T, mag, std])

    # Plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(T_values, magnetization_means, yerr=magnetization_stds, fmt='o-', capsize=5, label='Magnetization')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Magnetization")
    plt.title("Magnetization vs. Temperature")
    plt.legend()
    plt.grid()
    plt.show()

plot_magnetization(parse('TE-PAI-noSampling/data/circuits/N-1000-n-1-c-50-Δ-pi_over_1024-q-4-dT-0.01-T-0.1/'), 0.01)