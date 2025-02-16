import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
import quimb.tensor as qtn
import quimb as qu
from pyquest import unitaries
import time

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
        print(f"LOADDATA:  {len(value["gates_arr"])}")
        if value["gates_arr"] is None or value["sign_list"] is None:
            print(f"Warning: Incomplete dataset for parameters {key}. Missing files.")
    
    return data_storage

def toArray(data_storage):
    """Parse the data from the data storage dictionary and extract gates, signs, and overhead."""
    parsed_data = {}

    for params, data in data_storage.items():
        sign_list = data["sign_list"]
        gates_arr = data["gates_arr"]

        if sign_list is not None and gates_arr is not None:
            # Convert sign_list DataFrame to dictionary if it's a DataFrame
            if isinstance(sign_list, pd.DataFrame):
                sign_list_dict = sign_list.to_dict(orient="index")
            else:
                sign_list_dict = sign_list

            # Extract overhead (assuming "overhead" is a key in the sign_list data)
            overhead = sign_list_dict.get("overhead", None)
            if overhead is None:
                print(f"Error: Overhead not found in sign_list for parameters: {params}")
                raise KeyError(f"Overhead not found in sign_list for parameters: {params}")

            # Parse sign data (excluding "overhead")
            sign_data = {k: v for k, v in sign_list_dict.items() if k != "overhead"}
            
            # Parse gates array into a structured format¨
            gate_data = [[] for i in range(len(gates_arr))]
            for circuit_idx, circuit in gates_arr.items():
                snapshot_data = []
                for snap_idx, snapshot in circuit.items():
                    gate_list = []
                    for gate_idx, gate in snapshot.items():
                        #print(f"gate_data: {len(gate_data)} snap index: {snap_idx}")
                        gate_data[snap_idx-1].append((gate["gate_name"], gate["angle"], gate["qubits"]))
                    #snapshot_data.append(gate_list)
                #gate_data.append(snapshot_data)

            print(f"toArray: {len(gate_data)}")
            # Store the parsed data under the parameter key
            parsed_data[params] = {
                "gates_arr": gate_data,
                "sign_list": sign_data,
                "overhead": overhead  # Ensure overhead is included here
            }

    return parsed_data

def toQuimbMag(circuit_gates, n, gate_name_mapping, magnetization_operator, circuit):
    """Parse the 1 circuit to a magnetization measurement from quimb."""

    # Iterate over snapshots
    print(f"This circuit has {len(circuit_gates)} gates")
    for gate in circuit_gates:
        gate_name = gate[0]
        angle = gate[1]
        qubit_indices = gate[2]
        
        # Translate the gate name if it's in the mapping
        quimb_gate_name = gate_name_mapping.get(gate_name, gate_name)
        
        # Apply the gate to the circuit        
        if len(qubit_indices) == 1:
            circuit.apply_gate(gate_id=quimb_gate_name, qubits=[qubit_indices[0]], params=[angle])
        elif len(qubit_indices) == 2:
            circuit.apply_gate(gate_id=quimb_gate_name, qubits=qubit_indices, params=[angle])
        else:
            raise ValueError(f"Unsupported number of qubits for gate {quimb_gate_name}: {len(qubit_indices)}")

    state_vector = circuit.psi.to_dense()
    mag =  qu.expec(magnetization_operator, state_vector).real
    return mag

def getMags(gate_data, sign_data, n, T, dT):
    """
    Parse the arrays into quimb circuits for a single dataset.
    
    Args:
        gate_data: Nested list of gates for the circuits.
        n: Number of qubits.

    Returns:
        magnetizations: A list of the magnetization at the termination of each circuit.
    """

    # Calculate the magnetization
    magnetization_operator = sum(unitaries.Z([i]).as_matrix(n) for i in range(n))

    # Initialize the circuit with the specified number of qubits
    gate_name_mapping = {
        'XX': 'rxx',
        'YY': 'ryy',
        'ZZ': 'rzz',
    }

    n_timesteps = T / dT
    print(f"{T} / {dT} = {n_timesteps}")
    if  int(T % dT) != 0:
        print("Error: number of timesteps is a float.")

    n_paralell_circuits = len(gate_data) // n_timesteps
    print(f"{len(gate_data)} / {n_timesteps} = {n_paralell_circuits}")
    if int(len(gate_data) % n_timesteps) != 0:
        print("Error: number of paralell circuits is a float.")

    # Executing each run
    runs_magnetizations = []
    for i in range(int(n_paralell_circuits)):
        print(f"Running the {i}th run out of {n_paralell_circuits}.")
        current_circuit = qtn.Circuit(n)

        starting_gate = i * int(n_timesteps)  # Use correct interval
        final_gate = (i + 1) * int(n_timesteps)  # Ensuring correct slicing
        print(f"This run uses gates: {starting_gate} - {final_gate}")

        current_gates = gate_data[starting_gate:final_gate]
        
        magnetization = []
        for j, gates in enumerate(current_gates):
            circuit_num = starting_gate + j  # Correct circuit number indexing
            print(f"Running the {j}th circuit number {circuit_num}")
            sign = sign_data[str(circuit_num + 1)][0]
            magnetization.append(toQuimbMag(gates, n, gate_name_mapping, magnetization_operator, current_circuit) * sign)
        runs_magnetizations.append(magnetization)
        print(magnetization)
        del current_circuit

    runs_magnetizations_mean = np.mean(runs_magnetizations, axis=0)
    runs_magnetizations_std = np.std(runs_magnetizations, axis=0)

    return runs_magnetizations_mean, runs_magnetizations_std

def parse(relative_path, T):
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

    folder_path = os.path.abspath(relative_path)
    data_storage = loadData(folder_path)  # Load data from the folder.
    data_storage = toArray(data_storage)

    parsed_results = {}

    # Record the start time
    start_time = time.time()
    
    # Iterate through each dataset in data_storage
    i = 1
    for params, data in data_storage.items():
        N, n_snapshot, circuits, delta_name, dT, numQs = params

        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        print("Dataset: "+str(i)+" at: "+str(elapsed_time))
        i+=1

        # Extract gate_data, sign_data, and overhead from toArray
        gate_data = data["gates_arr"]
        sign_data = data["sign_list"]
        overhead = data["overhead"]

        # Parse gates and signs into quimb circuits
        magnetizations, stds = getMags(gate_data, sign_data, int(numQs), T, dT)

        # Store the results in the parsed_results dictionary
        parsed_results[params] = {
            "magnetizations": magnetizations, # stored run by run
            "stds": stds,
            "overhead": overhead,
            "params": params,
        }

    return parsed_results