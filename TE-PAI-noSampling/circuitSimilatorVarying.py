import os
import re
from matplotlib import pyplot as plt
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
            T = np.round(float(T), 2) if "." in T else int(T)  # Convert T to float if it contains a dot
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

def extract_data(data_storage):
    """
    Extracts parameter tuples, overheads, sign_lists, and gate_arrs from data_storage.

    Args:
        data_storage: Dictionary containing datasets with keys as parameter tuples.

    Returns:
        parameters: List of parameter tuples.
        overheads: List of overhead values.
        sign_lists: List of sign_list DataFrames.
        gate_arrs: List of gate_arr DataFrames.
    """

    parameters = []
    sign_lists = []
    gate_arrs = []

    for key, value in data_storage.items():
        if value["sign_list"] is not None and value["gates_arr"] is not None:
            parameters.append(key)
            sign_lists.append(value["sign_list"])
            gate_arrs.append(toArray(value["gates_arr"]))
        else:
            print(f"Warning: Incomplete data for {key}")

    overheads = [sign[0][0] for sign in sign_lists]
    sign_lists = [sign[1:] for sign in sign_lists]

    # Convert to numpy arrays
    parameters = np.array(parameters, dtype=object)
    overheads = np.array(overheads)
    sign_lists = np.array(sign_lists, dtype=object)
    gate_arrs = np.array(gate_arrs, dtype=object)

    # Check if all arrays have the same length
    lengths = [len(parameters), len(overheads), len(sign_lists), len(gate_arrs)]
    print(f"Lengths: Parameters={lengths[0]}, Overheads={lengths[1]}, Sign Lists={lengths[2]}, Gate Arrs={lengths[3]}")

    if len(set(lengths)) == 1:
        print("All arrays have the same length.")
    else:
        print("Length mismatch in extracted data!")

    return parameters, overheads, sign_lists, gate_arrs

def toArray(gate_dict):
    gate_arr = []
    for timestep_idx, timestep in gate_dict.items():
        timestep_data = []
        for circuit_idx, circuit in timestep.items():
            circuit_gates = []
            for gate_idx, gate in circuit.items():
                circuit_gates.append((gate["gate_name"], gate["angle"], gate["qubits"]))
            timestep_data.append(circuit_gates)
        gate_arr.append(timestep_data)  

    return gate_arr

def formatData(parameters, sign_lists, gate_arrs):
    n_timesteps = len(parameters)
    n_circuits = len(sign_lists[0])  # Assume all timesteps have same circuit count
    
    circuits_gates = [[] for _ in range(n_circuits)]
    circuits_signs = [[] for _ in range(n_circuits)]
    
    for i in range(n_timesteps):
        for j in range(n_circuits):
            try:
                gate = gate_arrs[i][0][j]  # Ensure this index is valid
            except IndexError:
                print(f"Index error: gate_arrs[{i}][0][{j}] does not exist")
                continue  # Skip invalid indices

            circuits_gates[j].append(gate)
            circuits_signs[j].append(sign_lists[i][j])

    return circuits_gates, circuits_signs

def toQuimbMag(circuit_gates, gate_name_mapping, magnetization_operator, circuit):
    """Parse the 1 circuit to a magnetization measurement from quimb."""

    # Iterate over snapshots
    for gate in circuit_gates[0]:
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
    del circuit
    return mag

def getMags(circuits_gates, circuits_signs, q):
    # Calculate the magnetization
    magnetization_operator = sum(unitaries.Z([i]).as_matrix(q) for i in range(q))

    gate_name_mapping = {
        'XX': 'rxx',
        'YY': 'ryy',
        'ZZ': 'rzz',
    }

    magnetizations = [[] for i in range(len(circuits_signs))]

    for i, (gates, signs) in enumerate(zip(circuits_gates, circuits_signs)):
        print(f"Calculating magnetization of circuit {i}")
        current_circuit = qtn.Circuit(q)
        magnetization = []
        for j, (gate, sign) in enumerate(zip(gates, signs)):
            print(f"Gate: {j}")
            sign = sign[0]
            magnetization.append(toQuimbMag(gates, gate_name_mapping, magnetization_operator, current_circuit) * sign)
        magnetizations[i] = magnetization
        print(magnetization)
    return magnetizations

def parse(folder_path, dT):
    # Extracting data
    data_storage = loadData(folder_path)
    parameters, overheads, sign_lists, gate_arrs = extract_data(data_storage)
    Ts = [p[4] for p in parameters]

    # Preparing timestep data
    N, n_snapshot, circuits_count, delta_name, _, q = parameters[0] # should all be the same
    print(np.shape(parameters), np.shape(sign_lists), len(gate_arrs))
    circuits_gates, circuits_signs  = formatData(parameters, sign_lists, gate_arrs)

    magnetizations = getMags(circuits_gates, circuits_signs, q)
    averages = np.mean(magnetizations, axis=0) / q
    stds = np.std(magnetizations, axis=0) / q

    # Plotting
    plt.errorbar(Ts, averages, yerr=stds, fmt='o', ecolor='red', capsize=4, label="Data with Error Bars")
    plt.xlabel('Time (t)')
    plt.ylabel('Magnetization')
    plt.title('Plot of Magnetization vs Time')
    plt.legend()
    plt.show()

    # Define the output directory
    output_dir = os.path.join('TE-PAI-noSampling', 'data', 'plotting')
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    # Save plotting data to a CSV file
    filename = f"N-{N}-n-{n_snapshot}-c-{circuits_count}-Δ-{delta_name}-T-{np.max(Ts)}-q-{q}-dT-{dT}.csv"
    output_path = os.path.join(output_dir, filename)

    # Create DataFrame for plotting data
    data_dict = {
        'x': Ts,
        'y': averages,
        'errorbars': stds
    }
    df_plot = pd.DataFrame(data_dict)

    try:
        df_plot.to_csv(output_path, index=False)
        print(f"Plotting data saved to: {output_path}")
    except Exception as e:
        print(f"Error saving plotting data: {e}")

    return

parse('TE-PAI-noSampling/data/circuits/N-1000-n-1-c-10-Δ-pi_over_1024-q-4-dT-0.0025-T-0.1/', 0.0025)









        
    







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
            
            # Parse gates array into a structured format
            gate_data = []
            for circuit_idx, circuit in gates_arr.items():
                snapshot_data = []
                for snap_idx, snapshot in circuit.items():
                    gate_list = []
                    for gate_idx, gate in snapshot.items():
                        gate_list.append((gate["gate_name"], gate["angle"], gate["qubits"]))
                    snapshot_data.append(gate_list)
                gate_data.append(snapshot_data)

            # Store the parsed data under the parameter key
            parsed_data[params] = {
                "gates_arr": gate_data,
                "sign_list": sign_data,
                "overhead": overhead  # Ensure overhead is included here
            }

    return parsed_data



def getMags(gate_data, n):
    """
    Parse the arrays into quimb circuits for a single dataset.
    
    Args:
        gate_data: Nested list of gates for the circuits.
        n: Number of qubits.

    Returns:
        magnetizations: A list of the magnetization at the termination of each circuit.
    """
    magnetizations = []

    # Initialize the circuit with the specified number of qubits
    gate_name_mapping = {
        'XX': 'rxx',
        'YY': 'ryy',
        'ZZ': 'rzz',
    }

    # Calculate the magnetization
    magnetization_operator = sum(unitaries.Z([i]).as_matrix(n) for i in range(n))

    for i, circuit_gates in enumerate(gate_data, start=1):
        print(f"Parsing Circuit {i} to get magnetization")
        mag = toQuimbMag(circuit_gates, n, gate_name_mapping, magnetization_operator)
        magnetizations.append(mag)

    return magnetizations

def parse(relative_path):
    """
    Parse the data from the specified folder path and return a dictionary containing the parsed results.
    
    Args:
        relative_path: The relative path to the folder containing JSON files.

    Returns:
        parsed_results: A dictionary where:
            - The key is the parameter tuple (N, n_snapshot, circuits, delta_name, T, numQs).
            - The value is another dictionary containing:
                - "circuits": List of quimb circuit objects.
                - "signs": List of sign data corresponding to the circuits.
                - "overhead": The overhead value extracted from the sign data.
                - "params": The parameter tuple for the dataset.
                - "gates_arr": The raw gates array data.
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
        N, n_snapshot, circuits, delta_name, T, numQs = params

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
        magnetizations = getMags(gate_data, int(numQs))

        # Store the results in the parsed_results dictionary
        parsed_results[params] = {
            "magnetizations": magnetizations,
            "overhead": overhead,
            "params": params,
            "sign_list": sign_data,
        }

    return parsed_results