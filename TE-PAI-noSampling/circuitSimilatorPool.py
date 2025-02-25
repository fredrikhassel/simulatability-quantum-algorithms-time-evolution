import os
import random
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import quimb.tensor as qtn
import quimb as qu
from pyquest import unitaries
import SIMULATOR
from qiskit.quantum_info import SparsePauliOp

def loadData(folder_path):
    """Load JSON data from a folder and organize it by parameter sets."""
    
    # Regular expression pattern to extract the shared parameters from filenames
    pattern = r"(sign_list|gates_arr)-N-(\d+)-n-(\d+)-(c|p)-(\d+)-Δ-([\w\-]+)-T-([\d\.]+)-q-(\d+)\.json"
    
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
            file_type, N, n_snapshot, PorC,circuits, delta_name, T, numQs = match.groups()
            
            # Convert parameters to appropriate types
            N = int(N)
            n_snapshot = int(n_snapshot)
            circuits = int(circuits)
            T = np.round(float(T), 4) if "." in T else int(T)  # Convert T to float if it contains a dot
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
    circuits_arr = []
    for timestep_idx, timestep in gate_dict.items():
        for circuit_idx, circuit in timestep.items():
            gates_arr = []
            for gate_idx, gate in circuit.items():
                gates_arr.append((gate["gate_name"], gate["angle"], gate["qubits"]))
            circuits_arr.append(gates_arr)

    for i,circ in enumerate(circuits_arr):
        print(f"Circuit {i+1} has {len(circ)} gates in it.")

    return circuits_arr

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

def toQuimbMag(circuit_gates, gate_name_mapping, circuit, q, index):
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

    #state_vector = circuit.psi.to_dense()
    #mag =  circuit.local_expectation(qu.pauli('Z'), (0))
    mag = measure(circuit, q)

    del circuit
    return mag

def measure(circuit,q):
    magnetization_operator = unitaries.Z([0]).as_matrix(q) #sum(unitaries.Z([i]).as_matrix(q) for i in range(q))#sum(unitaries.Z([i]).as_matrix(q) for i in range(q)) #unitaries.X([0]).as_matrix(q) #sum(unitaries.Z([i]).as_matrix(q) for i in range(q))
    #magnetization_operator = qu.ikron([qu.pauli("X")], dims=[q], inds=[(0,0)])  # Apply X at the zeroth qubit

    state_vector = circuit.psi.to_dense()
    #expect = state_vector.H @ magnetization_operator @ state_vector

    expect = qu.expec(state_vector, magnetization_operator).real
    #expect = np.abs(circuit.local_expectation(qu.pauli('X'), (0)))

    return (expect+1) / 2

def getMags(circuits_gates, circuits_signs, q, indices):
    # Calculate the magnetization

    gate_name_mapping = {
        'XX': 'rxx',
        'YY': 'ryy',
        'ZZ': 'rzz',
        'Z': 'rz'
    }
    
    magnetizations = []
    for i, (gates, signs, indices) in enumerate(zip(circuits_gates, circuits_signs, indices)):
        print(f"Calculating magnetizations of run {i}")
        print(f"This run will use these circuits: {indices}")

        current_circuit = qtn.Circuit(q)
        magnetization = [1]

        for j, (gate, sign, index) in enumerate(zip(gates, signs, indices)):
            if type(sign) != float:
                sign = sign[0]
            magnetization.append(toQuimbMag(gate, gate_name_mapping, current_circuit, q, index) * sign)
        magnetizations.append(magnetization)

        del current_circuit

    return magnetizations

def parse(folder_path, dT):
    # Extracting data
    data_storage = loadData(folder_path)
    parameters, overheads, sign_lists, gate_arrs = extract_data(data_storage)
    Ts = [p[4] for p in parameters]

    # Preparing timestep data
    N, n_snapshot, circuits_count, delta_name, _, q = parameters[0] # should all be the same
    
    circuits_gates, circuits_signs  = formatData(parameters, sign_lists, gate_arrs)

    magnetizations = getMags(circuits_gates, circuits_signs, q)
    averages = np.mean(magnetizations, axis=0)
    stds = np.std(magnetizations, axis=0)

    saveData(N, n_snapshot, circuits_count, delta_name, Ts, q, dT, averages, stds)

    return

def saveData(N, n_snapshot, circuits_count, delta_name, Ts, q, dT, averages, stds):
    # Define the output directory
    output_dir = os.path.join('TE-PAI-noSampling', 'data', 'plotting')
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    # Save plotting data to a CSV file
    filename = f"N-{N}-n-{n_snapshot}-p-{circuits_count}-Δ-{delta_name}-T-{np.max(Ts)}-q-{q}-dT-{dT}.csv"
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

        # Plotting
    plt.errorbar(Ts, averages, yerr=stds, fmt='o', ecolor='red', capsize=4, label="Data with Error Bars")
    plt.xlabel('Time (t)')
    plt.ylabel('Magnetization')
    plt.title('Plot of Magnetization vs Time')
    plt.legend()
    plt.show()

def split_arrays_randomly(arr1, arr2, num_parts):
    if len(arr1) != len(arr2):
        print(f"circuit_pool: {len(arr1)}, sign_pool: {len(arr2)}")
        raise ValueError("Both arrays must have the same length")
    if len(arr1) % num_parts != 0:
        raise ValueError("Array length must be evenly divisible by the number of parts")
    
    print("Splitting circuits and gates of length: ", len(arr1))

    indices = list(range(len(arr1)))
    random.shuffle(indices)  # Shuffle the indices randomly

    print("Arrays split into the following indices: ", indices)
    
    arr1_shuffled = [arr1[i] for i in indices]
    arr2_shuffled = [arr2[i] for i in indices]
    
    part_size = len(arr1) // num_parts
    
    arr1_parts = [arr1_shuffled[i * part_size:(i + 1) * part_size] for i in range(num_parts)]
    arr2_parts = [arr2_shuffled[i * part_size:(i + 1) * part_size] for i in range(num_parts)]
    index_parts = [indices[i * part_size:(i + 1) * part_size] for i in range(num_parts)]
    
    return arr1_parts, arr2_parts, index_parts

def parsePool(folder_path, dT):
    # Extracting data
    data_storage = loadData(folder_path)
    parameters, overheads, sign_lists, gate_arrs = extract_data(data_storage)

    N, n_snapshot, circuits_count, delta_name, T, q = parameters[0]
    circuit_pool = gate_arrs[0]
    sign_pool = sign_lists[0].reshape(-1)
    n_timesteps = round(T/dT)
    n_circuits = int(len(sign_pool) / n_timesteps)
    Ts = np.arange(0, T+dT, dT)

    print("Lets just walk through this:")
    print(f"We started out with a total of {len(sign_pool)} circuits")
    print(f"We will be running: {T} / {dT} = {n_timesteps} timesteps")
    print(f"So our circuit runs will be {n_timesteps} long so as to take total time {T}")
    print(f"that means that each circuit run will involve {len(sign_pool)} / {n_timesteps} = {n_circuits} paralell runs")
    quit
    circuits, signs, indices = split_arrays_randomly(circuit_pool, sign_pool, n_circuits)
    print(indices)
    print("So now signs have shapes: ",np.shape(signs))
    magnetizations = getMags(circuits, signs, q, indices)

    averages = np.mean(magnetizations, axis=0)
    stds = np.std(magnetizations, axis=0)

    print(averages)
    print(stds)

    saveData(N, n_snapshot, circuits_count, delta_name, Ts, q, dT, averages, stds)

parsePool('TE-PAI-noSampling/data/circuits/N-1000-n-1-p-2000-Δ-pi_over_1024-q-4-dT-0.005-T-0.2/', 0.005)