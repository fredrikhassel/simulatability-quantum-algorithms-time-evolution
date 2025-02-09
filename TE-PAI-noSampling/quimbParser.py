import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
import quimb.tensor as qtn
from pyquest import unitaries

def loadData(folder_path):
    """ Load CSV data from a folder and organize it by parameter sets."""

    # Regular expression pattern to extract the shared parameters from filenames
    pattern = r"(sign_list|gates_arr)-N-(\d+)-n-(\d+)-c-(\d+)-Δ-([^\-]+)-T-(\d+)-q-(\d+)\.csv"

    # Dictionary to store matched data by parameter set
    data_storage = defaultdict(lambda: {"sign_list": None, "gates_arr": None})

    # Load and organize CSV data
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Check if the file matches the naming pattern
            match = re.match(pattern, filename)
            if match:
                file_type, N, n_snapshot, circuits, delta_name, T, numQs = match.groups()
                key = (N, n_snapshot, circuits, delta_name, T, numQs)

                # Read CSV file into a DataFrame
                try:
                    df = pd.read_csv(file_path)
                    data_storage[key][file_type] = df
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    else:
        print(f"Folder not found: {folder_path}")

    if match:
        return data_storage, int(numQs), N, T

def toArray(data_storage):
    """ Parse the data from the data storage dictionary. """

    # Initialize parsed data
    gate_data = []
    sign_data = []

    # Iterate over the data storage
    for params, data in data_storage.items():
        N, n_snapshot, circuits, delta_name, T, numQs = params
        sign_list = data["sign_list"]
        gates_arr = data["gates_arr"]

        # Check if both sign_list and gates_arr are present
        if sign_list is not None and gates_arr is not None:
            # Extract gates from gates_arr DataFrame
            for _, row in gates_arr.iterrows():
                gate_data_str = str(row[1])  # Ensure the data is treated as a string
                gate_matches = re.findall(r"\(('.*?', .*?, \[.*?\])\)", gate_data_str)
                circuit_gates = []

                for gate_match in gate_matches:
                    try:
                        # Parse the string into a structured element
                        gate_name, gate_value, gate_indices = eval(gate_match)
                        circuit_gates.append([gate_name, float(gate_value), gate_indices])
                    except Exception as e:
                        print(f"Error parsing gate data: {e}")
                gate_data.append(circuit_gates)

            
            for _, row in sign_list.iterrows():
                sign_data_str = str(row[1])
                signs = [int(x) for x in re.findall(r"-?\d+", sign_data_str)]
                sign_data.append(signs)


    return gate_data, sign_data

def toQuimbTensor(circuit_gates, circuit_signs, n, N, T):
    """Parse the 1 circuit """
    sign = np.prod(circuit_signs)        

    # Initialize the circuit with the specified number of qubits
    circuit = qtn.Circuit(n)
    gate_name_mapping = {
        'XX': 'rxx',
        'YY': 'ryy',
        'ZZ': 'rzz',
    }

    mag = []
    magnetizationOperator = sum(unitaries.Z([i]).as_matrix(n) for i in range(n))

    # Calculate evenly spaced indices for execution
    execute_indices = np.linspace(0, len(circuit_gates) - 1, int(N), dtype=int)

    # Iterate over the gate data and apply each gate to the circuit
    for i,gate in enumerate(circuit_gates):
        gate_name = gate[0]
        angle = gate[1]
        qubit_indices = gate[2]

        # Translate the gate name if it's in the mapping
        quimb_gate_name = gate_name_mapping.get(gate_name, gate_name)

        # Apply the gate to the circuit
        if len(qubit_indices) == 1:
            # Single-qubit gate
            circuit.apply_gate(gate_id=quimb_gate_name, qubits=[qubit_indices[0]], params=[angle])
        elif len(qubit_indices) == 2:
            # Two-qubit gate
            circuit.apply_gate(gate_id=quimb_gate_name, qubits=qubit_indices, params=[angle])
        else:
            raise ValueError(f"Unsupported number of qubits for gate {quimb_gate_name}: {len(qubit_indices)}")
        
        if gate_name=="XX" and qubit_indices==[0,1]: # this should execute at most N times for each circuit
            # Calculate the magnetization
            state_vector = circuit.psi.to_dense()
            mag.append(np.vdot(state_vector, magnetizationOperator @ state_vector).real)

    return circuit, sign, mag

def getCircuits(gate_data, sign_data, n, N, T):
    """Parse the arrays into quimb circuits and their parity signs."""

    circuits = []
    signs = []
    mags = []

    i = 0
    for circuit_gates, circuit_signs in zip(gate_data, sign_data):
        i+=1
        print("Circuit: "+str(i))
        circuit, sign, mag = toQuimbTensor(circuit_gates, circuit_signs, n, N, T)
        circuits.append(circuit)
        signs.append(sign)
        mags.append(mag)

    return circuits, signs, mags

def parse(relative_path):
    """Parse the data from the specified folder path."""
    folder_path = os.path.abspath(relative_path)
    data_storage, n, N, T = loadData(folder_path)
    gate_data, sign_data = toArray(data_storage)
    circuits, signs, mags = getCircuits(gate_data, sign_data, n, N, T)
    return circuits, signs, mags, n, N, T