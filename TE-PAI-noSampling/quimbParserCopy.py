import json
import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
import quimb.tensor as qtn
from pyquest import unitaries
import quimb as qu

def loadData(folder_path):
    """ Load JSON data from a folder and organize it by parameter sets."""

    # Regular expression pattern to extract the shared parameters from filenames
    pattern = r"(sign_list|gates_arr)-N-(\d+)-n-(\d+)-c-(\d+)-Δ-([\w\-]+)-T-([\d\.]+)-q-(\d+)\.json"

    # Dictionary to store matched data by parameter set
    data_storage = defaultdict(lambda: {"sign_list": None, "gates_arr": None})

    # Load and organize CSV data
    if os.path.exists(folder_path):
        print("path exists")
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            print("found file: "+str(filename))
            # Check if the file matches the naming pattern
            match = re.match(pattern, filename)
            if match:
                print("match")
                file_type, N, n_snapshot, circuits, delta_name, T, numQs = match.groups()
                key = (N, n_snapshot, circuits, delta_name, T, numQs)

                df = pd.read_json(file_path, orient="index")
                data_storage[key][file_type] = df


    else:
        print(f"Folder not found: {folder_path}")

    if match:
        return data_storage

def toArray(data_storage):
    """ Parse the data from the data storage dictionary. """
    gate_data = []
    sign_data = []
    
    for params, data in data_storage.items():
        sign_list = data["sign_list"]
        gates_arr = data["gates_arr"]
        
        if sign_list is not None and gates_arr is not None:
            # Convert sign_list DataFrame to dictionary
            sign_list_dict = sign_list.to_dict(orient="index") if isinstance(sign_list, pd.DataFrame) else sign_list

            print("Sign List:")
            sign_data.append([sign_list_dict["overhead"]])  # Now "overhead" should be accessible
            
            sign_data.extend([[v] for k, v in sign_list_dict.items() if k != "overhead"])
            
            circuit_data = []
            for circuit_idx, circuit in gates_arr.items():
                snapshot_data = []
                for snap_idx, snapshot in circuit.items():
                    gate_list = []
                    for gate_idx, gate in snapshot.items():
                        gate_list.append((gate["gate_name"], gate["angle"], gate["qubits"]))
                    snapshot_data.append(gate_list)
                circuit_data.append(snapshot_data)
            gate_data.append(circuit_data)
    
    overhead = sign_data[0]
    return gate_data, sign_data[1:], overhead

def toQuimbTensor(circuit_gates, circuit_signs, n, N, T):
    """Parse the 1 circuit with measurements at the end of each snapshot."""
    #sign = np.prod(circuit_signs)

    names = ""
    
    # Initialize the circuit with the specified number of qubits
    circuit = qtn.Circuit(n)
    gate_name_mapping = {
        'XX': 'rxx',
        'YY': 'ryy',
        'ZZ': 'rzz',
    }
    
    mag = []
    magnetizationOperator = sum(unitaries.Z([i]).as_matrix(n) for i in range(n))
    
    # Iterate over snapshots
    for snapshot in circuit_gates:
        # Measure at the end of each snapshot
        state_vector = circuit.psi.to_dense()
        mag.append(qu.expec(magnetizationOperator, state_vector).real)

        for gate in snapshot:
            gate_name = gate[0]
            angle = gate[1]
            qubit_indices = gate[2]
            
            # Translate the gate name if it's in the mapping
            quimb_gate_name = gate_name_mapping.get(gate_name, gate_name)

            names += " " + quimb_gate_name + " " + str(qubit_indices) + " " + str(angle)
            
            # Apply the gate to the circuit
        
            if len(qubit_indices) == 1:
                circuit.apply_gate(gate_id=quimb_gate_name, qubits=[qubit_indices[0]], params=[angle])
            elif len(qubit_indices) == 2:
                circuit.apply_gate(gate_id=quimb_gate_name, qubits=qubit_indices, params=[angle])
            else:
                raise ValueError(f"Unsupported number of qubits for gate {quimb_gate_name}: {len(qubit_indices)}")
    #print(names)
    return circuit, circuit_signs, mag

def getCircuits(gate_data, sign_data, n, N, T):
    """Parse the arrays into quimb circuits and their parity signs."""

    circuits = []
    signs = []
    mags = []

    i = 0
    for circuit_gates, circuit_signs in zip(gate_data[0], sign_data):
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
    data_storage = loadData(folder_path)
    params = list(data_storage.keys())
    N, n_snapshot, circuits, delta_name, T, numQs = params[0]
    gate_data, sign_data, overhead = toArray(data_storage)
    circuits, signs, mags = getCircuits(gate_data, sign_data, int(numQs), int(N), T)
    return circuits, signs, mags, overhead, params[0]