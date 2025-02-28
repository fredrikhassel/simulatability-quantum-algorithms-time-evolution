import csv
import os
import json
import re
from collections import defaultdict
from matplotlib import pyplot as plt
import quimb.tensor as qtn
import numpy as np
from pyquest import unitaries
import quimb as qu
import pandas as pd
from HAMILTONIAN import Hamiltonian
from TROTTER import Trotter

def JSONtoDict(folder_name):
    files = os.listdir(folder_name)
    
    # Regular expressions for matching filenames
    gates_pattern = re.compile(r"gates_arr-N-(\d+)-n-(\d+)-c-(\d+)-Δ-([\w_]+)-T-([\d.]+)-q-(\d+)\.json")
    sign_pattern = re.compile(r"sign_list-N-(\d+)-n-(\d+)-c-(\d+)-Δ-([\w_]+)-T-([\d.]+)-q-(\d+)\.json")
    
    gates_dict = {}
    sign_dict = {}
    
    for file in files:
        gates_match = gates_pattern.match(file)
        sign_match = sign_pattern.match(file)
        
        if gates_match:
            key = gates_match.groups()
            gates_dict[key] = os.path.join(folder_name, file)
        elif sign_match:
            key = sign_match.groups()
            sign_dict[key] = os.path.join(folder_name, file)
    
    paired_data = {}
    
    for key in gates_dict:
        if key in sign_dict:
            with open(gates_dict[key], 'r') as f:
                gates_content = json.load(f)
            with open(sign_dict[key], 'r') as f:
                sign_content = json.load(f)
            
            paired_data[key] = {
                "gates": gates_content,
                "signs": sign_content
            }
        else:
            print(f"Warning: No matching sign_list file for {gates_dict[key]}")
    
    for key in sign_dict:
        if key not in gates_dict:
            print(f"Warning: No matching gates_arr file for {sign_dict[key]}")
    
    return paired_data

def DictToArr(dict):
    pool = (len(dict) == 1)
    arr = [(None,None) for _ in dict]
    Ts = []
    
    # Per timestep
    for i,(key, value) in enumerate(dict.items()):
        # Extracting parameters
        N,n,c,Δ,T,q = eval(str(key))
        params = [N,n,c,Δ,T,q]
        T = round(float(T), 8)
        Ts.append(T)

        sign_arr = []
        gate_tuples_arr = []
        del value["signs"]["overhead"]
        # Per circuit
        for j,(gates_content, sign_content) in enumerate(zip(value["gates"].items(), value["signs"].items())):
            sign = sign_content[1]
            gates = gates_content[1]["1"] # value, 1st snapshot
            gate_tuples = []
            for _,gate in gates.items():
                gate_name = gate["gate_name"]
                angle = gate["angle"]
                qubits = gate["qubits"]
                gate_tuples.append((gate_name, angle, qubits))
            sign_arr.append(sign)
            gate_tuples_arr.append(gate_tuples)
        arr[i] = (gate_tuples_arr, sign_arr)

    return arr, Ts,  params

def applyGates(circuit, gates):
    gate_name_mapping = {
    'XX': 'rxx',
    'YY': 'ryy',
    'ZZ': 'rzz',
    'Z': 'rz'
    }
    for gate in gates:
        # Getting gate details
        gate_name = gate[0]
        angle = gate[1]
        qubit_indices = gate[2]
        quimb_gate_name = gate_name_mapping[gate_name]

        # Apply the gate to the circuit       
        if len(qubit_indices) == 1:
            circuit.apply_gate(gate_id=quimb_gate_name, qubits=[qubit_indices[0]], params=[angle])
        elif len(qubit_indices) == 2:
            circuit.apply_gate(gate_id=quimb_gate_name, qubits=qubit_indices, params=[angle])
        else:
            raise ValueError(f"Unsupported number of qubits for gate {quimb_gate_name}: {len(qubit_indices)}")

def measure(circuit,q):
    state_vector = circuit.psi.to_dense()
    magnetization_operator = qu.ikron(qu.pauli('x'), [2]*q, 0)
    expect = qu.expec(magnetization_operator, state_vector)
    return (expect+1) / 2

def getSucessive(data_arrs,q):
    averages = []
    stds = []
    # Per timestep
    for (circuits_gates, circuiit_signs) in data_arrs:
        mags = []
        # Per circuit
        for gates, sign in zip(circuits_gates, circuiit_signs):
            quimb = qtn.Circuit(q)
            for i in range(q):
                quimb.apply_gate('H', qubits=[i])
            applyGates(quimb,gates)
            mag = measure(quimb,q)
            mags.append(mag*sign)
            del quimb
        averages.append(np.mean(mags))
        stds.append(np.std(mags))

    return averages,stds

def parse(folder):
    data_dict = JSONtoDict(folder)
    data_arrs,Ts,params = DictToArr(data_dict)
    N,n,c,Δ,T,q = params
    averages,stds = getSucessive(data_arrs,int(q))
    #plot_data(Ts, averages, stds)
    saveData(N,n,c,Δ,Ts,q,0.01,averages,stds)
    lie = trotter(100,10,float(T),int(q),compare=False,save=True)
    plot_data_from_folder("TE-PAI-noSampling/data/plotting")

def plot_data_from_folder(folderpath):
    quimb_pattern = re.compile(r'N-(\d+)-n-(\d+)-[cp]-(\d+)-Δ-(\w+)-T-([\d\.]+)-q-(\d+)-dT-([\d\.]+)\.csv')
    lie_pattern = re.compile(r'lie-N-(\d+)-T-((?:\d+\.\d+)|(?:\d+))-q-(\d+)\.csv')
    
    quimb_data = []
    lie_data = []
    
    for filename in os.listdir(folderpath):
        filepath = os.path.join(folderpath, filename)
        
        quimb_match = quimb_pattern.match(filename)
        lie_match = lie_pattern.match(filename)
        
        if quimb_match:
            df = pd.read_csv(filepath)
            if df.shape[1] >= 3:
                label = f"N-{quimb_match.group(1)} T-{quimb_match.group(5)} q-{quimb_match.group(6)} dT-{quimb_match.group(7)}"
                quimb_data.append((df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], label))
        
        elif lie_match:
            df = pd.read_csv(filepath)
            if df.shape[1] >= 2:
                label = f"N-{lie_match.group(1)} T-{lie_match.group(2)} q-{lie_match.group(3)}"
                lie_data.append((df.iloc[:, 0], df.iloc[:, 1], label))
    
    plt.figure(figsize=(10, 6))
    
    for x, y, error, label in quimb_data:
        plt.errorbar(x, y, yerr=error, fmt='o', label=f'Quimb Data ({label})', alpha=0.6)
    
    for x, y, label in lie_data:
        plt.plot(x, y, label=f'Lie Data ({label})')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Quimb and Lie Data Comparison')
    plt.show()

def plot_data(Ts, averages, stds):
    plt.figure(figsize=(8, 6))
    plt.errorbar(Ts, averages, yerr=stds, fmt='o', capsize=5, capthick=2, elinewidth=1, label="Data with Error Bars")
    plt.xlabel("T values")
    plt.ylabel("Averages")
    plt.title("Plot of Averages vs T with Error Bars")
    plt.legend()
    plt.grid(True)
    plt.show()

def saveData(N, n_snapshot, circuits_count, delta_name, Ts, q, dT, averages, stds):
    # Define the output directory
    output_dir = os.path.join('TE-PAI-noSampling', 'data', 'plotting')
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    # Save plotting data to a CSV file
    filename = f"{N}-n-{n_snapshot}-p-{circuits_count}-Δ-{delta_name}-T-{np.max(Ts)}-q-{q}-dT-{dT}.csv"
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

def trotterSimulation(hamil, N, n_snapshot, c, Δ_name, T, numQs):
    trotter = Trotter(hamil, N, n_snapshot, c, Δ_name, T, numQs)
    res, gates_arr = trotter.run()
    return res, gates_arr

def checkEqual(gates, gatesSim):
    # Assert that the number of snapshots is the same
    assert len(gates) == len(gatesSim), f"Number of snapshots are different: {len(gates)} vs {len(gatesSim)}"

    # Assert that the number of gates in each snapshot is the same
    for i in range(len(gates)):
        assert len(gates[i]) == len(gatesSim[i]), f"Number of gates in snapshot {i} are different: {len(gates[i])} vs {len(gatesSim[i])}"

    # Assert that the gates themselves are the same
    for i in range(len(gates)):
        for j in range(len(gates[i])):
            assert gates[i][j] == gatesSim[i][j], f"Gates at snapshot {i}, gate {j} are different: {gates[i][j]} vs {gatesSim[i][j]}"

    print("These arrays have the same structure and gates.")
    return True

def trotter(N, n_snapshot, T, q, compare, save=False):
    times = np.linspace(0, float(T), int(N))
    print(n_snapshot)
    Ts = np.linspace(0, float(T), n_snapshot+1)
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    terms = [hamil.get_term(t) for t in times]
    gates = []
    n = int(N / n_snapshot)
    
    for i in range(N):
        if i % n == 0:
            gates.append([])
        gates[-1] += [
            (pauli, 2 * coef * T / N, ind)
            for (pauli, ind, coef) in terms[i]
        ]
    
    circuit = qtn.Circuit(q)
    for i in range(q):
        circuit.apply_gate('H', qubits=[i])
    
    res = [measure(circuit, q)]
    for i, gs in enumerate(gates):
        applyGates(circuit, gs)
        res.append(measure(circuit, q))
    
    if save:
        save_path = os.path.join("TE-PAI-noSampling", "data", "plotting")
        os.makedirs(save_path, exist_ok=True)
        file_name = f"lie-N-{N}-T-{T}-q-{q}.csv"
        file_path = os.path.join(save_path, file_name)
        
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y"])
            writer.writerows(zip(Ts, res))
        print(f"Data saved to {file_path}")
    
    if not compare:
        return res
    
    if compare:
        checkEqual(gates, gatesSim)
        resSim, gatesSim = trotterSimulation(hamil, N, n_snapshot, 1, 1, T, q)
        print(resSim)
        print(res)
        plt.scatter(range(len(resSim)), resSim, color='blue', label='Qiskit')
        plt.scatter(range(len(res)), res, color='red', label='Quimb')
        plt.legend()
        plt.show()

parse("TE-PAI-noSampling/data/circuits/N-1000-n-1-c-10-Δ-pi_over_1024-q-4-dT-0.01-T-0.1")
#plot_data_from_folder("TE-PAI-noSampling/data/plotting")
