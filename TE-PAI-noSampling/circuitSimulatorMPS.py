import csv
import os
import json
from pathlib import Path
import re
import shutil
from matplotlib import pyplot as plt
import quimb.tensor as qtn
import numpy as np
import quimb as qu
import pandas as pd
from HAMILTONIAN import Hamiltonian
from TROTTER import Trotter
import h5py
import time
import opt_einsum as oe
import cotengra as ctg
from qiskit import QuantumCircuit, transpile
from scipy.stats import linregress
from scipy.optimize import curve_fit

def JSONtoDict(folder_name):
    try:
        files = os.listdir(folder_name)
    except FileNotFoundError:
        if folder_name.endswith('.0'):
            fallback_name = folder_name[:-2]  # removes the '.0'
            try:
                files = os.listdir(fallback_name)
            except FileNotFoundError:
                raise FileNotFoundError(f"Neither '{folder_name}' nor '{fallback_name}' could be found.")
        else:
            raise FileNotFoundError(f"'{folder_name}' could not be found.")    
    
    # Regular expressions for matching filenames
    gates_pattern = re.compile(r"gates_arr-N-(\d+)-n-(\d+)-[cp]-(\d+)-Δ-([\w_]+)-T-([\d.]+)-q-(\d+)\.json")
    sign_pattern = re.compile(r"sign_list-N-(\d+)-n-(\d+)-[cp]-(\d+)-Δ-([\w_]+)-T-([\d.]+)-q-(\d+)\.json")
    
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

def HDF5toDict(folder_name):
    sign_file_path = os.path.join(folder_name, "sign_list.json")
    gates_file_path = os.path.join(folder_name, "gates.h5")

    extracted_data = {}

    with open(sign_file_path, 'r') as f:
        metadata = json.load(f)

    # Extract stored parameters
    N, n, c, Δ, T, q = metadata["N"], metadata["n"], metadata["c"], metadata["Δ"], metadata["T"], metadata["q"]
    Δ =round(np.log2((Δ**-1)*np.pi))
    Δ = 'pi_over_'+str(2**Δ)

    sign_dict = metadata["signs"]

    # Load gates from HDF5
    with h5py.File(gates_file_path, 'r') as hdf_file:
        circuits_group = hdf_file["circuits"]
        gates_data = {
            int(circuit_idx): {
                int(snap_idx): json.loads(circuit_group[snap_idx][()].decode('utf-8'))
                for snap_idx in circuit_group
            }
            for circuit_idx, circuit_group in circuits_group.items()
        }

    # Store using parameters as key
    extracted_data[(N, n, c, Δ, T, q)] = {
        "signs": sign_dict,
        "gates": gates_data
    }

    return extracted_data

def DictToArr(dict, isJSON):

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
            gate_tuples = []
            if(isJSON):
                gates = gates_content[1]["1"] # value, 1st snapshot
                for _,gate in gates.items():
                    gate_name = gate["gate_name"]
                    angle = gate["angle"]
                    qubits = gate["qubits"]
                    gate_tuples.append((gate_name, angle, qubits))
            else:
                gates = gates_content[1]
                for gate in gates[1]:
                    gate_name = gate["gate_name"]
                    angle = gate["angle"]
                    qubits = gate["qubits"]
                    gate_tuples.append((gate_name, angle, qubits))
            
            sign_arr.append(sign)
            gate_tuples_arr.append(gate_tuples)
        arr[i] = (gate_tuples_arr, sign_arr)

    return arr, Ts,  params, pool

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

def measure(circuit,q, optimize):
    val = np.abs(circuit.local_expectation(qu.pauli('X'), (0)))
    return (val+1)/2

def getSucessive(data_arrs,q, flip=False):
    averages = []
    costs = []
    stds = []
    # Per timestep
    print(f"Calculating sucessive for {len(data_arrs)} timesteps")
    for (circuits_gates, circuit_signs) in data_arrs:
        print(f"Timestep: {len(averages)+1}")
        mags = []
        cs = 0
        # Per circuit
        for gates, sign in zip(circuits_gates, circuit_signs):
            quimb = getCircuit(q, flip=flip)
            applyGates(quimb,gates)
            mag = measure(quimb,q, None)
            mags.append(mag*sign)            
            cs += getComplexity(quimb)
            del quimb
        print(len(circuit_signs))
        costs.append(cs / len(circuit_signs))
        averages.append(np.mean(mags))
        stds.append(np.std(mags))

    return averages,stds,costs

def norm_fn(psi):
    # we could always define this within the loss function, but separating it
    # out can be clearer - it's also called before returning the optimized TN
    nfact = (psi.H @ psi)**0.5
    return psi.multiply(1 / nfact, spread_over='all')

def loss_fn(psi, ham):
    b, h, k = qtn.tensor_network_align(psi.H, ham, psi)
    energy_tn = b | h | k
    return energy_tn ^ ...

def getPool(data_arrs,params,dT, draw, optimize=None, flip=False, fixedCircuit=None):
    N,n,c,Δ,T,q = params
    T = float(T)
    q = int(q)
    circuit_pool, sign_pool = data_arrs[0]
    n_timesteps = round(T/dT)
    results = [[] for _ in range(n_timesteps)]
    costs = [[] for _ in range(n_timesteps)]
    n_circuits = int(len(sign_pool) / n_timesteps)
    Ts = np.arange(0, T+dT, dT)
    indices = generate_random_indices(len(circuit_pool), n_circuits, n_timesteps)

    print(f"Running pool for {len(indices)} runs")
    c = 1
    for run_indices in indices:
        print(f"Run: {c}")
        c+=1
        circuits = [circuit_pool[idx] for idx in run_indices]
        signs = [sign_pool[idx] for idx in run_indices]
        if fixedCircuit == None:
            quimb = getCircuit(q, flip=flip)
        else:
            quimb = fixedCircuit.copy()
        for i,(circuit,sign) in enumerate(zip(circuits,signs)):
            applyGates(quimb,circuit)
            costs[i].append(getComplexity(quimb))#quimb.psi.contraction_cost())
            #quimb.amplitude_rehearse(simplify_sequence='RL')['tn'].draw(color=['PSI0', 'RZ', 'RZZ', 'RXX', 'RYY', 'H'], layout="kamada_kawai")
            results[i].append(measure(quimb,q, optimize)*sign)

    costs = np.mean(costs, axis=1)
    averages = np.mean(results, axis=1)
    stds = np.std(results, axis=1)

    if fixedCircuit == None:
        averages = np.insert(averages, 0, 0)
    else:
        averages = np.insert(averages, 0, measure(fixedCircuit, q, optimize))
    stds = np.insert(stds, 0, 0)

    if draw:
        return averages,stds, quimb, costs
    else:
        return averages,stds, None, costs

def generate_random_indices(pool_size, output_length, entry_length):
    rng = np.random.default_rng(0)  # Create RNG instance with optional seed
    return [list(rng.integers(0, pool_size, size=entry_length)) for _ in range(output_length)]

def extract_dT_value(string):
    match = re.search(r'dT-([\d\.]+)', string)
    return float(match.group(1)) if match else None

def plot_data_from_folder(folderpath):
    quimb_pattern = re.compile(r'N-(\d+)-n-(\d+)-([cp])-(\d+)-Δ-(\w+)-T-([\d\.]+)-q-(\d+)-dT-([\d\.]+)\.csv')
    lie_pattern = re.compile(r'lie-N-(\d+)-T-((?:\d+\.\d+)|(?:\d+))-q-(\d+)\.csv')
    
    quimb_data = []
    lie_data = []
    
    for filename in os.listdir(folderpath):
        filepath = os.path.join(folderpath, filename)
        
        quimb_match = quimb_pattern.match(filename)
        lie_match = lie_pattern.match(filename)
        
        if lie_match:
            df = pd.read_csv(filepath)
            if df.shape[1] >= 2:
                label = f"N-{lie_match.group(1)} T-{lie_match.group(2)} q-{lie_match.group(3)}"
                lie_data.append((df.iloc[:, 0], df.iloc[:, 1], label))


        elif quimb_match:
            df = pd.read_csv(filepath)
            if df.shape[1] >= 3:

                char = quimb_match.group(3)
                lab = "ERROR"
                if char == "c":
                    lab = "paralell"
                if char == "p":
                    lab = "pool"

                if(len(lie_data) == 0):
                    label = f"N-{quimb_match.group(1)} {lab}-{quimb_match.group(4)} Δ-{quimb_match.group(5)} T-{quimb_match.group(6)} q-{quimb_match.group(7)} dT-{quimb_match.group(8)}"
                else:
                    label = f"N-{quimb_match.group(1)} {lab}-{quimb_match.group(4)} Δ-{quimb_match.group(5)} dT-{quimb_match.group(8)}"
                quimb_data.append((df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], label))
        
    plt.figure(figsize=(10, 6))
    
    for x, y, error, label in quimb_data:
        plt.errorbar(x, y, yerr=error, fmt='-o', label=f'Random ({label})', alpha=0.6)
    
    for x, y, label in lie_data:
        plt.plot(x, y, color="tab:red", label=f'Lie Data ({label})')
    
    plt.xlabel('Time')
    plt.ylabel('X expectation value')
    plt.legend()
    plt.title('Random tensor-networks and Lie Data Comparison')
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

def saveData(N, n_snapshot, circuits_count, delta_name, Ts, q, dT, averages, stds, char):
    # Define the output directory
    output_dir = os.path.join('TE-PAI-noSampling', 'data', 'plotting')
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    # Save plotting data to a CSV file
    filename = f"N-{N}-n-{n_snapshot}-{char}-{circuits_count}-Δ-{delta_name}-T-{np.max(Ts)}-q-{q}-dT-{dT}.csv"
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

def trotter(N, n_snapshot, T, q, compare, startTime=0, save=False, draw=False, flip=False, fixedCircuit=None):
    print(f"Running Trotter for N={N}, n_snapshot={n_snapshot}, T={T}, q={q}")
    circuit = None
    
    times = np.linspace(startTime, startTime+float(T), int(N))
    Ts = np.linspace(startTime, startTime+float(T), int(n_snapshot)+1)
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
    
    if fixedCircuit == None:
        res = [1]
    else:
        res = [measure(fixedCircuit, q, False)]

    complexity = [0]
    for i, gs in enumerate(gates):
        print(f"Snapshot {i+1} / {len(gates)}")

        # Fresh circuit each time
        if fixedCircuit == None:
            circuit = getCircuit(q, flip=flip)
        else:
            circuit = fixedCircuit.copy()

        # Apply gates up to current time
        for k in range(i + 1):
            applyGates(circuit, gates[k])

        if draw and i == 0:
            circuit.psi.draw(color=['PSI0', 'RZ', 'RZZ', 'RXX', 'RYY', 'H'], layout="kamada_kawai")
            qiskit = quimb_to_qiskit(circuit)
            qiskit.draw("mpl", scale=0.4)
            plt.show()

        # Measure immediately after applying
        result = measure(circuit, q, False)
        complexity.append(getComplexity(circuit))
        res.append(result)

    if save:
        save_path = os.path.join("TE-PAI-noSampling", "data", "plotting")
        os.makedirs(save_path, exist_ok=True)
        if fixedCircuit == None:
            file_name = f"lie-N-{N}-T-{T}-q-{q}.csv"
        else:
            file_name = f"lie-N-{N}-T-{T+1000}-q-{q}-fixedCircuit.csv"
        file_path = os.path.join(save_path, file_name)
        
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y"])
            writer.writerows(zip(Ts, res))
        print(f"Lie data saved to {file_path}")

        save_path = os.path.join("TE-PAI-noSampling", "data", "bonds")
        os.makedirs(save_path, exist_ok=True)
        file_name = f"lie-bond-N-{N}-T-{T}-q-{q}.csv"
        file_path = os.path.join(save_path, file_name)
        
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y"])
            writer.writerows(zip(Ts, complexity))
        print(f"Lie bonds saved to {file_path}")

    if not compare:
        return (Ts, res, complexity, circuit) 
    
    if compare:
        checkEqual(gates, gatesSim)
        resSim, gatesSim = trotterSimulation(hamil, N, n_snapshot, 1, 1, T, q)
        print(resSim)
        print(res)
        plt.scatter(range(len(resSim)), resSim, color='blue', label='Qiskit')
        plt.scatter(range(len(res)), res, color='red', label='Quimb')
        plt.legend()
        plt.show()

def quimb_to_qiskit(quimb_circ):
    """
    Convert a quimb circuit (an instance of qtn.Circuit) into a Qiskit QuantumCircuit.
    
    This function extracts the gate information from the quimb circuit and, 
    using a simple mapping, adds equivalent instructions to a new Qiskit circuit.
    
    Parameters
    ----------
    quimb_circ : qtn.Circuit
        A quimb circuit instance. It is assumed to have:
          • an attribute `N` for the number of qubits,
          • a property `gates` which is a sequence of Gate objects.
    
    Returns
    -------
    QuantumCircuit
        A Qiskit QuantumCircuit with the gates added in the same order.
    """
    # Import Qiskit and some gate classes we need
    from qiskit.circuit.library import (
        HGate, XGate, YGate, ZGate,
        RXGate, RYGate, RZGate, RXXGate, RYYGate, RZZGate, U3Gate,
        SwapGate, iSwapGate
    )

    # Create a Qiskit circuit with the same number of qubits
    qc = QuantumCircuit(quimb_circ.N)

    print(f"Number of gates: {len(quimb_circ.gates)}")
    
    # Loop over each gate in the quimb circuit (assumed to be in sequential order)
    for gate in quimb_circ.gates:
        # Normalize the gate label to uppercase for consistency.
        label = gate.label.upper()
        
        # Check if this gate has any control qubits.
        if gate.controls:
            # For controlled gates, we need to create a controlled version of a base gate.
            # We set up a mapping from the base gate label to a Qiskit gate class.
            base_gate_classes = {
                'H': HGate,
                'X': XGate,
                'Y': YGate,
                'Z': ZGate,
                'RX': RXGate,
                'RY': RYGate,
                'RZ': RZGate,
                'RXX': RXXGate,
                'RYY': RYYGate,
                'RZZ': RZZGate,
                'U3': U3Gate,
                'SWAP': SwapGate,
                'ISWAP': iSwapGate,
            }
            if label in base_gate_classes:
                params = gate.params if gate.params else []
                # Instantiate the base gate (pass parameters if any)
                base_gate = base_gate_classes[label](*params) if params else base_gate_classes[label]()
                # Create the controlled version. The number of controls is len(gate.controls)
                controlled_gate = base_gate.control(len(gate.controls))
                # In Qiskit the qubits for a controlled gate are provided as a list
                # with the control qubits first then the target(s)
                qubits = list(gate.controls) + list(gate.qubits)
                qc.append(controlled_gate, qubits)
            else:
                print("Warning: Unknown controlled gate:", gate.label)
        else:
            # No controls: use a dictionary mapping for single- or two-qubit gates.
            mapping = {
                'H': lambda qc, g: qc.h(g.qubits[0]),
                'X': lambda qc, g: qc.x(g.qubits[0]),
                'Y': lambda qc, g: qc.y(g.qubits[0]),
                'Z': lambda qc, g: qc.z(g.qubits[0]),
                'RX': lambda qc, g: qc.rx(g.params[0], g.qubits[0]),
                'RY': lambda qc, g: qc.ry(g.params[0], g.qubits[0]),
                'RZ': lambda qc, g: qc.rz(g.params[0], g.qubits[0]),
                'U3': lambda qc, g: qc.u3(g.params[0], g.params[1], g.params[2], g.qubits[0]),
                'CNOT': lambda qc, g: qc.cx(g.qubits[0], g.qubits[1]),
                'CX': lambda qc, g: qc.cx(g.qubits[0], g.qubits[1]),
                'CZ': lambda qc, g: qc.cz(g.qubits[0], g.qubits[1]),
                'SWAP': lambda qc, g: qc.swap(g.qubits[0], g.qubits[1]),
                'ISWAP': lambda qc, g: qc.iswap(g.qubits[0], g.qubits[1]),
                'RXX': lambda qc, g: qc.append(RXXGate(g.params[0]), [g.qubits[0], g.qubits[1]]),
                'RYY': lambda qc, g: qc.append(RYYGate(g.params[0]), [g.qubits[0], g.qubits[1]]),
                'RZZ': lambda qc, g: qc.append(RZZGate(g.params[0]), [g.qubits[0], g.qubits[1]]),

            }
            if label in mapping:
                gate = mapping[label](qc, gate)
            else:
                print("Warning: Unknown gate:", gate.label)
    
    return qc

def parse(folder, isJSON, draw, saveAndPlot, optimize=False, flip=False):
    folder = strip_trailing_dot_zero(folder)

    if isJSON:
        data_dict = JSONtoDict(folder)
    else:
        data_dict = HDF5toDict(folder)
    data_arrs,Ts,params,pool = DictToArr(data_dict, isJSON)
    N,n,c,Δ,T,q = params
    T = round(float(T), 8)
    char = "c"
    if not pool:
        averages,stds, costs = getSucessive(data_arrs,int(q), flip=flip)
    if pool:
        dT = extract_dT_value(folder)
        char = "p"
        averages, stds, circuit, costs = getPool(data_arrs, params,dT, draw, optimize, flip=flip)
        Ts = np.linspace(dT,T,len(averages))
        if draw:
            circuit.psi.draw(color=['PSI0', 'RZ', 'RZZ', 'RXX', 'RYY', 'H'], layout="kamada_kawai")
            if optimize:
                circuit.amplitude_rehearse(simplify_sequence='ADCRS')['tn'].draw(color=['PSI0', 'RZ', 'RZZ', 'RXX', 'RYY', 'H'], layout="kamada_kawai")
            qiskit = quimb_to_qiskit(circuit)
            if optimize:
                qiskit = transpile(qiskit, optimization_level=3)
            qiskit.draw("mpl", scale=0.4)
            plt.show()

    pattern = r"dT-([0-9]+(?:\.[0-9]+)?)"
    match = re.search(pattern, folder)
    saveData(N,n,c,Δ,Ts,q,float(match.group(1)),averages,stds,char)
    if saveAndPlot:
        trotter(100,10,float(T),int(q),compare=False,save=True, flip=flip)
        plot_data_from_folder("TE-PAI-noSampling/data/plotting")

    if costs[0] != None:
        return costs

def save_trotter(x, y, N, n, T, q, base_dir='TE-PAI-noSampling/data/plotting'):
    # Ensure target directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Build filename
    filename = f"trotter-bonds-N-{N}-n-{n}-{T}-q-{q}.csv"
    filepath = os.path.join(base_dir, filename)
    
    # Write CSV
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])
        for xi, yi in zip(x, y):
            writer.writerow([xi, yi])

def trotterThenTEPAI(folder, trotterN, trottern, trotterT, saveAndPlot=False, optimize=False, flip=False, confirm=False):
    folder = strip_trailing_dot_zero(folder)
    data_dict = JSONtoDict(folder)
    data_arrs,Ts,params,pool = DictToArr(data_dict, True)
    N,n,c,Δ,T,q = params
    T = round(float(T), 8)

    # Running the trotter simulation up to a time trotterT
    # NB: T =/= trotterT
    trotterTs, res, complexity, circuit = trotter(N=trotterN, n_snapshot=trottern, T=trotterT, q=int(q), compare=False, save=True, draw=False, flip=flip)
    save_trotter(trotterTs, complexity, trotterN, trottern, trotterT, q)

    if confirm:
        trotter(N=trotterN, n_snapshot=10, T=T, startTime=trotterT, q=int(q), compare=False, save=True, draw=False, flip=flip, fixedCircuit=circuit, )

    char = "c"
    dT = extract_dT_value(folder)
    char = "p"
    averages, stds, circuit, costs = getPool(data_arrs, params, dT, False, optimize, flip=flip, fixedCircuit = circuit)
    Ts = np.linspace(trotterT,trotterT + T,len(averages))

    pattern = r"dT-([0-9]+(?:\.[0-9]+)?)"
    match = re.search(pattern, folder)
    saveData(N,n,c,Δ,Ts,q,float(match.group(1)),averages,stds,char)
    organize_trotter_tepai()
   
def organize_trotter_tepai(
    plotting_dir: Path = Path("TE-PAI-noSampling/data/plotting"),
    target_base:  Path = Path("TE-PAI-noSampling/data/trotterThenTEPAI"),
):
    """
    Finds in `plotting_dir`:
      • One or two files matching `lie-N-{N}-T-{T}-q-{q}(…).csv`
      • Exactly one TE-PAI file matching
          N-{N2}-n-{n}-p-{p}-Δ-pi_over_{X}-T-{T2}-q-{q}-dT-{dt}.csv
      • Any number of trotter-bonds files matching
          trotter-bonds-N-{N}-n-{n}-{T}-q-{q}.csv

    Builds (or replaces) a folder named
      q-{q}-N1-{N1}-T1-{T1}-N2-{N2}-p-{p}-T2-{T2}-dt-{dt}

    Moves all lie-file(s), all trotter-bonds files, and the TE-PAI file
    into that folder, then into `target_base`.
    """
    # patterns
    pat_lie = re.compile(
        r"^lie-N-(?P<N>\d+)-T-(?P<T>[0-9.]+)-q-(?P<q>\d+)(?:-[^.]+)?\.csv$"
    )
    pat_tep = re.compile(
        r"^N-(?P<N2>\d+)-n-(?P<n>\d+)-p-(?P<p>\d+)-Δ-pi_over_[^–]+"
        r"-T-(?P<T2>[0-9.]+)-q-(?P<q>\d+)-dT-(?P<dt>[0-9.]+)\.csv$"
    )
    pat_trot = re.compile(
        r"^trotter-bonds-N-(?P<N>\d+)-n-(?P<n>\d+)-(?P<T>[0-9.]+)-q-(?P<q>\d+)\.csv$"
    )

    lie_runs    = []
    tep_match   = None
    trotter_runs = []

    # scan directory
    for f in plotting_dir.iterdir():
        if not f.is_file():
            continue
        if (m := pat_lie.match(f.name)):
            gd = m.groupdict()
            lie_runs.append((f,
                             float(gd["T"]),    # for sorting
                             gd["T"],           # for exact naming
                             gd["N"],
                             gd["q"]))
        elif (m := pat_tep.match(f.name)):
            tep_match = (f, m.groupdict())
        elif (m := pat_trot.match(f.name)):
            gd = m.groupdict()
            trotter_runs.append((f,
                                  gd["q"]))

    # require ≥1 lie-file and exactly 1 TE-PAI
    if len(lie_runs) < 1 or tep_match is None:
        raise FileNotFoundError(
            f"Need at least 1 lie-file + 1 TE-PAI file in {plotting_dir}, "
            f"found {len(lie_runs)} lie and "
            f"{'none' if tep_match is None else 'one'} TE-PAI."
        )

    # collect all q-values for consistency
    q_vals = { q for *_, q in lie_runs } \
           | { tep_match[1]["q"] } \
           | { q for _, q in trotter_runs }
    if len(q_vals) != 1:
        raise ValueError(f"q mismatch among files: {q_vals}")
    q_str = q_vals.pop()

    # pick the lie-run with smallest T
    lie_runs.sort(key=lambda tup: tup[1])
    _, _, T1_str, N1_str, _ = lie_runs[0]

    # TE-PAI params (use raw strings for naming)
    tep_f, tep_g = tep_match
    N2_str = tep_g["N2"]
    p_str  = tep_g["p"]
    T2_str = tep_g["T2"]
    dt_str = tep_g["dt"]

    # build folder name
    folder_name = (
        f"q-{q_str}"
        f"-N1-{N1_str}-T1-{T1_str}"
        f"-N2-{N2_str}-p-{p_str}-T2-{T2_str}-dt-{dt_str}"
    )

    # ensure target base exists
    target_base.mkdir(parents=True, exist_ok=True)

    staging = plotting_dir / folder_name
    target  = target_base  / folder_name

    # replace existing
    if staging.exists():
        shutil.rmtree(staging)
    if target.exists():
        shutil.rmtree(target)

    staging.mkdir(parents=True)

    # move lie-files
    for f, _, _, _, _ in lie_runs:
        shutil.move(str(f), staging / f.name)

    # move trotter-bonds files
    for f, _ in trotter_runs:
        shutil.move(str(f), staging / f.name)

    # move TE-PAI file
    shutil.move(str(tep_f), staging / tep_f.name)

    # relocate folder to target_base
    shutil.move(str(staging), target)

    print(f"Moved {len(lie_runs)} lie-file(s), {len(trotter_runs)} trotter-bonds file(s), "
          f"+ 1 TE-PAI file into {target}")

def plot_trotter_then_tepai(
    q: int,
    N1: int,
    T1: float,
    N2: int,
    p: int,
    T2: float,
    dt: float,
    base_dir: Path = Path("TE-PAI-noSampling/data/trotterThenTEPAI")
):
    """
    Locate folder q-{q}-N1-{N1}-T1-{T1}-N2-{N2}-p-{p}-T2-{T2}-dt-{dt}, then:
      • If 2 lie-*.csv:
         – smaller-T: darkblue solid
         – larger-T: gray dashed
      • If 1 lie-*.csv:
         – that run: darkblue solid
      • TE-PAI: tab:blue errorbars

    X axis = time, Y axis = x expectation value.
    Title = "Trotterization followed by TE-PAI for {q} qubits for total time {T1+T2}"
    """
    folder = base_dir / f"q-{q}-N1-{N1}-T1-{float(T1)}-N2-{N2}-p-{p}-T2-{float(T2)}-dt-{dt}"
    if not folder.is_dir():
        raise FileNotFoundError(f"No such folder: {folder}")

    pat_lie = re.compile(r"^lie-N-(?P<N>\d+)-T-(?P<T>[0-9.]+)-q-(?P<q>\d+)(?:-fixedCircuit)?\.csv$")
    pat_tep = re.compile(
        r"^N-(?P<N2>\d+)-n-(?P<n>\d+)-p-(?P<p>\d+)-Δ-pi_over_[^–]+"
        r"-T-(?P<T2>[0-9.]+)-q-(?P<q>\d+)-dT-(?P<dt>[0-9.]+)\.csv$"
    )

    lie_runs = []
    tep_file = None

    for f in folder.iterdir():
        if not f.is_file(): continue
        if (m := pat_lie.match(f.name)):
            gd = m.groupdict()
            lie_runs.append((f, float(gd["T"]), int(gd["N"])))
        elif pat_tep.match(f.name):
            tep_file = f

    if tep_file is None:
        raise FileNotFoundError("No TE-PAI file found in folder")

    # load TE-PAI
    x_t, y_t, e_t = [], [], []
    with tep_file.open() as fp:
        next(fp)
        for line in fp:
            xi, yi, erri = line.strip().split(',')
            x_t.append(float(xi)); y_t.append(float(yi)); e_t.append(float(erri))

    plt.figure()
    # handle 1 or 2 lie runs
    if len(lie_runs) == 1:
        file_s, _, N_s = lie_runs[0]
        x_s, y_s = [], []
        with file_s.open() as fp:
            next(fp)
            for line in fp:
                xi, yi = line.strip().split(',')
                x_s.append(float(xi)); y_s.append(float(yi))
        plt.plot(x_s, y_s,
                 linestyle='-',
                 color='darkblue',
                 label=f"Trotterization-N-{N_s}")
    elif len(lie_runs) == 2:
        # sort by T
        lie_runs.sort(key=lambda t: t[1])
        (f_s, _, N_s), (f_l, _, N_l) = lie_runs
        # small T
        x_s, y_s = [], []
        with f_s.open() as fp:
            next(fp)
            for line in fp:
                xi, yi = line.strip().split(',')
                x_s.append(float(xi)); y_s.append(float(yi))
        plt.plot(x_s, y_s,
                 linestyle='-',
                 color='darkblue',
                 label=f"Trotterization-N-{N_s}")
        # large T
        x_l, y_l = [], []
        with f_l.open() as fp:
            next(fp)
            for line in fp:
                xi, yi = line.strip().split(',')
                x_l.append(float(xi)); y_l.append(float(yi))
        plt.plot(x_l, y_l,
                 linestyle='--',
                 color='gray',
                 label=f"Trotterization-N-{N_l}")
    else:
        raise FileNotFoundError("Expected 1 or 2 lie files in folder")

    # TE-PAI continuation
    plt.errorbar(x_t, y_t, yerr=e_t,
                 fmt='o', linestyle='-',
                 color='tab:blue',
                 label=f"TE-PAI continuation-N-{N2}-p-{p}")

    plt.xlabel("time")
    plt.ylabel("x expectation value")
    total_time = T1 + T2
    plt.title(f"Trotterization followed by TE-PAI for {q} qubits for total time {total_time}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def compareCosts(poolCosts, successCosts, T, N):
    print(f"Pool total costs per timestep: {poolCosts}")
    print(f"Successive subtotal costs per timesteps: {successCosts}")
    successCosts = np.add.accumulate(np.array(successCosts))
    print(f"Successive total costs per timestep: {successCosts}")

    times = np.linspace(0, T, N)
    
    # Take logarithm of the costs
    log_poolCosts = np.log(poolCosts)
    log_successCosts = np.log(successCosts)
    
    # Fit a straight line to the log of poolCosts
    slope_pool, intercept_pool, r_value_pool, _, _ = linregress(times, log_poolCosts)
    
    # Fit a straight line to the log of successCosts
    slope_success, intercept_success, r_value_success, _, _ = linregress(times, log_successCosts)
    
    # Generate fitted exponential curves
    fitted_pool = np.exp(intercept_pool + slope_pool * times)
    fitted_success = np.exp(intercept_success + slope_success * times)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(5, 10), sharex=True)

    # First subplot: Original plot
    axes[0].semilogy(times, poolCosts, label="Contraction cost of pool approach")
    axes[0].semilogy(times, successCosts, label="Contraction cost of successive approach")
    axes[0].set_ylabel("Average total cost")
    axes[0].set_title("Plot of Averages vs T with Error Bars")
    axes[0].legend()
    axes[0].grid(True)

    # Second subplot: Log-transformed data to check for exponential growth
    axes[1].scatter(times, log_poolCosts, label="Log of Pool Costs")
    axes[1].plot(times, intercept_pool + slope_pool * times, color='red', linestyle="dashed", label=f"Linear Fit (R²={r_value_pool**2:.4f})")
    axes[1].set_ylabel("log(Total cost)")
    axes[1].set_title("Log of Pool Costs vs Time")
    axes[1].legend()
    axes[1].grid(True)

    # Third subplot: Log-transformed data for successive approach
    axes[2].scatter(times, log_successCosts, label="Log of Successive Costs")
    axes[2].plot(times, intercept_success + slope_success * times, color='blue', linestyle="dashed", label=f"Linear Fit (R²={r_value_success**2:.4f})")
    axes[2].set_ylabel("log(Total cost)")
    axes[2].set_title("Log of Successive Costs vs Time")
    axes[2].legend()
    axes[2].grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()

def showComplexity(costs, T, N, output_folder=None):
    print(f"Costs: {costs}")
    times = np.linspace(0, T, N)
    
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        df = pd.DataFrame({'Time': times, 'Cost': costs})
        csv_path = os.path.join(output_folder, 'contraction_costs.csv')
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
    
    #plt.plot(times, costs, label="Contraction costs")
    #plt.xlabel("Time")
    #plt.ylabel("Total cost")
    #plt.title("Contraction costs over time")
    #plt.legend()
    #plt.grid(True)
    #plt.show()
    
def getComplexity(circuit):
    #result = circuit.amplitude_rehearse(optimize="greedy")
    #return result['W']
    return circuit.psi.max_bond()
    #rehs = circuit.to_dense_rehearse()
    #cs = rehs['tree'].contraction_cost()
   #return cs

def plotComplexityFromFolder(folder_path, semilogy=True):
    if folder_path is None:
        print("No folder path provided.")
        return
    
    csv_path = os.path.join(folder_path, 'contraction_costs.csv')
    
    if not os.path.isfile(csv_path):
        print(f"No CSV file found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    if semilogy:
        plt.semilogy(df['Time'], df['Cost'], label="bond")
    else:
        plt.plot(df['Time'], df['Cost'], label="bond")
    plt.xlabel("T / T_max")
    plt.ylabel("Max_bond")
    plt.title("The size of the largest bond over time")
    plt.legend()
    plt.grid(True)
    plt.show()

def getCircuit(q, flip=False):
    quimb = qtn.CircuitMPS(q, cutoff = 1e-12)

    for i in range(q):
        quimb.apply_gate('H', qubits=[i])

    if flip:
        middle = np.floor(q/2)
        quimb.apply_gate('Z', qubits=[int(middle)])
        #state_vector = quimb.psi.to_dense()
        #qu.cprint(state_vector)

    return quimb

def strip_trailing_dot_zero(folder_name):
    if '-T-' in folder_name:
        parts = folder_name.split('-T-')
        head = parts[0]
        tail = parts[1]
        if tail.endswith('.0'):
            tail = tail[:-2]  # remove the .0
        return f"{head}-T-{tail}"
    return folder_name

def plot_bond_data(folder_path="TE-PAI-noSampling/data/bonds"):
    """
    Scans the given folder for CSV files matching either pattern:
      • lie-bond-N-{N}-T-{T}-q-{q}.csv
      • trotter-bonds-N-{N}-n-{n}-{T}-q-{q}.csv

    Reads each file (ignoring the first row as a header),
    and plots time vs. max_bond curves on a single plot.
    If no matching files are found, simply returns without error.
    """
    pat_lie = re.compile(
        r"^lie-bond-N-(?P<N>[^-]+)-T-(?P<T>[^-]+)-q-(?P<q>[^.]+)\.csv$"
    )
    pat_trot = re.compile(
        r"^trotter-bonds-N-(?P<N>[^-]+)-n-(?P<n>[^-]+)-(?P<T>[^-]+)-q-(?P<q>[^.]+)\.csv$"
    )

    plt.figure()
    found_any = False

    for filename in os.listdir(folder_path):
        fp = os.path.join(folder_path, filename)

        m1 = pat_lie.match(filename)
        if m1:
            gd = m1.groupdict()
            df = pd.read_csv(fp, header=0)
            x, y = df.iloc[:, 0], df.iloc[:, 1]
            label = f"lie-bond N={gd['N']}, T={gd['T']}, q={gd['q']}"
            plt.plot(x, y, label=label)
            found_any = True
            continue

        m2 = pat_trot.match(filename)
        if m2:
            gd = m2.groupdict()
            df = pd.read_csv(fp, header=0)
            x, y = df.iloc[:, 0], df.iloc[:, 1]
            label = f"trotter-bonds N={gd['N']}, n={gd['n']}, T={gd['T']}, q={gd['q']}"
            plt.plot(x, y, label=label)
            found_any = True
            continue

    if not found_any:
        return

    plt.title("Bond size over time")
    plt.xlabel("time")
    plt.ylabel("max_bond")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_bond_data(r"TE-PAI-noSampling\data\trotterThenTEPAI\q-10-N1-100-T1-2-N2-1000-p-100-T2-3.0-dt-0.1")
#organize_trotter_tepai()
if False:
    plot_trotter_then_tepai(
        q = 10,
        N1= 100,
        T1= 2,
        N2= 1000,
        p =100,
        T2 = 3,
        dt= 0.1)

path = "TE-PAI-noSampling/data/circuits/N-1000-n-1-p-50-Δ-pi_over_1024-q-4-dT-0.1-T-1.0"
#plotComplexityFromFolder(path, False)
#plot_data_from_folder("TE-PAI-noSampling/data/plotting")

if False:
    #trotter(500, 10, 0.1, 10, compare=False, save=True, draw=False)
    costs = parse(path, 
                isJSON=True, 
                draw=False, 
                saveAndPlot=True, 
                optimize=False)
    showComplexity(costs, 1, 10, path)

if False:
    poolCosts = parse("TE-PAI-noSampling/data/circuits/N-1000-n-1-p-100-Δ-pi_over_1024-q-4-dT-0.01-T-0.1", 
        isJSON=True, 
        draw=False, 
        saveAndPlot=False, 
        optimize=False)

    successCosts = parse("TE-PAI-noSampling/data/circuits/N-1000-n-1-c-10-Δ-pi_over_1024-q-4-dT-0.01-T-0.1", 
        isJSON=True, 
        draw=False, 
        saveAndPlot=False, 
        optimize=False)

    compareCosts(poolCosts, successCosts, 0.1, 10)