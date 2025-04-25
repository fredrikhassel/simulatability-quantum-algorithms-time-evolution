import csv
import os
import json
import re
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
    files = os.listdir(folder_name)
    
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

def getSucessive(data_arrs,q):
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
            quimb = getCircuit(q)
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

def getPool(data_arrs,params,dT, draw, optimize=None):
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
        quimb = getCircuit(q)
        for i,(circuit,sign) in enumerate(zip(circuits,signs)):
            applyGates(quimb,circuit)
            costs[i].append(getComplexity(quimb))#quimb.psi.contraction_cost())
            #quimb.amplitude_rehearse(simplify_sequence='RL')['tn'].draw(color=['PSI0', 'RZ', 'RZZ', 'RXX', 'RYY', 'H'], layout="kamada_kawai")
            results[i].append(measure(quimb,q, optimize)*sign)

    costs = np.mean(costs, axis=1)
    averages = np.mean(results, axis=1)
    stds = np.std(results, axis=1)

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

def trotter(N, n_snapshot, T, q, compare, save=False, draw=False):
    times = np.linspace(0, float(T), int(N))
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
    
    
    # Create a NEW circuit for EACH gate group — this is the key change!
    res = [1]
    for i, gs in enumerate(gates):
        print(f"Snapshot {i+1} / {len(gates)}")

        # Fresh circuit each time
        circuit = getCircuit(q)

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
        res.append(result)

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

def parse(folder, isJSON, draw, saveAndPlot, optimize=False):
    if isJSON:
        data_dict = JSONtoDict(folder)
    else:
        data_dict = HDF5toDict(folder)
    data_arrs,Ts,params,pool = DictToArr(data_dict, isJSON)
    N,n,c,Δ,T,q = params
    T = round(float(T), 8)
    char = "c"
    if not pool:
        averages,stds, costs = getSucessive(data_arrs,int(q))
    if pool:
        dT = extract_dT_value(folder)
        char = "p"
        averages, stds, circuit, costs = getPool(data_arrs, params,dT, draw, optimize)
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
        lie = trotter(100,10,float(T),int(q),compare=False,save=True)
        plot_data_from_folder("TE-PAI-noSampling/data/plotting")

    if costs[0] != None:
        return costs

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
        plt.semilogy(df['Time'], df['Cost'], label="Contraction costs")
    else:
        plt.plot(df['Time'], df['Cost'], label="Contraction costs")
    plt.xlabel("Time")
    plt.ylabel("Total cost")
    plt.title("Contraction costs over time (from CSV)")
    plt.legend()
    plt.grid(True)
    plt.show()

def getCircuit(q):
    quimb = qtn.CircuitMPS(q, cutoff = 1e-12)
    for i in range(q):
        quimb.apply_gate('H', qubits=[i])
    return quimb

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