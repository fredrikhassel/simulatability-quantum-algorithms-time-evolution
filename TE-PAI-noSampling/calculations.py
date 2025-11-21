"""
Core calculation functions for TE-PAI package.
"""
import os
import json
import re
from collections import defaultdict
import numpy as np
import quimb.tensor as qtn
import quimb as qu
import h5py
from HAMILTONIAN import Hamiltonian
from TROTTER import Trotter
from main import TE_PAI
from scipy.stats import linregress
from scipy.linalg import expm
from itertools import combinations
import csv
import pandas as pd
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from helpers import saveData, save_trotter, strip_trailing_dot_zero, organize_trotter_tepai, quimb_to_qiskit
from plotting import plot_data_from_folder
from qiskit import QuantumCircuit, transpile
import sys
import numpy as np
import multiprocessing as mp


# --- Data loading and parsing ---
def JSONtoDict(folder_name):

    files = os.listdir(folder_name)

    # Regex: optional batch digits after prefix, then the six parameters
    gates_pattern = re.compile(
        r"^gates_arr(\d*)-N-(\d+)-n-(\d+)-[cp]-(\d+)-Δ-([\w_]+)-T-([\d.]+)-q-(\d+)\.json$"
    )
    sign_pattern = re.compile(
        r"^sign_list(\d*)-N-(\d+)-n-(\d+)-[cp]-(\d+)-Δ-([\w_]+)-T-([\d.]+)-q-(\d+)\.json$"
    )

    # Collect all filepaths by parameter-set → list of (batch, path)
    gates_files = defaultdict(list)
    sign_files = defaultdict(list)

    for fname in files:
        gm = gates_pattern.match(fname)
        sm = sign_pattern.match(fname)

        if gm:
            batch_str, *params = gm.groups()
            batch = int(batch_str) if batch_str else 1
            key = tuple(params)  # (N, n, cp, Δ, T, q)
            gates_files[key].append((batch, os.path.join(folder_name, fname)))

        elif sm:
            batch_str, *params = sm.groups()
            batch = int(batch_str) if batch_str else 1
            key = tuple(params)
            sign_files[key].append((batch, os.path.join(folder_name, fname)))

    paired_data = {}

    # For each parameter-set that has gates files...
    for key, g_list in gates_files.items():
        if key not in sign_files:
            for _b, path in g_list:
                print(f"Warning: No matching sign_list file for {path}")
            continue

        s_list = sign_files[key]

        # Align on batch index
        batches = sorted({b for b, _ in g_list} & {b for b, _ in s_list})
        if len(batches) < len(g_list) or len(batches) < len(s_list):
            # some gates or signs are unmatched
            extra_g = set(b for b, _ in g_list) - set(batches)
            extra_s = set(b for b, _ in s_list) - set(batches)
            for b in extra_g:
                print(f"Warning: No matching sign_list for gates_arr{b}-…")
            for b in extra_s:
                print(f"Warning: No matching gates_arr for sign_list{b}-…")

        # Build offset-rebased combined dicts
        combined_gates = {}
        combined_signs = {}
        offset = 0

        for b in sorted(batches):
            g_path = next(path for (batch, path) in g_list if batch == b)
            s_path = next(path for (batch, path) in s_list if batch == b)

            with open(g_path, 'r') as f:
                g_data = json.load(f)
            with open(s_path, 'r') as f:
                s_data = json.load(f)


            # pull out overhead if you need it later (optional)
            overhead = s_data.get("overhead")

            # only keep strictly numeric keys for sorting & rebasing
            numeric_keys = [k for k in s_data.keys() if k.isdigit()]

            for orig_key in sorted(numeric_keys, key=int):
                new_key = str(offset + int(orig_key))
                combined_signs[new_key] = s_data[orig_key]

            # if you still want to carry the overhead float through:
            if overhead is not None:
                combined_signs["overhead"] = overhead

            # Merge g_data
            for orig_key in sorted(g_data.keys(), key=int):
                new_key = str(offset + int(orig_key))
                combined_gates[new_key] = g_data[orig_key]

            # bump offset
            offset = max(offset, max(int(k) for k in combined_gates.keys()))

        paired_data[key] = {
            "gates": combined_gates,
            "signs": combined_signs
        }

    # Warn about any sign-only sets
    for key in sign_files:
        if key not in gates_files:
            for b, path in sign_files[key]:
                print(f"Warning: No matching gates_arr file for {path}")

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

def parse(folder, isJSON, draw, saveAndPlot, H_name, optimize=False, flip=False):
    if not os.path.isdir(folder):
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
        averages, stds, circuit, costs = getPool_mp(data_arrs, params,dT, draw, optimize, flip=flip)
        Ts = np.linspace(0,T,len(averages))
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
    saveData(N,n,c,Δ,Ts,q,float(match.group(1)),averages,stds,char,H_name)

    if saveAndPlot:
        trotter(100,10,float(T),int(q),compare=False,save=True, flip=flip)
        plot_data_from_folder("TE-PAI-noSampling/data/plotting")

    if costs[0][0] != None:
        return costs

def parse_path(path):
    pattern = re.compile(
        r'N-(?P<N>[^-]+)'
        r'-n-(?P<n>[^-]+)'
        r'-p-(?P<p>[^-]+)'
        r'-Δ-(?P<Δ>[^-]+)'
        r'-q-(?P<q>[^-]+)'
        r'-dT-(?P<dT>[^-]+)'
        r'-T-(?P<T>[^-]+)'
    )
    m = pattern.search(path)
    if not m:
        raise ValueError(f"Path does not match expected format: {path!r}")

    def _convert(val):
        # try int, then float, otherwise leave as string
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val

    params = {k: _convert(v) for k, v in m.groupdict().items()}
    return params

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

def generate_random_indices(pool_size, output_length, entry_length):
    rng = np.random.default_rng()  # Create RNG instance with optional seed
    return [list(rng.integers(0, pool_size, size=entry_length)) for _ in range(output_length)]

def getCircuit(q, flip=False, mps=True, max_bond = None):
    if mps:
        if max_bond is None:
            quimb = qtn.CircuitMPS(q, cutoff = 1e-12)
        else:
            quimb = qtn.CircuitMPS(q, max_bond = max_bond, cutoff = 1e-12)
    else:
        quimb = qtn.Circuit(q)
    for i in range(q):
        quimb.apply_gate('H', qubits=[i])
    if flip:
        middle = np.floor(q/2)
        quimb.apply_gate('Z', qubits=[int(middle)])
    return quimb

def showComplexity(costs, T, N, output_folder=None):
    print(f"Costs: {costs}")
    times = np.linspace(0, T, N)
    
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        df = pd.DataFrame({'Time': times, 'Cost': costs})
        csv_path = os.path.join(output_folder, 'contraction_costs.csv')
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")

def save_lengths(n,q,tepai_dT,n1,n2,N1,N2,Δ,NNN,base_dir):
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    if not NNN:
        hamil = Hamiltonian.spin_chain_hamil(q, freqs)
        len1 = q*(1+4*N1)
        len2 = q*(1+4*N2)
    if NNN:
        hamil = Hamiltonian.next_nearest_neighbor_hamil(q, freqs)
        len1 = q*(1+7*N1)
        len2 = q*(1+7*N2)
    te_pai = TE_PAI(hamil, q, Δ, tepai_dT, 1000, 1)
    lentep = te_pai.expected_num_gates
    lengths1 = [(len1/n1)*i for i in range(1, n1+1)]
    lengths2 = [(len2/n2)*i for i in range(1, n1+1)]
    lengthstep = [(lentep/n)*i for i in range(1, n+1)]
    filename = f"lengths.csv"
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['trotter1', 'trotter2', 'TEPAI'])
        writer.writerow([lengths1, lengths2, lengthstep])

# --- Gate application and measurement ---
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
            #print(f"Applying gate {quimb_gate_name} on qubit {qubit_indices[0]} with angle {angle}")
            circuit.apply_gate(gate_id=quimb_gate_name, qubits=qubit_indices, params=[angle])
        else:
            raise ValueError(f"Unsupported number of qubits for gate {quimb_gate_name}: {len(qubit_indices)}")

def measure(circuit):
    val = np.real(circuit.local_expectation(qu.pauli('X'), (0)))
    return (val+1)/2

# --- Main calculation functions ---
def mainCalc(tepaiPath, finalT1, N1, n1, N2, finalT2, confirm=False, flip=True):
    params = parse_path(tepaiPath)
    q = params['q']
    dT = params['dT']
    print(params['Δ'])
    pattern = r'^pi_over_(\d+(?:\.\d+)?)$'
    m = re.match(pattern, params['Δ'])
    divisor = float(m.group(1))
    Δ = np.pi / divisor
    NNN = "NNN_" in tepaiPath

    # Performing main trotterization
    ts1, res1, comp1, circ1 = trotter(N=N1, n_snapshot=n1, T=finalT1, q=int(q), compare=False, save=True, draw=False, flip=flip, NNN=NNN)
    circuit = circ1.copy()

    base_dir='TE-PAI-noSampling/data/plotting'
    if NNN:
        base_dir='TE-PAI-noSampling/NNN_data/plotting'

    save_trotter(ts1, comp1[0], comp1[1], N1, n1, finalT1, q, base_dir=base_dir)

    # Performing continuing trotterization
    if confirm:
            ts2, res2, comp2, circ2 = trotter(N=N2, n_snapshot=10, T=finalT2, startTime=finalT1, q=int(q), compare=False, save=True, draw=False, flip=flip, fixedCircuit=circuit, NNN=NNN)
            save_trotter(ts2, comp2[0], comp2[1], N2, 10, finalT2, q, fixed=True, base_dir=base_dir)

    # Performing TE-PAI
    tepaiPath = strip_trailing_dot_zero(tepaiPath)
    data_dict = JSONtoDict(tepaiPath)
    data_arrs,Ts,params2,pool = DictToArr(data_dict, True)
    circuit = circ1.copy()
    averages, stds, circuit, costs = getPool(data_arrs, params2, dT, False, False, flip=flip, fixedCircuit = circuit)
    Ts = np.linspace(finalT1,finalT1 + finalT2,len(averages))
    saveData(params['N'],params['n'],params['p'],params['Δ'],Ts,q,dT,averages,stds,"p", NNN=NNN)

    # Saving TEPAI costs
    filename = f"TEPAI-bonds-N-{params['N']}-n-{params['n']}-T-{params['T']}-q-{q}.csv"
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'z'])
        for xi, yi, zi in zip(Ts, costs[0], costs[1]):
            writer.writerow([xi, yi, zi])

    # Saving circuit lengths
    n = int(params['T'] / params['dT'])
    len1 = len(circ1.gates)
    len2 = len(circ2.gates)
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    if not NNN:
        hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    else:
        hamil = Hamiltonian.next_nearest_neighbor_hamil(q, freqs)
    te_pai = TE_PAI(hamil, q, Δ, params['dT'], 1000, 1)
    lentep = te_pai.expected_num_gates
    lengths1 = [(len1/n1)*i for i in range(1, n1+1)]
    lengths2 = [(len2/10)*i for i in range(1, 11)]
    lengthstep = [(lentep/n)*i for i in range(1, n+1)]
    filename = f"lengths.csv"
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['trotter1', 'trotter2', 'TEPAI'])
        writer.writerow([lengths1, lengths2, lengthstep])

    target_base = Path("TE-PAI-noSampling/data/trotterThenTEPAI")
    if NNN:
        target_base = Path("TE-PAI-noSampling/NNN_data/trotterThenTEPAI")
    organize_trotter_tepai(plotting_dir=base_dir, target_base=target_base)

def mainCalc2(tepaiPath, finalT1, N1, n1, finalT2, confirm=False, flip=True):
    params = parse_path(tepaiPath)
    q = params['q']
    dT = params['dT']
    print(params['Δ'])
    pattern = r'^pi_over_(\d+(?:\.\d+)?)$'
    m = re.match(pattern, params['Δ'])
    divisor = float(m.group(1))
    Δ = np.pi / divisor
    NNN = "NNN_" in tepaiPath
    base_dir = 'TE-PAI-noSampling/data/plotting'
    if NNN:
        base_dir = 'TE-PAI-noSampling/NNN_data/plotting'

    # Performing main trotterization
    ts1, res1, comp1, circuits = trotter(N=N1, n_snapshot=n1, T=finalT2, q=int(q), compare=False, save=True, draw=False, flip=flip, circuitList=True, NNN=NNN)
    #ts1 = np.linspace(0, finalT2, n1)
    circuit = None
    for index,t in enumerate(ts1):
        if t == finalT1:
            circuit = circuits[index]
            break
    if circuit == None:
        print(f"Error: No circuit found at {finalT1} when looking in {ts1}")
        return
    circ1 = circuit.copy()
    save_trotter(ts1, comp1[0], comp1[1], N1, n1, finalT2, q, base_dir=base_dir)

    # Performing TE-PAI
    tepaiPath = strip_trailing_dot_zero(tepaiPath)
    data_dict = JSONtoDict(tepaiPath)
    data_arrs,Ts,params2,pool = DictToArr(data_dict, True)
    circuit = circuit.copy()
    averages, stds, circuit, costs = getPool(data_arrs, params2, dT, False, False, flip=flip, fixedCircuit = circuit)
    Ts = np.linspace(finalT1,finalT2, len(averages))

    saveData(params['N'],params['n'],params['p'],params['Δ'],Ts,q,dT,averages,stds,"p",NNN=NNN)
    circ2 = circuits[-1].copy()

    # Saving TEPAI costs
    filename = f"TEPAI-bonds-N-{params['N']}-n-{params['n']}-T-{params['T']}-q-{q}.csv"
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'z'])
        for xi, yi, zi in zip(Ts, costs[0], costs[1]):
            writer.writerow([xi, yi, zi])

    # Saving circuit lengths
    n = int(params['T'] / params['dT'])
    len1 = len(circ1.gates)
    len2 = len(circ2.gates)
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    if not NNN:
        hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    if NNN:
        hamil = Hamiltonian.next_nearest_neighbor_hamil(q, freqs)
    te_pai = TE_PAI(hamil, q, Δ, params['dT'], 1000, 1)
    lentep = te_pai.expected_num_gates
    lengths1 = [(len1/n1)*i for i in range(1, n1+1)]
    lengths2 = [(len2/n1)*i for i in range(1, n1+1)]
    lengthstep = [(lentep/n)*i for i in range(1, n+1)]
    filename = f"lengths.csv"
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['trotter1', 'trotter2', 'TEPAI'])
        writer.writerow([lengths1, lengths2, lengthstep])

    target_base = Path("TE-PAI-noSampling/data/trotterThenTEPAI")
    if NNN:
        target_base = Path("TE-PAI-noSampling/NNN_data/trotterThenTEPAI")
    organize_trotter_tepai(plotting_dir=Path(base_dir), target_base=target_base)

def fullCalc(tepaiPath, T, N, n, flip=True, skip_trotter=False):
    params = parse_path(tepaiPath)
    q = params['q']
    dT = params['dT']
    print(params['Δ'])
    pattern = r'^pi_over_(\d+(?:\.\d+)?)$'
    m = re.match(pattern, params['Δ'])
    divisor = float(m.group(1))
    Δ = np.pi / divisor
    H_name = "SCH"
    base_dir = 'TE-PAI-noSampling/data/plotting'
    if "NNN_" in tepaiPath:
        H_name == "NNN"
        base_dir = 'TE-PAI-noSampling/NNN_data/plotting'
    if "2D_" in tepaiPath:
        H_name == "2D"
        base_dir = 'TE-PAI-noSampling/2D_data/plotting'

    # Performing main trotterization
    if not skip_trotter:
        ts1, res1, comp1, circuits = trotter(N=N, n_snapshot=n, T=T, q=int(q), compare=False, save=True, draw=False, flip=flip, circuitList=True)
        save_trotter(ts1, comp1[0], comp1[1], N, n, T, q, base_dir=base_dir)

    # Performing TE-PAI
    tepaiPath = strip_trailing_dot_zero(tepaiPath)
    data_dict = JSONtoDict(tepaiPath)
    data_arrs,Ts,params2,pool = DictToArr(data_dict, True)
    N2,n1,c1,Δ1,_,q1 = params2
    params2 = N2,n1,c1,Δ1,T,q1
    averages, stds, circuit, costs = getPool(data_arrs, params2, dT, False, False, flip=flip)
    circ1 = circuit.copy()
    Ts = np.linspace(0,T,len(averages))

    saveData(params['N'],params['n'],params['p'],params['Δ'],Ts,q,dT,averages,stds,"p", H_name)
    circ2 = circuits[-1].copy()

    # Saving TEPAI costs
    filename = f"TEPAI-bonds-N-{params['N']}-n-{params['n']}-T-{params['T']}-q-{q}.csv"
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'z'])
        for xi, yi, zi in zip(Ts, costs[0], costs[1]):
            writer.writerow([xi, yi, zi])

    # Saving circuit lengths
    n = int(params['T'] / params['dT'])
    len1 = len(circ1.gates)
    len2 = len(circ2.gates)
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    if H_name == "SCH":
        hamil = Hamiltonian.spin_chain_hamil(q, freqs)
        target_base = Path("TE-PAI-noSampling/data/fullCalc")
    if H_name == "NNN":
        hamil = Hamiltonian.next_nearest_neighbor_hamil(q, freqs)
        target_base = Path("TE-PAI-noSampling/NNN_data/fullCalc")
    if H_name == "2D":
        hamil = Hamiltonian.lattice_2d_hamil(q, freqs=freqs)
        target_base = Path("TE-PAI-noSampling/2D_data/fullCalc")
    te_pai = TE_PAI(hamil, q, Δ, params['dT'], 1000, 1)
    lentep = te_pai.expected_num_gates
    n1 = int(n1)
    lengths1 = [(len1/n1)*i for i in range(1, n1+1)]
    lengths2 = [(len2/n1)*i for i in range(1, n1+1)]
    lengthstep = [(lentep/n)*i for i in range(1, n+1)]
    filename = f"lengths.csv"
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['trotter1', 'trotter2', 'TEPAI'])
        writer.writerow([lengths1, lengths2, lengthstep])

    organize_trotter_tepai(plotting_dir=Path(base_dir), target_base=target_base)

def manyCalc(tepaiPath, Tf, Tis, N, n, flip=True,):
    params = parse_path(tepaiPath)
    q = params['q']
    dT = params['dT']
    print(params['Δ'])
    pattern = r'^pi_over_(\d+(?:\.\d+)?)$'
    m = re.match(pattern, params['Δ'])
    divisor = float(m.group(1))
    Δ = np.pi / divisor
    ts, res, comp, circuits = trotter(N=N, n_snapshot=n, T=Tf, q=int(q), compare=False, save=True, draw=False, flip=flip, circuitList=True)
    #save_trotter(ts, comp[0], comp[1], N, n, Tf, q)

    # Performing TE-PAI
    tepaiPath = strip_trailing_dot_zero(tepaiPath)
    data_dict = JSONtoDict(tepaiPath)
    data_arrs,Ts,params2,pool = DictToArr(data_dict, True)
    data_indices = []
    mini = 10000
    for T in Tis:
        change = Tf-T
        string = np.floor(change/dT)
        shots  = len(data_arrs[0][0]) / string
        if shots < mini:
            mini = shots
    for T in Tis:
        change = Tf-T
        string = np.floor(change/dT)
        data_indices.append(int(string * mini))

    # Calculating timestep lengths
    trotterLen = circuits[0].num_gates
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    te_pai = TE_PAI(hamil, q, Δ, params['dT'], 1000, 1)
    tepaiLen = te_pai.expected_num_gates

    for i,Ti in enumerate(Tis):
        if Ti == 0:
            circuit = getCircuit(q, flip, True)
        else:
            closest_index = min(range(len(ts)), key=lambda i: abs(ts[i] - Ti))
            circuit = circuits[closest_index]
        print(f"Starting with circuit with: {circuit.num_gates}  gates.")

        # Cutting data to same number of shots
        data = data_arrs
        circuit_pool, sign_pool = data[0]
        circuit_pool = circuit_pool[:data_indices[i]]
        sign_pool = sign_pool[:data_indices[i]]
        data[0] = circuit_pool, sign_pool

        # Changint parameters:
        N2,n,c,Δ,T,q = params2
        T = Tf-Ti
        params2 = N2,n,c,Δ,T,q

        # Performing TEPAI
        averages, stds, circuit, costs = getPool(data_arrs=data, params=params2, dT=dT, 
                                                 draw=False, optimize=False, flip=flip, fixedCircuit = circuit)
        Ts = np.linspace(Ti,Tf,len(averages))

        # Saving TEPAI
        base_dir='TE-PAI-noSampling/data/plotting'
        filename = f"N-{N}-n-{n}-p-{params['p']}-Δ-{params['Δ']}-T-{Tf}-q-{q}-dT-{dT}-Ti-{Ti}.csv"
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Lengths(Trotter-TEPAI)', trotterLen, tepaiLen])
            for xi, yi, zi in zip(Ts, averages, stds):
                writer.writerow([xi, yi, zi])
        base_dir='TE-PAI-noSampling/data/plotting'
        filename = f"TEPAI-bonds-N-{params['N']}-n-{params['n']}-T-{params['T']}-q-{q}-Ti-{Ti}.csv"
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y', 'z'])
            for xi, yi, zi in zip(Ts, costs[0], costs[1]):
                writer.writerow([xi, yi, zi])

        # Define the destination directory
        destination_root = 'TE-PAI-noSampling/data/manyCalc'
        foldername = f"N-{N}-p-{params['p']}-Δ-{params['Δ']}-T-{Tf}-q-{q}"
        destination_path = os.path.join(destination_root, foldername)

        # Create the folder if it doesn't exist
        os.makedirs(destination_path, exist_ok=True)

        # List of files to move
        files_to_move = [
            f"N-{N}-n-{n}-p-{params['p']}-Δ-{params['Δ']}-T-{Tf}-q-{q}-dT-{dT}-Ti-{Ti}.csv",
            f"TEPAI-bonds-N-{params['N']}-n-{params['n']}-T-{params['T']}-q-{q}-Ti-{Ti}.csv",
            f"lie-N-{N}-T-{Tf}-q-{q}.csv",
            f"lie-bond-N-{N}-T-{Tf}-q-{q}.csv"
        ]

        # Move each file
        source_base = 'TE-PAI-noSampling/data/plotting'
        for filename in files_to_move:
            src = os.path.join(source_base, filename)
            dst = os.path.join(destination_path, filename)
            if os.path.exists(src):
                shutil.move(src, dst)

        # Saving circuit lengths

# --- Performance and complexity ---
def getComplexity(circuit):
    mps = circuit.psi
    info = mps.contraction_info(optimize="greedy")
    return circuit.psi.max_bond(), info.naive_cost

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

def getPool(data_arrs,params, dT, draw, optimize=None, flip=False, fixedCircuit=None, start=None):
    N,n,c,Δ,T,q = params
    print(f"Pool with: T:{T}, dT:{dT}")
    T = float(T)
    q = int(q)
    circuit_pool, sign_pool = data_arrs[0]
    n_timesteps = round(T/dT)
    results = [[] for _ in range(n_timesteps)]
    bonds = [[] for _ in range(n_timesteps)]
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
            bond, cost = getComplexity(quimb)
            bonds[i].append(bond)
            costs[i].append(cost)
            results[i].append(measure(quimb,q, optimize)*sign)
            if start != None:
                print(f"Starting tol: {start} versus {cost}")
                if cost > start:
                    break

    costs = np.mean(costs, axis=1)
    bonds = np.mean(bonds, axis=1)
    averages = np.mean(results, axis=1)
    stds = np.std(results, axis=1)

    if fixedCircuit == None:
        averages = np.insert(averages, 0, 1)
    else:
        averages = np.insert(averages, 0, measure(fixedCircuit, q, optimize))
    stds = np.insert(stds, 0, 0)

    if draw:
        return averages,stds, quimb, [bonds,costs]
    else:
        return averages,stds, quimb, [bonds,costs]
    
# --- worker globals set via pool initializer ---
_CIRCUIT_POOL = None
_SIGN_POOL = None
_Q = None
_FLIP = None
_FIXED_CIRCUIT = None
_OPTIMIZE = None
_START = None

def _init_pool(circuit_pool, sign_pool, q, flip, fixedCircuit, optimize, start):
    """Initializer to set globals in worker processes (avoids re-sending big arrays)."""
    global _CIRCUIT_POOL, _SIGN_POOL, _Q, _FLIP, _FIXED_CIRCUIT, _OPTIMIZE, _START
    _CIRCUIT_POOL = circuit_pool
    _SIGN_POOL = sign_pool
    _Q = int(q)
    _FLIP = bool(flip)
    _FIXED_CIRCUIT = fixedCircuit
    _OPTIMIZE = optimize
    _START = start

def _single_run(run_indices):
    """Execute one run (one sequence of timesteps). Returns per-timestep scalars."""
    # Build circuit state for this run
    if _FIXED_CIRCUIT is None:
        quimb = getCircuit(_Q, flip=_FLIP)
    else:
        quimb = _FIXED_CIRCUIT.copy()

    bonds_run = []
    costs_run = []
    results_run = []

    # Advance through timesteps
    for idx in run_indices:
        circuit = _CIRCUIT_POOL[idx]
        sign = _SIGN_POOL[idx]
        applyGates(quimb, circuit)
        bond, cost = getComplexity(quimb)
        bonds_run.append(bond)
        costs_run.append(cost)
        results_run.append(measure(quimb, _Q, _OPTIMIZE) * sign)

        if _START is not None:
            # Keep this print to preserve original side-effect behavior
            print(f"Starting tol: {_START} versus {cost}")
            if cost > _START:
                break

    return bonds_run, costs_run, results_run


def getPool_mp(data_arrs, params, dT, draw, optimize=None, flip=False, fixedCircuit=None, start=None):
    """
    Parallel version of getPool:
    - Uses multiprocessing to parallelize runs.
    - Worker count: Linux=mp.cpu_count(), Windows=4.
    - Preserves return values, shapes, and side effects as in the original.
    """
    N, n, c, Δ, T, q = params  # keep unpack to match original expectations
    print(f"Pool with: T:{T}, dT:{dT}")
    T = float(T)
    q = int(q)

    circuit_pool, sign_pool = data_arrs[0]
    n_timesteps = round(T / dT)
    results = [[] for _ in range(n_timesteps)]
    bonds = [[] for _ in range(n_timesteps)]
    costs = [[] for _ in range(n_timesteps)]
    n_circuits = int(len(sign_pool) / n_timesteps)
    Ts = np.arange(0, T + dT, dT)  # preserved even if unused, to avoid behavior change

    indices = generate_random_indices(len(circuit_pool), n_circuits, n_timesteps)

    print(f"Running pool for {len(indices)} runs")

    # Decide worker count based on platform
    if sys.platform.startswith("win"):
        n_workers = 4
        ctx = mp.get_context("spawn")
    elif sys.platform.startswith("linux"):
        n_workers = mp.cpu_count()
        # 'fork' is efficient on Linux; fall back to default if unavailable
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()
    else:
        n_workers = mp.cpu_count()
        ctx = mp.get_context()  # sensible default for other OSes

    # Run all runs in parallel, preserving input order for deterministic aggregation
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_pool,
        initargs=(circuit_pool, sign_pool, q, flip, fixedCircuit, optimize, start),
        maxtasksperchild=None,
    ) as pool:
        for run_idx, (b_run, c_run, r_run) in enumerate(pool.imap(_single_run, indices, chunksize=1), start=1):
            # Preserve the original "Run: X" output (printed once per completed run)
            print(f"Run: {run_idx}")
            # Aggregate per-timestep scalars from this run
            for i in range(len(r_run)):
                bonds[i].append(b_run[i])
                costs[i].append(c_run[i])
                results[i].append(r_run[i])

    # Compute statistics per timestep (robust to occasional early breaks)
    # Using per-list means matches the original when all lists are equal length,
    # and is safer if any run exits early due to `start`.
    costs = np.array([np.mean(x) if len(x) > 0 else np.nan for x in costs])
    bonds = np.array([np.mean(x) if len(x) > 0 else np.nan for x in bonds])
    averages = np.array([np.mean(x) if len(x) > 0 else np.nan for x in results])
    stds = np.array([np.std(x) if len(x) > 0 else np.nan for x in results])

    # Insert t=0 values exactly as the original does
    if fixedCircuit is None:
        averages = np.insert(averages, 0, 1)
    else:
        averages = np.insert(averages, 0, measure(fixedCircuit.copy(), q, optimize))
    stds = np.insert(stds, 0, 0)

    # To preserve the original return shape/meaning of `quimb`, rebuild the
    # last run's circuit state deterministically in the parent.
    if fixedCircuit is None:
        quimb = getCircuit(q, flip=flip)
    else:
        quimb = fixedCircuit.copy()
    last_run_indices = indices[-1]
    for idx in last_run_indices:
        applyGates(quimb, circuit_pool[idx])
        _, cost = getComplexity(quimb)
        if start is not None and cost > start:
            break

    # Return values match the original function exactly
    if draw:
        return averages, stds, quimb, [bonds, costs]
    else:
        return averages, stds, quimb, [bonds, costs]

# --- Utility functions ---
def extract_dT_value(string):
    match = re.search(r'dT-([\d\.]+)', string)
    return float(match.group(1)) if match else None

def paulis_anticommute(p1, p2, overlap_count):
    # p1,p2 strings like "XX","Z"; 
    # they anticommute on each overlapping index if the chars differ
    # return True if an odd number of overlaps anticommute
    odd = 0
    for c1, c2 in zip(p1, p2):
        if c1 in 'XYZ' and c2 in 'XYZ' and c1 != c2:
            odd ^= 1
    return bool(odd)

def trotter_error_bound(hamil: Hamiltonian, r: int, T: float):
    # get the time-independent list of (Pauli, support, coeff)
    terms = hamil.get_term(0)
    S = 0.0
    for (p1, supp1, c1), (p2, supp2, c2) in combinations(terms, 2):
        if set(supp1) & set(supp2):
            # restrict to the overlapping positions
            # build local strings for only those positions:
            # here we assume terms are max‐length 2, so you can simplify
            if paulis_anticommute(p1, p2, len(set(supp1)&set(supp2))):
                S += 2 * abs(c1 * c2)
    print(S)
    return T**2 / (2 * r) * S

def trotterSimulation(hamil, N, n_snapshot, c, Δ_name, T, numQs):
    trotter = Trotter(hamil, N, n_snapshot, c, Δ_name, T, numQs)
    res, gates_arr = trotter.run()
    return res, gates_arr

def trotter(N, n_snapshot, T, q, compare, H_name, startTime=0, save=False, draw=False, flip=False, fixedCircuit=None, mps=True, circuitList = False, X=None):
    print(f"Running Trotter for N={N}, n_snapshot={n_snapshot}, T={T}, q={q}")
    circuit = None
    circuits = []
    dT = T / n_snapshot    
    times = np.linspace(startTime, startTime+float(T), int(N))
    Ts = np.linspace(startTime+dT, startTime+float(T), int(n_snapshot))
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    if H_name == "SCH":
        hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    if H_name == "NNN":
        hamil = Hamiltonian.next_nearest_neighbor_hamil(q, freqs)
    if H_name == "2D":
        hamil = Hamiltonian.lattice_2d_hamil(q, freqs=freqs)
    
    # Precompute all terms for each time step
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
        res = []#[1]
    else:
        res = [measure(fixedCircuit, q, False)]

    bonds = []
    lengths = []
    costs = []
    for i, gs in enumerate(gates):
        print(f"Snapshot {i+1} / {len(gates)}")

        # Fresh circuit each time
        if fixedCircuit == None:
            circuit = getCircuit(q, flip=flip, mps=mps, max_bond=X)
        else:
            circuit = fixedCircuit.copy()
        if draw:
            circuit = getCircuit(q, flip=flip, mps=False)

        # Apply gates up to current time
        for k in range(i + 1):
            applyGates(circuit, gates[k])

        # Measure immediately after applying
        bond, cost = getComplexity(circuit)
        result = measure(circuit)
        costs.append(cost)
        bonds.append(bond)
        res.append(result)
        lengths.append(circuit.num_gates)
        circuits.append(circuit.copy())

    lengths = [l*bonds[i]**3 for i,l in enumerate(lengths)]

    if save:
        if H_name == "SCH":
            save_path = os.path.join("TE-PAI-noSampling", "data", "plotting")
        if H_name == "NNN":
            save_path = os.path.join("TE-PAI-noSampling", "NNN_data", "plotting")
        if H_name == "2D":
            save_path = os.path.join("TE-PAI-noSampling", "2D_data", "plotting")
        os.makedirs(save_path, exist_ok=True)
        if fixedCircuit == None:
            if X is None:
                file_name = f"lie-N-{N}-T-{T}-q-{q}.csv"
            else:
                file_name = f"lie-N-{N}-T-{T}-q-{q}-X-{X}.csv"
        else:
            file_name = f"lie-N-{N}-T-{T+1000}-q-{q}-fixedCircuit.csv"
        file_path = os.path.join(save_path, file_name)
        
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y"])
            writer.writerows(zip(Ts, res))
        print(f"Lie data saved to {file_path}")

        if H_name == "SCH":
            save_path = os.path.join("TE-PAI-noSampling", "data", "plotting")
        if H_name == "NNN":
            save_path = os.path.join("TE-PAI-noSampling", "NNN_data", "plotting")
        if H_name == "2D":
            save_path = os.path.join("TE-PAI-noSampling", "2D_data", "plotting")
        os.makedirs(save_path, exist_ok=True)
        if X is None:
            file_name = f"lie-bond-N-{N}-T-{T}-q-{q}.csv"
        else:
            file_name = f"lie-bond-N-{N}-T-{T}-q-{q}-X-{X}.csv"
        file_path = os.path.join(save_path, file_name)
        
        #Ts = [t+float(T/n_snapshot) for t in Ts]
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y", "z", "l"])
            writer.writerows(zip(Ts, bonds, costs, lengths))
        print(f"Lie bonds saved to {file_path}")

    if not compare and not circuitList:
        return (Ts, res, [bonds, costs], circuit) 
    if not compare and circuitList:
        return (Ts, res, [bonds, costs], circuits)

def trotterComparison(N, n_snapshot, T, q):
    _, res10, _, _ = trotter(10, 10, T, q, False, flip=True)
    Ts, res, complexity, circuit = trotter(N, n_snapshot, T, q, False, flip=True)
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)

    # 1) Pauli matrices and tensor‐product helper
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    H1 = (X + Z)/np.sqrt(2)             # single‐qubit Hadamard
    paulis = {'X':X, 'Y':Y, 'Z':Z}

    def kron_n(ops):
        out = ops[0]
        for A in ops[1:]:
            out = np.kron(out, A)
        return out

    def term_op(gate, qubits, n):
        """Build the 2^n×2^n operator for gate on given qubits."""
        if gate in ("XX","YY","ZZ"):
            p = gate[0]
            ops = [paulis[p] if i in qubits else I for i in range(n)]
        else:
            ops = [paulis[gate] if i in qubits else I for i in range(n)]
        return kron_n(ops)

    # 2) Instantiate your Hamiltonian (time‐independent in practice)
    H     = Hamiltonian.spin_chain_hamil(q, freqs)

    # 3) Assemble the full H matrix at any t (all coeffs are constant)
    terms = H.get_term(0)   # [(gate, qubits, coef), ...]
    dim   = 2**q
    H_mat = np.zeros((dim,dim), dtype=complex)
    H_all  = kron_n([H1]*q)             # H⊗H⊗…⊗H on all qubits
    for gate, qubits, coef in terms:
        H_mat += coef * term_op(gate, qubits, q)

    # 4) Exponentiate and evolve

    P = []
    P10 = []
    indices = np.linspace(0, len(Ts) - 1, 11, dtype=int)
    Ts10 = [Ts[i] for i in indices]
    print(Ts10)
    print(Ts)

    for t in Ts:
        U   = expm(-1j * H_mat * t)   # exact time‐evolution operator
        psi0 = np.zeros(dim, dtype=complex)
        psi0[q//2] = 1.                   # example: start in |5⟩
        psi0 = H_all @ psi0
        psiT  = U @ psi0

        # 5) Measure X₀ and compute P(X₀=+1)
        X0    = term_op('X', [0], q)
        expX0 = np.real(np.vdot(psiT, X0 @ psiT))
        P_plus = (expX0 + 1)/2
        P.append(P_plus)
        if t in Ts10:
            P10.append(P_plus)


    error = np.array(P) - np.array(res)
    error10 = np.array(P10) - np.array(res10)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    # Top: Exact vs Trotterized
    ax1.plot(Ts, P,
            label='Exact exponentiated Hamiltonian',
            color="gray")

    # scatter with larger points
    ax1.scatter(Ts,  res,  label=f'Trotterization N = {N}',
                marker='o', s=30)    # <-- was s=16
    ax1.scatter(Ts10, res10, label='Trotterization N = 10',
                marker='o', s=30)    # <-- was s=16

    ax1.set_ylabel(r"$P(\langle X_0 \rangle = +1)$")
    ax1.set_title(r'Probability of measuring $\langle X_0 \rangle$ = +1')
    ax1.legend()

    # Bottom: Absolute error
    # use markersize (diameter) for the plot markers
    ax2.plot(Ts,   error,
            label=f'N = {N}, Abs. mean = {np.mean(np.abs(error)):.3f}',
            marker='o', markersize=6)   # <-- you can tweak this
    ax2.plot(Ts10, error10,
            label=f'N = 10, Abs. mean = {np.mean(np.abs(error10)):.2f}',
            marker='o', markersize=6)

    ax2.set_xlabel('Time T')
    ax2.set_ylabel('Error')
    ax2.set_title('Trotterization Error')
    ax2.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig("TE-PAI-noSampling/data/trotterComparison.png", dpi=300)

def trotterThenTEPAI(folder, trotterN, trottern, trotterT, saveAndPlot=False, optimize=False, flip=False, confirm=False,  base_dir='TE-PAI-noSampling/data/plotting', NNN=False):
    folder = strip_trailing_dot_zero(folder)
    data_dict = JSONtoDict(folder)
    data_arrs,Ts,params,pool = DictToArr(data_dict, True)
    N,n,c,Δ,T,q = params
    T = round(float(T), 8)

    # Running the trotter simulation up to a time trotterT
    # NB: T =/= trotterT
    trotterTs, res, complexity, circuit = trotter(N=trotterN, n_snapshot=trottern, T=trotterT, q=int(q), compare=False, save=True, draw=False, flip=flip, NNN=NNN)
    
    base_dir: str = 'TE-PAI-noSampling/data/plotting'
    if NNN:
        base_dir = 'TE-PAI-noSampling/NNN_data/plotting'
    save_trotter(trotterTs, complexity[0], complexity[1], trotterN, trottern, trotterT, q, base_dir=base_dir)

    if confirm:
        confirmTs, _, confirmComp, _ = trotter(N=trotterN, n_snapshot=10, T=T, startTime=trotterT, q=int(q), compare=False, save=True, draw=False, flip=flip, fixedCircuit=circuit, NNN=NNN)
        save_trotter(confirmTs, confirmComp[0], confirmComp[1], trotterN, 10, T, q, base_dir=base_dir)

    char = "c"
    dT = extract_dT_value(folder)
    char = "p"
    averages, stds, circuit, costs = getPool(data_arrs, params, dT, False, optimize, flip=flip, fixedCircuit = circuit)
    Ts = np.linspace(trotterT,trotterT + T,len(averages))

    pattern = r"dT-([0-9]+(?:\.[0-9]+)?)"
    match = re.search(pattern, folder)
    saveData(N,n,c,Δ,Ts,q,float(match.group(1)),averages,stds,char, NNN=NNN)

    filename = f"TEPAI-bonds-N-{N}-n-{n}-T-{T}-q-{q}.csv"
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'z'])
        for xi, yi, zi in zip(Ts, costs[0], costs[1]):
            writer.writerow([xi, yi, zi])

    if not NNN:
        organize_trotter_tepai()
    if NNN:
        organize_trotter_tepai(
            plotting_dir=Path("TE-PAI-noSampling/NNN_data/plotting"),
            target_base=Path("TE-PAI-noSampling/NNN_data/trotterThenTEPAI")
        )

def getTrotterPai(folder):
    pat_lie = re.compile(r"^lie-N-(?P<N>\d+)-T-(?P<T>[0-9.]+)-q-(?P<q>\d+)(?:-fixedCircuit)?\.csv$")
    pat_tep = re.compile(
        r"^N-(?P<N2>\d+)-n-(?P<n>\d+)-p-(?P<p>\d+)-Δ-pi_over_[^–]+"
        r"-T-(?P<T2>[0-9.]+)-q-(?P<q>\d+)-dT-(?P<dt>[0-9.]+)\.csv$"
    )
    # Match both trotter-bonds and TEPAI-bonds (type captured)
    pat_bonds = re.compile(
        r"^(?P<type>trotter|TE-?PAI)-bonds-N-(?P<N>\d+)-n-(?P<n>\d+)-T-(?P<T>\d+(?:\.\d+)?)-q-(?P<q>\d+)(?:-fixedCircuit)?\.csv$",
        re.IGNORECASE
    )
    
    # Containers for runs
    lie_runs = []
    tep_file = None
    bond_runs = []
    trotterLengths = []
    tepLengths = []

    # Scan directory for matching files
    folder_name = folder
    folder = Path(folder)
    for f in folder.iterdir():
        if not f.is_file():
            continue
        if (m := pat_lie.match(f.name)):
            gd = m.groupdict()
            lie_runs.append((f, float(gd["T"]), int(gd["N"])))
        elif pat_tep.match(f.name):
            tep_file = f
        elif (m := pat_bonds.match(f.name)):
            gd = m.groupdict()
            bond_runs.append((f, gd))

    if tep_file is None:
        raise FileNotFoundError("No TE-PAI file found in folder")
    if len(bond_runs) != 3:
        raise FileNotFoundError(f"Expected 3 bond CSV files (2 trotter + 1 TEPAI) in folder FOUND ONLY {len(bond_runs)}")

    # Load TE-PAI data
    x_t, y_t, e_t = [], [], []
    n_tep = 0
    q = int(tep_file.name.split("-")[11])
    Δ = int(tep_file.name.split("-")[7].split("_")[2])
    p = int(tep_file.name.split("-")[5])
    Δ = np.pi / Δ
    T1= float(folder_name.split("-")[10])
    N1= float(folder_name.split("-")[8])
    T2= float(folder_name.split("-")[16])
    N2= float(folder_name.split("-")[12])
    with tep_file.open() as fp:
        next(fp)
        for line in fp:
            n_tep += 1
            xi, yi, erri = line.strip().split(',')
            x_t.append(float(xi)); y_t.append(float(yi)); e_t.append(float(erri))

    dt2 = (T2 - T1) / n_tep
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)

    NNN = "NNN_" in folder_name
    if not NNN:
        hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    if NNN:
        hamil = Hamiltonian.next_nearest_neighbor_hamil(q, freqs)
    te_pai = TE_PAI(hamil, q, Δ, dt2, 1000, n_tep)
    tep_len = te_pai.expected_num_gates

    return lie_runs, bond_runs, x_t, y_t, e_t, q, N1, N2, p, T1, dt2, tep_len