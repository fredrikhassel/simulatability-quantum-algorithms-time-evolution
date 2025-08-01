from collections import defaultdict
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
from main import TE_PAI
import h5py
import time
import opt_einsum as oe
import cotengra as ctg
from qiskit import QuantumCircuit, transpile
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from scipy.linalg import expm
from collections import Counter
from scipy.stats import norm
import ast
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from itertools import combinations

plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,
        'lines.linewidth': 2,
        'lines.markersize': 5,
        'errorbar.capsize': 3,
        'savefig.dpi': 300,
        'figure.autolayout': True,
    })

tab_colors = {
    "red":    mcolors.TABLEAU_COLORS["tab:red"],
    "green":  mcolors.TABLEAU_COLORS["tab:green"],
    "blue":   mcolors.TABLEAU_COLORS["tab:blue"],
    "purple":   mcolors.TABLEAU_COLORS["tab:purple"],
    "cyan":   mcolors.TABLEAU_COLORS["tab:cyan"],
    "orange": mcolors.TABLEAU_COLORS["tab:orange"]
}

gate_colors = {
    "h":   (tab_colors["red"],   "white"),
    "z":   (tab_colors["green"], "white"),
    "rxx": (tab_colors["blue"],  "white"),
    "ryy": (tab_colors["purple"],  "white"),
    "rzz": (tab_colors["cyan"],  "white"),
    "zz":  (tab_colors["blue"],  "white"),
    "rz":  (tab_colors["orange"], "black"),
}

def JSONtoDict(folder_name):
    try:
        files = os.listdir(folder_name)
    except FileNotFoundError:
        # fallback from “folder.0” → “folder”
        if folder_name.endswith('.0'):
            fallback_name = folder_name[:-2]
            try:
                files = os.listdir(fallback_name)
                folder_name = fallback_name
            except FileNotFoundError:
                raise FileNotFoundError(f"Neither '{folder_name}' nor '{fallback_name}' could be found.")
        else:
            try:
                files = os.listdir(folder_name+".0")
                folder_name = folder_name+".0"
            except FileNotFoundError:
                raise FileNotFoundError(f"'{folder_name}' could not be found.")

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

def measure(circuit,q=0, optimize=False):
    val = np.real(circuit.local_expectation(qu.pauli('X'), (0)))
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

def generate_random_indices(pool_size, output_length, entry_length):
    rng = np.random.default_rng(0)  # Create RNG instance with optional seed
    return [list(rng.integers(0, pool_size, size=entry_length)) for _ in range(output_length)]

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

def plot_data_from_folder(folderpath, ax=None, trotBounds=None):
    quimb_pattern = re.compile(r'N-(\d+)-n-(\d+)-([cp])-(\d+)-Δ-(\w+)-T-([\d\.]+)-q-(\d+)-dT-([\d\.]+)\.csv')
    lie_pattern = re.compile(r'lie-N-(\d+)-T-((?:\d+\.\d+)|(?:\d+))-q-(\d+)\.csv')
    
    quimb_data = []
    lie_data = []
    
    for filename in os.listdir(folderpath):
        print(filename)
        filepath = os.path.join(folderpath, filename)
        
        quimb_match = quimb_pattern.match(filename)
        lie_match = lie_pattern.match(filename)
        
        if lie_match:
            df = pd.read_csv(filepath)
            if df.shape[1] >= 2:
                label = f"N = {lie_match.group(1)}"
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
                    delta = quimb_match.group(5)
                    denominator = delta[len("pi_over_"):]
                    power = int(denominator).bit_length() - 1
                    delta = r"\frac{{\pi}}{{2^{{{}}}}}".format(power)
                    label = f"Pool size = {quimb_match.group(4)}, Δ = ${delta}$, dT = {quimb_match.group(8)}"
                quimb_data.append((df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], label))
        

    # prepare axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
        created_fig = True
    
    # plot quimb error-bars
    for x, y, err, label in quimb_data:
        x = x.to_list()
        y = y.to_list()
        err = err.to_list()
        x.insert(0,0)
        y.insert(0,1)
        err.insert(0,0)

        ax.errorbar(
            x, y, yerr=err,
            label=f'TE-PAI ({label})',
            capsize=3, elinewidth=1.5,
            marker='o', markersize=5, color="tab:green"
        )
    
    # plot lie data
    colors = ["gray", "black"]
    linestyles = ["--", "-"]

    if(len(lie_data)==1):
        x, y, label = lie_data[0]
        x = x.to_list()
        y = y.to_list()
        ax.plot(
            x, y,
            label=f'Trotterization ({label})',
            linewidth=2,
            color="black",
        )

    else:
        for i,(x, y, label) in enumerate(lie_data):
            ax.plot(
                x, y,
                label=f'Trotterization ({label})',
                linewidth=2,
                color=colors[i%2],
                linestyle=linestyles[i%2]
            )
    
    ax.grid(True, which='both')
    ax.legend(loc='upper right')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\langle X_0 \rangle$')
    if created_fig:
        plt.tight_layout()
        plt.show()
    
    return ax

def plot_data_two_folders(folder1, folder2, q, N):
    """
    Plot the contents of two folders side-by-side for direct comparison,
    and add rising Trotter‐error bars to the “Trotterization” curve in the second plot.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # first plot unchanged
    plot_data_from_folder(folder1, ax=axes[0])
    axes[0].text(
        0.05, 0.65, os.path.basename(folder1),
        transform=axes[0].transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

    # build your spin-chain Hamiltonian
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    # call l1_norm with T=1 to get ||H||_1, then square for C
    C_bound = hamil.l1_norm(1)**2

    # second plot – draw the curves
    plot_data_from_folder(folder2, ax=axes[1])
    axes[1].text(
        0.05, 0.65, os.path.basename(folder2),
        transform=axes[1].transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

    ylims = axes[1].get_ylim()
    # find and decorate the Trotter line
    for line in axes[1].get_lines():
        if line.get_label().startswith("Trotterization"):
            x = line.get_xdata()
            y = line.get_ydata()
            

            trotter_err = (x**2) / (2 * N)#trotter_error_bound(hamil, N, x) #(x**2) / (2 * N) #* C_bound
            axes[1].errorbar(
                x, y,
                yerr=trotter_err,
                fmt='none',
                ecolor='gray',
                alpha=0.7,
                label="Trotter error bound",
                zorder=0
            )
            break
    # Restore y-limits so errorbars don't expand the axis
    
    #axes[1].set_ylim(ylims)
    axes[1].legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("firstTEPAI")
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

def trotter(N, n_snapshot, T, q, compare, startTime=0, save=False, draw=False, flip=False, fixedCircuit=None, mps=True, circuitList = False):
    print(f"Running Trotter for N={N}, n_snapshot={n_snapshot}, T={T}, q={q}")
    circuit = None
    circuits = []
    dT = T / n_snapshot    
    times = np.linspace(startTime, startTime+float(T), int(N))
    Ts = np.linspace(startTime+dT, startTime+float(T), int(n_snapshot))
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
            circuit = getCircuit(q, flip=flip, mps=mps)
        else:
            circuit = fixedCircuit.copy()
        if draw:
            circuit = getCircuit(q, flip=flip, mps=False)

        # Apply gates up to current time
        for k in range(i + 1):
            applyGates(circuit, gates[k])

        if draw and i == 0:
            print(gates[0])
            print(f"length of gates: {len(gates[0])}")
            tags   = ['PSI0', 'Z', 'RZ', 'RZZ', 'RXX', 'RYY', 'H']
            colors = [
                gate_colors[tag.lower()][0]
                if tag.lower() in gate_colors else "#888888"
                for tag in tags
            ]
            fig1, ax1 = plt.subplots()
            circuit.psi.draw(color=tags,
            custom_colors=colors, layout="kamada_kawai", ax=ax1)
            ax1.axis('off')
            #fig1.savefig("trotterEx", dpi=300, bbox_inches="tight")
            qiskit = quimb_to_qiskit(circuit)
            style = {
                "displaycolor": gate_colors,
                "textcolor":      "#222222",    # global text
                "gatetextcolor":  "#FFFFFF",    # fallback gate label color
                "linecolor":      "#666666",    # qubit wire
                "creglinecolor":  "#999999",    # classical wire
                "backgroundcolor": "#FFFFFF",   # canvas
            }
            qiskit.draw("mpl", scale=1, style=style)
            #plt.savefig("trotterExQiskit", dpi=300, bbox_inches="tight")
            #return

        # Measure immediately after applying
        bond, cost = getComplexity(circuit)
        result = measure(circuit, q, False)
        costs.append(cost)
        bonds.append(bond)
        res.append(result)
        lengths.append(circuit.num_gates)
        circuits.append(circuit.copy())

    lengths = [l*bonds[i]**3 for i,l in enumerate(lengths)]

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

        save_path = os.path.join("TE-PAI-noSampling", "data", "plotting")
        os.makedirs(save_path, exist_ok=True)
        file_name = f"lie-bond-N-{N}-T-{T}-q-{q}.csv"
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

    if compare:
        checkEqual(gates, gatesSim)
        resSim, gatesSim = trotterSimulation(hamil, N, n_snapshot, 1, 1, T, q)
        print(resSim)
        print(res)
        plt.scatter(range(len(resSim)), resSim, color='blue', label='Qiskit')
        plt.scatter(range(len(res)), res, color='red', label='Quimb')
        plt.legend()
        plt.show()

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
        averages, stds, circuit, costs = getPool(data_arrs, params,dT, draw, optimize, flip=flip)
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
    saveData(N,n,c,Δ,Ts,q,float(match.group(1)),averages,stds,char)
    if saveAndPlot:
        trotter(100,10,float(T),int(q),compare=False,save=True, flip=flip)
        plot_data_from_folder("TE-PAI-noSampling/data/plotting")

    if costs[0][0] != None:
        return costs

def save_trotter(x, y, z, N, n, T, q, base_dir='TE-PAI-noSampling/data/plotting', fixed=False):
    # Ensure target directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Build filename
    filename = f"trotter-bonds-N-{N}-n-{n}-T-{T}-q-{q}.csv"
    if fixed:
        filename = f"trotter-bonds-N-{N}-n-{n}-T-{T}-q-{q}-fixedCircuit.csv"
    filepath = os.path.join(base_dir, filename)
    
    # Write CSV
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'z'])
        for xi, yi, zi in zip(x, y, z):
            writer.writerow([xi, yi, zi])

def mainCalc(tepaiPath, finalT1, N1, n1, N2, finalT2, confirm=False, flip=True):
    params = parse_path(tepaiPath)
    q = params['q']
    dT = params['dT']
    print(params['Δ'])
    pattern = r'^pi_over_(\d+(?:\.\d+)?)$'
    m = re.match(pattern, params['Δ'])
    divisor = float(m.group(1))
    Δ = np.pi / divisor

    # Performing main trotterization
    ts1, res1, comp1, circ1 = trotter(N=N1, n_snapshot=n1, T=finalT1, q=int(q), compare=False, save=True, draw=False, flip=flip)
    circuit = circ1.copy()
    save_trotter(ts1, comp1[0], comp1[1], N1, n1, finalT1, q)

    # Performing continuing trotterization
    if confirm:
            ts2, res2, comp2, circ2 = trotter(N=N2, n_snapshot=10, T=finalT2, startTime=finalT1, q=int(q), compare=False, save=True, draw=False, flip=flip, fixedCircuit=circuit)
            save_trotter(ts2, comp2[0], comp2[1], N2, 10, finalT2, q, fixed=True)

    # Performing TE-PAI
    tepaiPath = strip_trailing_dot_zero(tepaiPath)
    data_dict = JSONtoDict(tepaiPath)
    data_arrs,Ts,params2,pool = DictToArr(data_dict, True)
    circuit = circ1.copy()
    averages, stds, circuit, costs = getPool(data_arrs, params2, dT, False, False, flip=flip, fixedCircuit = circuit)
    Ts = np.linspace(finalT1,finalT1 + finalT2,len(averages))
    saveData(params['N'],params['n'],params['p'],params['Δ'],Ts,q,dT,averages,stds,"p")

    # Saving TEPAI costs
    base_dir='TE-PAI-noSampling/data/plotting'
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
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
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

    organize_trotter_tepai()

def mainCalc2(tepaiPath, finalT1, N1, n1, finalT2, confirm=False, flip=True):
    params = parse_path(tepaiPath)
    q = params['q']
    dT = params['dT']
    print(params['Δ'])
    pattern = r'^pi_over_(\d+(?:\.\d+)?)$'
    m = re.match(pattern, params['Δ'])
    divisor = float(m.group(1))
    Δ = np.pi / divisor

    # Performing main trotterization
    ts1, res1, comp1, circuits = trotter(N=N1, n_snapshot=n1, T=finalT2, q=int(q), compare=False, save=True, draw=False, flip=flip, circuitList=True)
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
    save_trotter(ts1, comp1[0], comp1[1], N1, n1, finalT2, q)

    # Performing TE-PAI
    tepaiPath = strip_trailing_dot_zero(tepaiPath)
    data_dict = JSONtoDict(tepaiPath)
    data_arrs,Ts,params2,pool = DictToArr(data_dict, True)
    circuit = circuit.copy()
    averages, stds, circuit, costs = getPool(data_arrs, params2, dT, False, False, flip=flip, fixedCircuit = circuit)
    Ts = np.linspace(finalT1,finalT2, len(averages))

    saveData(params['N'],params['n'],params['p'],params['Δ'],Ts,q,dT,averages,stds,"p")
    circ2 = circuits[-1].copy()

    # Saving TEPAI costs
    base_dir='TE-PAI-noSampling/data/plotting'
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
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
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

    organize_trotter_tepai()

def fullCalc(tepaiPath, T, N, n, flip=True):
    params = parse_path(tepaiPath)
    q = params['q']
    dT = params['dT']
    print(params['Δ'])
    pattern = r'^pi_over_(\d+(?:\.\d+)?)$'
    m = re.match(pattern, params['Δ'])
    divisor = float(m.group(1))
    Δ = np.pi / divisor

    # Performing main trotterization
    ts1, res1, comp1, circuits = trotter(N=N, n_snapshot=n, T=T, q=int(q), compare=False, save=True, draw=False, flip=flip, circuitList=True)
    save_trotter(ts1, comp1[0], comp1[1], N, n, T, q)

    # Performing TE-PAI
    tepaiPath = strip_trailing_dot_zero(tepaiPath)
    data_dict = JSONtoDict(tepaiPath)
    data_arrs,Ts,params2,pool = DictToArr(data_dict, True)
    N2,n1,c1,Δ1,_,q1 = params2
    params2 = N2,n1,c1,Δ1,T,q1
    averages, stds, circuit, costs = getPool(data_arrs, params2, dT, False, False, flip=flip)
    circ1 = circuit.copy()
    Ts = np.linspace(0,T,len(averages))

    saveData(params['N'],params['n'],params['p'],params['Δ'],Ts,q,dT,averages,stds,"p")
    circ2 = circuits[-1].copy()

    # Saving TEPAI costs
    base_dir='TE-PAI-noSampling/data/plotting'
    filename = f"TEPAI-bonds-N-{params['N']}-n-{params['n']}-T-{params['T']}-q-{q}.csv"
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'z'])
        for xi, yi, zi in zip(Ts, costs[0], costs[1]):
            writer.writerow([xi, yi, zi])

    # Saving circuit lengths
    n2 = int(T / params['dT'])
    len1 = len(circ1.gates)
    len2 = len(circ2.gates)
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    te_pai = TE_PAI(hamil, q, Δ, params['dT'], 1000, 1)
    lentep = te_pai.expected_num_gates
    lengths1 = [(len1/n)*i for i in range(1, n+1)]
    lengths2 = [(len2/n)*i for i in range(1, n+1)]
    lengthstep = [(lentep/n2)*i for i in range(1, n2+1)]
    filename = f"lengths.csv"
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['trotter1', 'trotter2', 'TEPAI'])
        writer.writerow([lengths1, lengths2, lengthstep])

    organize_trotter_tepai(target_base = Path("TE-PAI-noSampling/data/fullCalc"))

def plotMainCalc2(folder, both=True, justLengths=False):
    trotsim  = [[], []]; paisim  = [[], [], []]
    trotbond = [[], []]; paibond = [[], []]
    trotcost = [[], []]; paicost = [[], []]
    
    params = folder.split("-")
    folder = Path(folder)
    for file in folder.iterdir():
        match file.name:
            case "lengths.csv":
                with file.open() as fp:
                    next(fp)
                    line = next(fp)
                    reader = csv.reader([line.strip()])
                    row = next(reader)
                    trotterLengths = ast.literal_eval(row[0])
                    tePAILengths   = ast.literal_eval(row[2])

            case name if name.startswith("lie"):
                second = name.endswith("fixedCircuit.csv")
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi = map(float, line.split(','))
                        trotsim[0].append(xi); trotsim[1].append(yi)

            case name if name.startswith("N"):
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        paisim[0].append(xi); paisim[1].append(yi); paisim[2].append(zi)

            case name if name.startswith("trotter-bonds"):
                second = name.endswith("fixedCircuit.csv")
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        trotbond[0].append(xi)
                        trotbond[1].append(yi)
                        trotcost[0].append(xi)
                        trotcost[1].append(zi)

            case name if name.startswith("TEPAI-bonds"):
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        paibond[0].append(xi)
                        paibond[1].append(yi)
                        paicost[0].append(xi)
                        paicost[1].append(zi)


    index = -1
    for i,t in enumerate(trotsim[0]):
        if t == paisim[0][0]:
            index = i
    if index == -1:
        print("Error: time pai start time {paisim[0][0]} not found in trotter times {trotsim[0]}")
        return

    tePAILengths = [t+trotterLengths[index] for t in tePAILengths]

    if paibond[1][0] != 0:
        paibond[0] = [p+(paibond[0][1]-paibond[0][0]) for p in paibond[0]]
        paibond[0].insert(0,trotbond[0][index])
        paibond[1].insert(0,trotbond[1][index])
    if trotbond[1][0] != 0:
        trotbond[1].insert(0,0)
        trotbond[0].insert(0,0)
    if tePAILengths[0] != 0:
        tePAILengths.insert(0,trotterLengths[index])
    if trotterLengths[0] != 0:
        trotterLengths.insert(0,0)

    paicost[0] = [p + (paicost[0][1]-paicost[0][0]) for p in paicost[0]]
    paicost[0].insert(0,trotcost[0][index])
    trotcost[0].insert(0,0)


    trotcost[1] = np.array(trotterLengths) * np.array(trotbond[1])**3
    if not justLengths:
        paicost[1]   = np.array(tePAILengths) * np.array(paibond[1])**3
    if justLengths:
        paicost[1]   = np.array(tePAILengths) * np.array(trotbond[1][index+1:])**3

    title = f"q={params[6]} | Δ={params[3]}-{params[4]} | N={params[8]} | p={params[14]}" 

    threshold  = paicost[1].max()
    print(threshold)
    print(trotcost[1])
    cutoff_idx = np.argmax(trotcost[1] > threshold)
    cutoff_time = trotcost[0][cutoff_idx]


    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    fig.suptitle(title)

    # 1) Full sim
    ax0 = axes[0]
    ax0.plot(trotsim[0], trotsim[1], color='black', label='Init Trotter')
    ax0.errorbar(paisim[0], paisim[1], yerr=paisim[2],
                 color='tab:green', label='TE-PAI')
    ax0.set_ylabel('Observable')
    ax0.set_title('1) Full Simulation')
    ax0.legend(); ax0.grid(True)

    # 2) Zoom only on TE-PAI window (masking)
    te_start = paisim[0][0]
    te_end = paisim[0][-1]
    ax1 = axes[1]
    t1, y1 = np.array(trotsim[0]), np.array(trotsim[1])
    tp, yp, ep = np.array(paisim[0]), np.array(paisim[1]), np.array(paisim[2])
    m1 = (t1 >= te_start) & (t1 <= te_end)
    mp = (tp >= te_start) & (tp <= te_end)
    ax1.plot(t1[m1], y1[m1], color='black', label='Init Trotter')
    ax1.errorbar(tp[mp], yp[mp], yerr=ep[mp], color='tab:green', label='TE-PAI')
    ax1.set_xlabel('Time')
    ax1.set_title('2) Zoom on TE-PAI Region')
    ax1.legend(); ax1.grid(True)

    # 3) Cutoff & improvement regions
    ax2 = axes[2]
    trotsim[0] = np.array(trotsim[0])
    trotsim[1] = np.array(trotsim[1])
    ax2.plot(trotsim[0][trotsim[0] <= cutoff_time],
             trotsim[1][trotsim[0] <= cutoff_time],
             color='black', label='Trotter (cutoff)')
    ax2.axvline(cutoff_time, linestyle='--', color='black')
    ax2.errorbar(paisim[0], paisim[1], yerr=paisim[2],
                 color='tab:green', label='TE-PAI')
    ax2.axvline(te_end, linestyle='--', color='tab:green')
    ax2.axvspan(te_start, cutoff_time, color='gray', alpha=0.2)
    ax2.axvspan(cutoff_time, te_end, label='TE-PAI advantage',     color='tab:green', alpha=0.2)
    ax2.set_title('3) Cutoff & Improvement') 
    ax2.legend(); ax2.grid(True)

    # 4) Bond sizes
    ax3 = axes[3]
    ax3.plot(trotbond[0], trotbond[1], color='black', label='Init Trotter')
    ax3.plot(paibond[0],  paibond[1],   color='tab:green', label='TE-PAI')
    ax3.set_xlabel('Time'); ax3.set_ylabel('Max bond')
    ax3.set_title('4) Bond Dimension')
    ax3.legend(); ax3.grid(True)

    # 5) Gate counts
    ax4 = axes[4]
    adj_te  = tePAILengths.copy()

    index = -1
    for i,t in enumerate(trotsim[0]):
        if t == paisim[0][0]:
            index = i
    if index == -1:
        print("Error: time pai start time {paisim[0][0]} not found in trotter times {trotsim[0]}")
        return
            
    len_last = trotterLengths[index]
    ax4.plot(trotbond[0], trotterLengths, color='black', label='Init Trotter')
    ax4.plot(paisim[0],   tePAILengths,   color='tab:green', label='TE-PAI')
    #ax4.plot(times,       adj_te,         color="red", label="others")
    ax4.set_xlabel('Time'); ax4.set_ylabel('# gates')
    ax4.set_title('5) Gate Count')
    ax4.legend(); ax4.grid(True)

    # 6) Calculation cost
    #for change,cost in enumerate(paicost[1]):
    #    if cost in trotcost[1]:
    #        break

    ax5 = axes[5]
    ax5.plot(trotcost[0], trotcost[1], color='black', label='Init Trotter')
    ax5.plot(paicost[0],  paicost[1],    color='tab:green', label='TE-PAI')
    ax5.set_xlabel('Time'); ax5.set_ylabel('Cost')
    if not justLengths:
        ax5.set_title('6) Compute cost for individual bond sizes')
    else:
        ax5.set_title('6) Compute cost for shared bond size')
    ax5.legend(); ax5.grid(True)

    plt.tight_layout()
    plt.show()

def plotMainCalcOld(folder, both=True, justLengths=False):
    # Load simulation and bond data
    trotsim, paisim = [[], []], [[], [], []]
    trotbond, paibond = [[], []], [[], []]
    trotcost, paicost = [[], []], [[], []]
    params = folder.split("-")
    folder = Path(folder)
    for file in folder.iterdir():
        name = file.name
        if name == "lengths.csv":
            with file.open() as fp:
                next(fp)
                lengths_line = next(fp).strip()
                row = next(csv.reader([lengths_line]))
                trotterLengths = ast.literal_eval(row[0])
                tePAILengths = ast.literal_eval(row[2])
        elif name.startswith("lie"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x,y = map(float, line.split(',') )
                    trotsim[0].append(x); trotsim[1].append(y)
        elif name.startswith("N"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x,y,z = map(float, line.split(','))
                    paisim[0].append(x); paisim[1].append(y); paisim[2].append(z)
        elif name.startswith("trotter-bonds"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x,y,z = map(float, line.split(','))
                    trotbond[0].append(x); trotbond[1].append(y)
                    trotcost[0].append(x); trotcost[1].append(z)
        elif name.startswith("TEPAI-bonds"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x,y,z = map(float, line.split(','))
                    paibond[0].append(x); paibond[1].append(y)
                    paicost[0].append(x); paicost[1].append(z)

    # Align start of TE-PAI simulation
    idx = next((i for i,t in enumerate(trotsim[0]) if t == paisim[0][0]), -1)
    if idx < 0:
        raise ValueError("Cannot align TE-PAI start time.")
    tePAILengths = [t + trotterLengths[idx] for t in tePAILengths]

    # Insert initial points for alignment
    if paibond[1] and paibond[1][0] != 0:
        shift = paibond[0][1] - paibond[0][0]
        paibond[0] = [p + shift for p in paibond[0]]
        paibond[0].insert(0, trotbond[0][idx]); paibond[1].insert(0, trotbond[1][idx])
    if trotbond[1] and trotbond[1][0] != 0:
        trotbond[0].insert(0, 0); trotbond[1].insert(0, 0)
    if tePAILengths and tePAILengths[0] != 0:
        tePAILengths.insert(0, trotterLengths[idx])
    if trotterLengths and trotterLengths[0] != 0:
        trotterLengths.insert(0, 0)
    # Align cost time arrays
    shift_cost = paicost[0][1] - paicost[0][0]
    paicost[0] = [p + shift_cost for p in paicost[0]]
    paicost[0].insert(0, trotcost[0][idx])
    trotcost[0].insert(0, 0)

    # Compute costs
    trotcost[1] = np.array(trotterLengths) * (np.array(trotbond[1])**3)
    if not justLengths:
        paicost[1] = np.array(tePAILengths) * (np.array(paibond[1])**3)
    else:
        paicost[1] = np.array(tePAILengths) * (np.array(trotbond[1][idx+1:])**3)

    # Compute cutoff time
    threshold = max(paicost[1])
    cutoff_idx = int(np.argmax(np.array(trotcost[1]) > threshold))
    cutoff_time = trotcost[0][cutoff_idx]
    te_start, te_end = paisim[0][0], paisim[0][-1]

    # Convert simulation arrays to numpy
    t_arr = np.array(trotsim[0]); y_arr = np.array(trotsim[1])
    mask_zoom = (t_arr >= te_start) & (t_arr <= te_end)

    # --- Figure 1: Cutoff & Improvement with TE-PAI zoom inset ---
    fig1, ax = plt.subplots(figsize=(8,6))
    fig1.suptitle(f"q={params[6]} | Δ={params[3]}-{params[4]} | N={params[8]} | p={params[14]}")
    mask_cut = t_arr <= cutoff_time
    ax.plot(t_arr[mask_cut], y_arr[mask_cut], color='black', label='Trotterization (cutoff)')
    ax.axvline(cutoff_time, linestyle='--', color='black')
    ax.errorbar(paisim[0], paisim[1], yerr=paisim[2], color='tab:green', label='TE-PAI')
    ax.axvline(te_end, linestyle='--', color='tab:green')
    ax.axvspan(te_start, cutoff_time, color='gray', alpha=0.2)
    ax.axvspan(cutoff_time, te_end, color='tab:green', alpha=0.2, label='TE-PAI advantage')
    y_min_zoom = min(np.min(y_arr[mask_zoom]), np.min(paisim[1]))
    y_max_zoom = max(np.max(y_arr[mask_zoom]), np.max(paisim[1]))
    rect = Rectangle((te_start, y_min_zoom), te_end-te_start, y_max_zoom-y_min_zoom,
                     linestyle='--', edgecolor='black', fill=False)
    ax.add_patch(rect)
    ax.set_xlabel('Time'); ax.set_ylabel(r'$\langle X_0 \rangle$')
    ax.set_title('Cutoff & Improvement with Zoom')
    legend = ax.legend(loc='upper right', framealpha=1)
    legend.get_frame().set_facecolor('white')
    ax.grid(True)
    inset = inset_axes(ax, width='50%', height='40%', loc='lower left')
    inset.plot(t_arr[mask_zoom], y_arr[mask_zoom], color='gray', linestyle='--', label='Trotterization (continued)')
    inset.errorbar(paisim[0], paisim[1], yerr=paisim[2], fmt='o', color='tab:green', label='_nolegend_')
    inset.set_xlim(te_start, te_end)
    inset.set_xticks([]); inset.set_yticks([])
    #inset.text(0.7, 0.4, 'Zoomed in', transform=inset.transAxes,
    #           ha='center', va='bottom', fontsize='small', weight='bold')
    inset_legend = inset.legend(loc='upper right', framealpha=1)
    inset_legend.get_frame().set_facecolor('white')
    inset.grid(True)
    plt.savefig("cutoffImprovement")
    #plt.tight_layout()

    # --- Figure 2: Bond, Gate Count, Compute Cost ---
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax3, ax4, ax5 = axes

    # 1) Bond Dimension
    ax3.plot(trotbond[0], trotbond[1], color='black', label='Trotterization')
    ax3.plot(paibond[0], paibond[1], color='tab:green', label='TE-PAI')
    max_te_pa = max(paibond[1])
    top_y = ax3.get_ylim()[1]
    mask = np.array(trotbond[1]) > max_te_pa
    if mask.any():
        x_min = np.array(trotbond[0])[mask].min()
        x_max = np.array(trotbond[0])[mask].max()
        ax3.add_patch(Rectangle((x_min, max_te_pa), x_max - x_min, top_y - max_te_pa,
                                 color='gray', alpha=0.3, label='Additional cost'))
    ax3.set_ylabel('Dimension')
    ax3.set_title(r"$\mathbf{A}$: Largest bond dimension")
    ax3.grid(True)
    ax3.legend(loc='upper left', framealpha=1)

    # 2) Gate Count
    ax4.plot(trotbond[0], trotterLengths, color='black')
    ax4.plot(paisim[0], tePAILengths, color='tab:green')
    max_te_len = max(tePAILengths)
    top_y = ax4.get_ylim()[1]
    mask = np.array(trotterLengths) > max_te_len
    if mask.any():
        x_min = np.array(trotbond[0])[mask].min()
        x_max = np.array(trotbond[0])[mask].max()
        ax4.add_patch(Rectangle((x_min, max_te_len), x_max - x_min, top_y - max_te_len,
                                 color='gray', alpha=0.3, label='_nolegend_'))
    ax4.set_ylabel('Gates')
    ax4.set_title(r"$\mathbf{B}$: Circuit gate count")
    ax4.grid(True)

    # 3) Compute Cost
    ax5.plot(trotcost[0], trotcost[1], color='black')
    ax5.plot(paicost[0], paicost[1], color='tab:green')
    max_te_cost = max(paicost[1])
    top_y = ax5.get_ylim()[1]
    mask = np.array(trotcost[1]) > max_te_cost
    if mask.any():
        x_min = np.array(trotcost[0])[mask].min()
        x_max = np.array(trotcost[0])[mask].max()
        ax5.add_patch(Rectangle((x_min, max_te_cost), x_max - x_min, top_y - max_te_cost,
                                 color='gray', alpha=0.3, label='_nolegend_'))
    ax5.set_ylabel('Cost ')
    ax5.set_title(r"$\mathbf{C}$: Computational depth")
    ax5.set_xlabel('Time')
    ax5.grid(True)

    for ax in (ax3, ax4, ax5):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 4))
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlabel('Time')

    plt.tight_layout()
    plt.savefig("Costs")
    #plt.show()

def plotMainCalc3(folder, both=True, justLengths=False, aligned=False):
    # Load simulation and bond data
    trotsim, paisim = [[], []], [[], [], []]
    trotbond, paibond = [[], []], [[], []]
    trotcost, paicost = [[], []], [[], []]
    params = folder.split("-")
    folder = Path(folder)
    for file in folder.iterdir():
        name = file.name
        if name == "lengths.csv":
            with file.open() as fp:
                next(fp)
                lengths_line = next(fp).strip()
                row = next(csv.reader([lengths_line]))
                trotterLengths = ast.literal_eval(row[0])
                tePAILengths = ast.literal_eval(row[2])
        elif name.startswith("lie"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x, y = map(float, line.split(','))
                    trotsim[0].append(x);
                    trotsim[1].append(y)
        elif name.startswith("N"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x, y, z = map(float, line.split(','))
                    paisim[0].append(x);
                    paisim[1].append(y);
                    paisim[2].append(z)
        elif name.startswith("trotter-bonds"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x, y, z = map(float, line.split(','))
                    trotbond[0].append(x);
                    trotbond[1].append(y)
                    trotcost[0].append(x);
                    trotcost[1].append(z)
        elif name.startswith("TEPAI-bonds"):
            with file.open() as fp:
                next(fp)
                for line in fp:
                    x, y, z = map(float, line.split(','))
                    paibond[0].append(x);
                    paibond[1].append(y)
                    paicost[0].append(x);
                    paicost[1].append(z)

    # Alignment block: only apply when aligned=False
    if not aligned:
        # Align start of TE-PAI simulation
        idx = next((i for i, t in enumerate(trotsim[0]) if t == paisim[0][0]), -1)
        if idx < 0:
            raise ValueError("Cannot align TE-PAI start time.")
        tePAILengths = [t + trotterLengths[idx] for t in tePAILengths]
        
        # Insert initial points for alignment
        if paibond[1] and paibond[1][0] != 0:
            shift = paibond[0][1] - paibond[0][0]
            paibond[0] = [p + shift for p in paibond[0]]
            paibond[0].insert(0, trotbond[0][idx]); paibond[1].insert(0, trotbond[1][idx])
        if trotbond[1] and trotbond[1][0] != 0:
            trotbond[0].insert(0, 0); trotbond[1].insert(0, 0)
        if tePAILengths and tePAILengths[0] != 0:
            tePAILengths.insert(0, trotterLengths[idx])
        if trotterLengths and trotterLengths[0] != 0:
            trotterLengths.insert(0, 0)

        # Align cost time arrays
        shift_cost = paicost[0][1] - paicost[0][0]
        paicost[0] = [p + shift_cost for p in paicost[0]]
        paicost[0].insert(0, trotcost[0][idx])
        trotcost[0].insert(0, 0)

    else:
        shift = paibond[0][1] - paibond[0][0]
        paibond[0] = [p + shift for p in paibond[0]]
        shift_cost = paicost[0][1] - paicost[0][0]
        paicost[0] = [p + shift_cost for p in paicost[0]]
        tePAILengths.insert(0,0)
        paibond[1].insert(0,0)
        paibond[0].insert(0,0)
        paicost[0].insert(0,0)
        trotterLengths.insert(0,0)
        trotcost[0].insert(0,0)
        trotcost[1].insert(0,0)
        trotbond[0].insert(0,0)
        trotbond[1].insert(0,0)

    # Compute costs
    trotcost[1] = np.array(trotterLengths) * (np.array(trotbond[1])**3)
    if not justLengths:
        paicost[1] = np.array(tePAILengths) * (np.array(paibond[1])**3)
    else:
        paicost[1] = np.array(tePAILengths) * (np.array(trotbond[1][idx+1:])**3)

    # Compute cutoff time
    threshold = max(paicost[1])
    cutoff_idx = int(np.argmax(np.array(trotcost[1]) > threshold))
    cutoff_time = trotcost[0][cutoff_idx]
    te_start, te_end = paisim[0][0], paisim[0][-1]

    # Convert simulation arrays to numpy
    t_arr = np.array(trotsim[0]); y_arr = np.array(trotsim[1])
    mask_zoom = (t_arr >= te_start) & (t_arr <= te_end)

    # ——— Increase all font sizes globally ———
    plt.rcParams.update({
        'font.size': 14,            # base font size
        'figure.titlesize': 18,     # suptitle
        'axes.titlesize': 16,       # axes titles
        'axes.labelsize': 14,       # x/y labels
        'xtick.labelsize': 12,      # tick labels
        'ytick.labelsize': 12,
        'legend.fontsize': 12,      # legend text
    })

    # --- Combined Figure: Cutoff & Improvement (top) + 3-panel metrics (bottom) ---
    fig = plt.figure(figsize=(13, 11))
    # 2 rows, 3 columns; top row spans all 3 columns, reduced vertical gap
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

    # Row 1: main plot
    ax_main = fig.add_subplot(gs[0, :])
    fig.suptitle(f"q={params[6]} | Δ={params[3]}-{params[4]} | N={params[8]} | p={params[14]}", y=0.97)

    # Mask and cutoff
    mask_cut = t_arr <= cutoff_time
    ax_main.plot(t_arr[mask_cut], y_arr[mask_cut], color='black', label='Trotterization (cutoff)')
    ax_main.axvline(cutoff_time, linestyle='--', color='black')
    ax_main.errorbar(paisim[0], paisim[1], yerr=paisim[2], color='tab:green', label='TE-PAI')
    ax_main.axvline(te_end, linestyle='--', color='tab:green')
    ax_main.axvspan(te_start, cutoff_time, color='gray', alpha=0.2)
    ax_main.axvspan(cutoff_time, te_end, color='tab:green', alpha=0.2, label='TE-PAI advantage')

    # Zoom rectangle
    y_min_zoom = min(np.min(y_arr[mask_zoom]), np.min(paisim[1]))
    y_max_zoom = max(np.max(y_arr[mask_zoom]), np.max(paisim[1]))
    if not aligned:
        rect = Rectangle((te_start, y_min_zoom), te_end - te_start, y_max_zoom - y_min_zoom,
                        linestyle='--', edgecolor='black', fill=False)
        ax_main.add_patch(rect)

    ax_main.set_xlabel('Time')
    ax_main.set_ylabel(r'$\langle X_0 \rangle$')
    ax_main.set_title(r"$\mathbf{A}$: TE-PAI simulation advantage")
    legend = ax_main.legend(loc='upper right', framealpha=1)
    legend.get_frame().set_facecolor('white')
    ax_main.grid(True)

    # Inset zoom
    if not aligned:
        inset = inset_axes(ax_main, width='50%', height='40%', loc='lower left')
        inset.plot(t_arr[mask_zoom], y_arr[mask_zoom], color='gray', linestyle='--', label='Trotterization (continued)')
        inset.errorbar(paisim[0], paisim[1], yerr=paisim[2], fmt='o', color='tab:green', label='_nolegend_')
        inset.set_xlim(te_start, te_end)
        inset.set_xticks([]); inset.set_yticks([])
        il = inset.legend(loc='upper right', framealpha=1)
        il.get_frame().set_facecolor('white')
        inset.grid(True)

    # Row 2, panel A: bond dimension
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(trotbond[0], trotbond[1], color='black', label='Trotterization')
    ax3.plot(paibond[0], paibond[1], color='tab:green', label='TE-PAI')
    max_te_pa = max(paibond[1]); top_y = ax3.get_ylim()[1]
    mask_excess = np.array(trotbond[1]) > max_te_pa
    if mask_excess.any():
        x_min = np.array(trotbond[0])[mask_excess].min()
        x_max = np.array(trotbond[0])[mask_excess].max()
        ax3.add_patch(Rectangle((x_min, max_te_pa), x_max - x_min, top_y - max_te_pa,
                                color='gray', alpha=0.3))
    ax3.set_ylabel('Dimension')
    ax3.set_title(r"$\mathbf{B}$: Largest bond dimension", pad=10)
    ax3.legend(loc='upper left', framealpha=1)
    ax3.grid(True)
    ax3.set_xlabel('Time')

    # Row 2, panel B: gate count
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(trotbond[0], trotterLengths, color='black')
    ax4.plot(paisim[0], tePAILengths, color='tab:green')
    max_te_len = max(tePAILengths); top_y = ax4.get_ylim()[1]
    mask_len = np.array(trotterLengths) > max_te_len
    if mask_len.any():
        x_min = np.array(trotbond[0])[mask_len].min(); x_max = np.array(trotbond[0])[mask_len].max()
        ax4.add_patch(Rectangle((x_min, max_te_len), x_max - x_min, top_y - max_te_len,
                                color='gray', alpha=0.3))
    ax4.set_ylabel('Gates')
    ax4.set_title(r"$\mathbf{C}$: Circuit gate count", pad=10)
    ax4.grid(True)
    ax4.set_xlabel('Time')

    # Row 2, panel C: compute cost
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(trotcost[0], trotcost[1], color='black')
    ax5.plot(paicost[0], paicost[1], color='tab:green')
    max_te_cost = max(paicost[1]); top_y = ax5.get_ylim()[1]
    mask_cost = np.array(trotcost[1]) > max_te_cost
    if mask_cost.any():
        x_min = np.array(trotcost[0])[mask_cost].min(); x_max = np.array(trotcost[0])[mask_cost].max()
        ax5.add_patch(Rectangle((x_min, max_te_cost), x_max - x_min, top_y - max_te_cost,
                                color='gray', alpha=0.3))
    ax5.set_ylabel('Flops')
    ax5.set_title(r"$\mathbf{D}$: Computational depth", pad=10)
    ax5.set_xlabel('Time')
    ax5.grid(True)

    print(f"Trotter final cost: {trotcost[1][-1]}")
    print(f"TE-PAI final cost: {paicost[1][-1]}")
    print(f"Ratio: {paicost[1][-1] / trotcost[1][-1]}")


    # Final layout and save
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # force a draw so that any prior formatting is realized
    fig.canvas.draw()

    # always use our fallback style
    for ax, data, base_label in (
            (ax3, trotbond[1], 'Dimension'),
            (ax4, trotterLengths, 'Gates'),
            (ax5, paicost[1], 'Flops'),
        ):

        # 1) compute the exponent
        maxval = np.nanmax(np.abs(data))
        exp = int(np.floor(np.log10(maxval))) if maxval > 0 else 0

        if exp > 1:
            # 2) rescale tick values
            ticks = ax.get_yticks()
            scaled = ticks / 10**exp
           #ax.set_yticks(scaled)
            ax.set_yticklabels([f"{t:g}" for t in scaled])

            # 3) set ylabel with LaTeX superscript
            ax.set_ylabel(rf"{base_label} ($\times10^{{{exp}}}$)")
        else:
            # no scaling needed
            ax.set_ylabel(base_label)
    
    plt.show()
    if aligned:
        plt.savefig("fullCalc_combined")
    else:
        plt.savefig("cutoffImprovement")

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

def plotManyCalc(folder, justLengths=False):
    param_parts = folder.split("/")[-1].split("-")
    trotterData = [[], [], [0], [0]]
    paiDatas = []
    order = {}

    params = folder.split("-")
    folder = Path(folder)
    for file in folder.iterdir():
        match file.name:
            case name if name.startswith("lie-N"):
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi = map(float, line.split(','))
                        trotterData[0].append(xi); trotterData[1].append(yi)

            case name if name.startswith("N"):
                Ti = float(name.split('-')[-1].replace('.csv', ''))
                order[Ti] = (len(paiDatas))
                closest_index = min(range(len(trotterData[0])), key=lambda i: abs(trotterData[0][i] - Ti))
                if closest_index != 0:
                    paiDatas.append([[], [], [], trotterData[2][closest_index+1:closest_index+2]])
                else:
                    paiDatas.append([[], [], [], [0]])

                with file.open() as fp:
                    header = next(fp)
                    numbers = header.split(',')[1:]  # Skip the first element, get the numbers
                    numbers = [float(num) for num in numbers]
                    trotterLen = numbers[0]; tepaiLen = numbers[1]
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        paiDatas[-1][0].append(xi); paiDatas[-1][1].append(yi); paiDatas[-1][2].append(zi)

            case name if name.startswith("lie-bond"):
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi, li = map(float, line.split(','))
                        trotterData[2].append(yi)
                        trotterData[3].append(li)
                        
            case name if name.startswith("TEPAI-bonds"):
                Ti = float(name.split('-')[-1].replace('.csv', ''))
                index = order[Ti]
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        paiDatas[index][3].append(yi)

    # Calculating costs
    trotterLens = [trotterLen*i for i in range(len(trotterData[0])+1)]
    tepaiLens = []
    tepaiCosts = []
    for i,dataset in enumerate(paiDatas):
        startTime = dataset[0][0]
        closest_index = min(range(len(trotterData[0])), key=lambda i: abs(trotterData[0][i] - startTime))
        if closest_index != 0:
            lengths = np.array([trotterLens[closest_index+1] + i*tepaiLen for i in range(len(dataset[0]))])
        else:
            lengths = np.array([i*tepaiLen for i in range(len(dataset[0]))])
        lengths = np.append(trotterLens[:closest_index], lengths)
        bonds = np.array(dataset[3])
        #bonds = np.append(trotterData[2][:closest_index], bonds)
        #bonds = np.insert(bonds, 0, trotterData[2][closest_index])
        tepaiLens.append(lengths)
        #bonds = np.insert(bonds,0,0)
        tepaiCosts.append(lengths[closest_index:]*bonds**3)

    # Unpack trotter
    t_times, t_vals, t_bonds, t_costs = trotterData

    if justLengths:
        data = paiDatas[0]
        cost = []
        for i,time in enumerate(t_times):
            ind = min(range(len(data[0])), key=lambda i: abs(data[0][i] - time))
            cost.append(trotterLens[i]*data[3][ind]**3) 
        cost.insert(0,0)
        t_costs = cost

    t_times = np.insert(t_times, 0,0)
    t_vals = np.insert(t_vals, 0,1)


 

    # Convert list to dict
    param_dict = {param_parts[i]: param_parts[i+1] for i in range(0, len(param_parts)-1, 2)}

    # Extract required parameters
    q = param_dict.get('q')
    delta = param_dict.get('Δ')
    N = param_dict.get('N')
    p = param_dict.get('p')

    # Format the title
    title = f"q={q} | Δ={delta} | N={N} | p={p}"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title)
    # 1. Values ± std
    ax = axes[0, 0]
    ax.plot(t_times, t_vals, label='Trotter', color="black")
    for i, ds in enumerate(paiDatas):
        p_times, p_vals, p_stds, _ = ds
        ax.errorbar(p_times, p_vals, yerr=p_stds, fmt='x', capsize=3, label=f'TEPAI from T={p_times[0]}')
    ax.set(title='Values vs Time', xlabel='Time', ylabel='Value')
    ax.legend()
    ax.grid(True)

    # 2. Bonds
    ax = axes[0, 1]
    #if t_bonds[0] != 0:
    #    t_bonds = np.insert(t_bonds, 0, 0)
    ax.plot(t_times, t_bonds, label='Trotter', color="black")
    for i, ds in enumerate(paiDatas):
        p_times, _, _, p_bonds = ds
        p_bonds = p_bonds
        diff = len(p_bonds)-len(p_times)
        ax.plot(p_times, p_bonds[len(p_bonds)-len(p_times):], marker='o', linestyle='--', label=f'TEPAI from T={p_times[0]}')
    ax.set(title='Bonds vs Time', xlabel='Time', ylabel='Bonds')
    ax.legend()
    ax.grid(True)

    # 3. Lengths
    ax = axes[1, 0]
    ax.plot(t_times, trotterLens, label='Trotter', color="black")
    for i, ds in enumerate(paiDatas):
        p_times = ds[0]
        ax.plot(p_times, tepaiLens[i][len(tepaiLens[i])-len(p_times):], label=f'TEPAI from T={p_times[0]}')
    ax.set(title='Chain Length vs Time', xlabel='Time', ylabel='Length')
    ax.legend()
    ax.grid(True)

    # 4. Costs
    ax = axes[1, 1]
    #if t_costs[0] != 0:
    #    t_costs = np.insert(t_costs, 0, 0)
    ax.plot(t_times, t_costs, label='Trotter', color="black")
    for i, ds in enumerate(paiDatas):
        p_times = ds[0]
        ax.plot(p_times, tepaiCosts[i][len(tepaiCosts[i])-len(p_times):], label=f'TEPAI from T={p_times[0]}')
    if not justLengths:
        ax.set(title='Cost calculated from individual bonds', xlabel='Time', ylabel='Cost')
    else:
        ax.set(title='Cost calculated from one shared bond', xlabel='Time', ylabel='Cost')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plotManyCalc2(folder, justLengths=False):
    param_parts = folder.split("/")[-1].split("-")
    trotterData = [[], [], [0], [0]]
    paiDatas = []
    order = {}

    params = folder.split("-")
    folder = Path(folder)
    for file in folder.iterdir():
        match file.name:
            case name if name.startswith("lie-N"):
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi = map(float, line.split(','))
                        trotterData[0].append(xi); trotterData[1].append(yi)

            case name if name.startswith("N"):
                Ti = float(name.split('-')[-1].replace('.csv', ''))
                order[Ti] = (len(paiDatas))
                closest_index = min(range(len(trotterData[0])), key=lambda i: abs(trotterData[0][i] - Ti))
                if closest_index != 0:
                    paiDatas.append([[], [], [], trotterData[2][closest_index+1:closest_index+2]])
                else:
                    paiDatas.append([[], [], [], [0]])

                with file.open() as fp:
                    header = next(fp)
                    numbers = header.split(',')[1:]  # Skip the first element, get the numbers
                    numbers = [float(num) for num in numbers]
                    trotterLen = numbers[0]; tepaiLen = numbers[1]
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        paiDatas[-1][0].append(xi); paiDatas[-1][1].append(yi); paiDatas[-1][2].append(zi)

            case name if name.startswith("lie-bond"):
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi, li = map(float, line.split(','))
                        trotterData[2].append(yi)
                        trotterData[3].append(li)
                        
            case name if name.startswith("TEPAI-bonds"):
                Ti = float(name.split('-')[-1].replace('.csv', ''))
                index = order[Ti]
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        paiDatas[index][3].append(yi)

    # Calculating costs
    trotterLens = [trotterLen*i for i in range(len(trotterData[0])+1)]
    tepaiLens = []
    tepaiCosts = []
    for i,dataset in enumerate(paiDatas):
        startTime = dataset[0][0]
        closest_index = min(range(len(trotterData[0])), key=lambda i: abs(trotterData[0][i] - startTime))
        if closest_index != 0:
            lengths = np.array([trotterLens[closest_index+1] + i*tepaiLen for i in range(len(dataset[0]))])
        else:
            lengths = np.array([i*tepaiLen for i in range(len(dataset[0]))])
        lengths = np.append(trotterLens[:closest_index], lengths)
        bonds = np.array(dataset[3])
        #bonds = np.append(trotterData[2][:closest_index], bonds)
        #bonds = np.insert(bonds, 0, trotterData[2][closest_index])
        tepaiLens.append(lengths)
        #bonds = np.insert(bonds,0,0)
        tepaiCosts.append(lengths[closest_index:]*bonds**3)

    # Unpack trotter
    t_times, t_vals, t_bonds, t_costs = trotterData

    if justLengths:
        data = paiDatas[0]
        cost = []
        for i,time in enumerate(t_times):
            ind = min(range(len(data[0])), key=lambda i: abs(data[0][i] - time))
            cost.append(trotterLens[i]*data[3][ind]**3) 
        cost.insert(0,0)
        t_costs = cost

    t_times = np.insert(t_times, 0,0)
    t_vals = np.insert(t_vals, 0,1)

    # Unpack trotter
    t_times, t_vals, t_bonds, t_costs = trotterData
    t_times = np.insert(t_times, 0, 0)
    t_vals = np.insert(t_vals, 0, 1)


    # Convert list to dict
    param_dict = {param_parts[i]: param_parts[i+1] for i in range(0, len(param_parts)-1, 2)}

    # Extract required parameters
    q = param_dict.get('q')
    delta = param_dict.get('Δ')
    N = param_dict.get('N')
    p = param_dict.get('p')

    # Format the title
    title = f"q={q} | Δ={delta} | N={N} | p={p}"

    # --- Begin modified plotting ---
    # Two subplots: Values and Chain Length
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title)

    # 1. Values ± std with inset zoom
    ax = axes[0]
    ax.plot(t_times, t_vals, label='Trotter', color='black')
    for ds in paiDatas:
        p_times, p_vals, p_stds, _ = ds
        p_stds = []
        denominator = int(delta.split("pi_over_")[1])
        d =  np.pi / denominator
        for time in p_times:
            p_stds.append(calcOverhead(int(q), time-p_times[0], d))
        ax.errorbar(p_times, p_vals, yerr=p_stds,
                    fmt=' ',               # no marker, no line
                    capsize=5,
                    #ecolor='black',        # optional: sets error bar color
                    elinewidth=3,        # optional: sets error bar thickness
                    label=f'TEPAI from T={p_times[0]}')
    #ax.set(title='Values vs Time', ylabel='Value')
    ax.set_ylabel(r'$\langle X_0 \rangle$')
    ax.legend(loc='upper right')
    ax.grid(True)

    # Determine zoom region: last 0.5 units in t, with padding
    tmax = t_times.max()
    region = 0.5
    pad = 0.05  # padding on x-axis before and after
    x0 = max(0, tmax - region - pad)
    x1 = tmax + pad

    # Gather all y-data in region
    mask_main = (t_times >= x0) & (t_times <= x1)
    y_vals = t_vals[mask_main]
    for ds in paiDatas:
        pts = np.array(ds[0])
        vals = np.array(ds[1])
        mask = (pts >= x0) & (pts <= x1)
        if np.any(mask):
            y_vals = np.concatenate([y_vals, vals[mask]])
    y0, y1 = y_vals.min(), y_vals.max()

    # Add padding on y-axis
    y_pad_frac = 0.1  # 10% padding
    ypad = (y1 - y0) * y_pad_frac
    y0_p = y0 - ypad
    y1_p = y1 + ypad

    # Add dashed rectangle on main axes
    rect = Rectangle((x0, y0_p), x1 - x0, y1_p - y0_p,
                     edgecolor='black', linestyle='--', fill=False)
    ax.add_patch(rect)

    # Create inset axes (lower left corner)
    axins = inset_axes(ax, width='50%', height='40%', loc='lower left', borderpad=2)
    axins.plot(t_times, t_vals, color='black')
    for ds in paiDatas:
        p_times, p_vals, p_stds, _ = ds
        axins.scatter(p_times, p_vals,
                    marker='x', s=100)
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0_p, y1_p)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.legend(title="Mean values")
    # 2. Chain Length vs Time (gate count)
    ax2 = axes[1]
    ax2.plot(t_times, trotterLens, label='Trotter', color='black')
    for i, ds in enumerate(paiDatas):
        p_times = ds[0]
        ax2.plot(p_times, tepaiLens[i][-len(p_times):], label=f'TEPAI from T={p_times[0]}')
    #ax2.set(title='Chain Length vs Time', xlabel='Time', ylabel='Length')
    ax2.set_ylabel('Circuit gate count')
    ax2.set_xlabel('Time')
    ax2.grid(True)

    for ax in (ax, ax2):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 4))
        ax.yaxis.set_major_formatter(formatter)


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    #plt.savefig("manyCalcPlot")
    #plt.show()


    t_vals_quadratic = np.array(t_times)
    # --- Compute constant for Hamiltonian subplot ---
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=int(q))
    hamil = Hamiltonian.spin_chain_hamil(int(q), freqs)
    val = hamil.l1_norm(0.1)  # Evaluate 1-norm at T = 0.1
    print("H(T=0.1) L1 norm =", val)

    # --- Setup the quadratic dependence in time based on val ---
    quad_norm = []
    for t in t_times:
        quad_norm.append(hamil.l1_norm(t)**2)


    #quad_norm = (val * t_vals_quadratic) ** 2

    # --- Create figure with two subplots ---
    fig2, (ax_quad, ax_norm) = plt.subplots(1, 2, figsize=(10, 4))

    # --- Subplot 1: Gate count comparison ---
    # Compute constant c for quadratic gate count: G(T) = c T^2
    T1 = t_times[1]
    G1 = trotterLens[1]
    c = G1 / (T1 ** 2)
    quad_gate_count = c * t_vals_quadratic ** 2

    # First TE-PAI data
    tepai_times = paiDatas[0][0]
    tepai_gates = tepaiLens[0][-len(tepai_times):]

    # Plotting
    ax_quad.plot(t_times, trotterLens, label='Trotter', color='black')
    ax_quad.plot(t_vals_quadratic, quad_gate_count, '--', color='tab:red', label='Gate count with constant error')
    ax_quad.plot(tepai_times, tepai_gates, label=f'TE-PAI from T={tepai_times[0]}', color='tab:blue')

    ax_quad.set_title("Circuit gate count")
    ax_quad.set_ylabel("Gates")
    ax_quad.set_xlabel("Time")
    ax_quad.legend()
    ax_quad.grid(True)

    # --- Subplot 2: Quadratic growth from H-norm ---
    ax_norm.plot(t_vals_quadratic, quad_norm, color='tab:green', linestyle='--', label=fr'TE-PAI at constant error')
    ax_norm.set_title("Circuit shot count")
    ax_norm.set_xlabel("Time")
    ax_norm.set_ylabel("Shots")
    ax_norm.legend()
    ax_norm.grid(True)

    # Format y axes with scientific notation if needed
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))
    ax_quad.yaxis.set_major_formatter(formatter)
    ax_norm.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig("constantErr")
    plt.show()

def calcOverhead(q, T, Δ):
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    return np.exp(2 * hamil.l1_norm(T) * np.tan(Δ / 2))-1

def plotMainCalc(folder, both=True):
    trotsim1  = [[], []]; trotsim2  = [[], []]; paisim  = [[], [], []]
    trotbond1 = [[], []]; trotbond2 = [[], []]; paibond = [[], []]
    trotcost1 = [[], []]; trotcost2 = [[], []]; paicost = [[], []]
    
    params = folder.split("-")
    folder = Path(folder)
    for file in folder.iterdir():
        match file.name:
            case "lengths.csv":
                with file.open() as fp:
                    next(fp)
                    line = next(fp)
                    reader = csv.reader([line.strip()])
                    row = next(reader)
                    trotterLengths1 = ast.literal_eval(row[0])
                    trotterLengths2 = ast.literal_eval(row[1])
                    tePAILengths   = ast.literal_eval(row[2])

            case name if name.startswith("lie"):
                second = name.endswith("fixedCircuit.csv")
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi = map(float, line.split(','))
                        if second:
                            trotsim2[0].append(xi); trotsim2[1].append(yi)
                        else:
                            trotsim1[0].append(xi); trotsim1[1].append(yi)

            case name if name.startswith("N"):
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        paisim[0].append(xi); paisim[1].append(yi); paisim[2].append(zi)

            case name if name.startswith("trotter-bonds"):
                second = name.endswith("fixedCircuit.csv")
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        if not second:
                            trotbond1[0].append(xi)
                            trotbond1[1].append(yi)
                            trotcost1[0].append(xi)
                            trotcost1[1].append(zi)
                        else:
                            trotbond2[0].append(xi)
                            trotbond2[1].append(yi)
                            trotcost2[0].append(xi)
                            trotcost2[1].append(zi)

            case name if name.startswith("TEPAI-bonds"):
                with file.open() as fp:
                    next(fp)
                    for line in fp:
                        xi, yi, zi = map(float, line.split(','))
                        paibond[0].append(xi)
                        paibond[1].append(yi)
                        paicost[0].append(xi)
                        paicost[1].append(zi)

    if trotbond1[1][0] != 0:
        trotbond1[1].insert(0,0)
        trotbond1[0].insert(0,0)

    # complexity
    trotcost1[1] = np.array(trotterLengths1) * np.array(trotbond1[1])**3
    times2        = [t - (trotbond2[0][0] - trotbond1[0][-1]) for t in trotcost2[0]]
    trotcost2[1] = np.array(trotterLengths2 + np.max(trotterLengths1)) * np.array(trotbond2[1])**3
    paicost[1]   = np.array(tePAILengths   + np.max(trotterLengths1)) * np.array(paibond[1])**3

    title = f"q={params[6]} | Δ={params[3]}-{params[4]} | N1={params[8]} | N2={params[12]} | p={params[14]}" 

   # 1) Grab the final values of the initial run:
    t0        = trotbond1[0][-1]       # final time of init trotter
    bond_last = trotbond1[1][-1]       # final bond of init trotter
    cost_last = trotcost1[1][-1]       # final cost of init trotter
    len_last  = trotterLengths1[-1]    # final # gates of init trotter

    # 2) Make everything a NumPy array:
    trotbond2[0]      = np.array(trotbond2[0])
    trotbond2[1]      = np.array(trotbond2[1])
    paibond   [0]     = np.array(paibond   [0])
    paibond   [1]     = np.array(paibond   [1])
    trotcost2[0]      = np.array(trotcost2[0])
    trotcost2[1]      = np.array(trotcost2[1])
    paicost   [0]     = np.array(paicost   [0])
    paicost   [1]     = np.array(paicost   [1])
    trotterLengths2   = np.array(trotterLengths2)
    tePAILengths      = np.array(tePAILengths)

    # 3) Prepend the initial‐final to each continuation series:
    if both:
        trotbond2[0]     = np.concatenate(([t0],           trotbond2[0]))
        trotbond2[1]     = np.concatenate(([bond_last],    trotbond2[1]))
        paibond   [0]    = np.concatenate(([t0],           paibond[0]))
        paibond   [1]    = np.concatenate(([bond_last],    paibond[1]))

        trotcost2[0]     = np.concatenate(([t0],           trotcost2[0]))
        trotcost2[1]     = np.concatenate(([cost_last],    trotcost2[1]))
        paicost   [0]    = np.concatenate(([t0],           paicost[0]))
        paicost   [1]    = np.concatenate(([cost_last],    paicost[1]))

    trotterLengths2  = np.concatenate(([len_last],     trotterLengths2))
    tePAILengths     = np.concatenate(([len_last],     tePAILengths))

    te_start = paisim[0][0]
    te_end   = paisim[0][-1]
    times = trotbond2[0]

    t_trot = np.array(trotsim1[0] + trotsim2[0])
    y_trot = np.array(trotsim1[1] + trotsim2[1])
    t_cost = np.concatenate([trotcost1[0], trotcost2[0]])
    c_cost = np.concatenate([trotcost1[1], trotcost2[1]])
    threshold  = paicost[1].max()
    cutoff_idx = np.argmax(c_cost > threshold)
    cutoff_time = t_cost[cutoff_idx]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    fig.suptitle(title)

    # 1) Full sim
    ax0 = axes[0]
    ax0.plot(trotsim1[0], trotsim1[1], color='black', label='Init Trotter')
    ax0.plot(trotsim2[0], trotsim2[1], color='gray', linestyle='--',  label='Cont Trotter')
    ax0.errorbar(paisim[0], paisim[1], yerr=paisim[2],
                 color='tab:green', label='TE-PAI')
    ax0.set_ylabel('Observable')
    ax0.set_title('1) Full Simulation')
    ax0.legend(); ax0.grid(True)

    # 2) Zoom only on TE-PAI window (masking)
    ax1 = axes[1]
    t1, y1 = np.array(trotsim1[0]), np.array(trotsim1[1])
    t2, y2 = np.array(trotsim2[0]), np.array(trotsim2[1])
    tp, yp, ep = np.array(paisim[0]), np.array(paisim[1]), np.array(paisim[2])
    m1 = (t1 >= te_start) & (t1 <= te_end)
    m2 = (t2 >= te_start) & (t2 <= te_end)
    mp = (tp >= te_start) & (tp <= te_end)
    ax1.plot(t1[m1], y1[m1], color='black', label='Init Trotter')
    ax1.plot(t2[m2], y2[m2], color='gray',  label='Cont Trotter', linestyle="--")
    ax1.errorbar(tp[mp], yp[mp], yerr=ep[mp], color='tab:green', label='TE-PAI')
    ax1.set_xlabel('Time')
    ax1.set_title('2) Zoom on TE-PAI Region')
    ax1.legend(); ax1.grid(True)

    # 3) Cutoff & improvement regions
    ax2 = axes[2]
    ax2.plot(t_trot[t_trot <= cutoff_time],
             y_trot[t_trot <= cutoff_time],
             color='black', label='Trotter (cutoff)')
    ax2.axvline(cutoff_time, linestyle='--', color='black')
    ax2.errorbar(paisim[0], paisim[1], yerr=paisim[2],
                 color='tab:green', label='TE-PAI')
    ax2.axvline(te_end, linestyle='--', color='tab:green')
    ax2.axvspan(te_start, cutoff_time, color='gray', alpha=0.2)
    ax2.axvspan(cutoff_time, te_end, label='TE-PAI advantage',     color='tab:green', alpha=0.2)
    ax2.set_title('3) Cutoff & Improvement') 
    ax2.legend(); ax2.grid(True)

    # 4) Bond sizes
    ax3 = axes[3]
    ax3.plot(trotbond1[0], trotbond1[1], color='black', label='Init Trotter')
    ax3.plot(times,       trotbond2[1], color='gray',  label='Cont Trotter')
    ax3.plot(paibond[0],  paibond[1],   color='tab:green', label='TE-PAI')
    ax3.set_xlabel('Time'); ax3.set_ylabel('Max bond')
    ax3.set_title('4) Bond Dimension')
    ax3.legend(); ax3.grid(True)

    # 5) Gate counts
    ax4 = axes[4]
    adj_t2  = trotterLengths2.copy()
    adj_te  = tePAILengths.copy()

    if both:
        adj_t2[1:] += len_last
        adj_te[1:] += len_last
    else:
        adj_t2[1:] += len_last
        adj_te[1:] += len_last
        times = np.insert(times, 0, np.max(trotbond1[0]))

    ax4.plot(trotbond1[0], trotterLengths1, color='black', label='Init Trotter')
    ax4.plot(times,       adj_t2,          color='gray',  label='Cont Trotter')
    ax4.plot(times,       adj_te,          color='tab:green', label='TE-PAI')
    ax4.set_xlabel('Time'); ax4.set_ylabel('# gates')
    ax4.set_title('5) Gate Count')
    ax4.legend(); ax4.grid(True)
    times = trotbond2[0]

    # 6) Calculation cost
    ax5 = axes[5]
    ax5.plot(trotcost1[0], trotcost1[1], color='black', label='Init Trotter')
    ax5.plot(times,        trotcost2[1],  color='gray',  label='Cont Trotter')
    ax5.plot(times,        paicost[1],    color='tab:green', label='TE-PAI')
    ax5.set_xlabel('Time'); ax5.set_ylabel('Cost')
    ax5.set_title('6) Compute Cost')
    ax5.legend(); ax5.grid(True)

    plt.tight_layout()
    plt.show()

def trotterThenTEPAI(folder, trotterN, trottern, trotterT, saveAndPlot=False, optimize=False, flip=False, confirm=False,  base_dir='TE-PAI-noSampling/data/plotting'):
    folder = strip_trailing_dot_zero(folder)
    data_dict = JSONtoDict(folder)
    data_arrs,Ts,params,pool = DictToArr(data_dict, True)
    N,n,c,Δ,T,q = params
    T = round(float(T), 8)

    # Running the trotter simulation up to a time trotterT
    # NB: T =/= trotterT
    trotterTs, res, complexity, circuit = trotter(N=trotterN, n_snapshot=trottern, T=trotterT, q=int(q), compare=False, save=True, draw=False, flip=flip)
    save_trotter(trotterTs, complexity[0], complexity[1], trotterN, trottern, trotterT, q)

    if confirm:
        confirmTs, _, confirmComp, _ = trotter(N=trotterN, n_snapshot=10, T=T, startTime=trotterT, q=int(q), compare=False, save=True, draw=False, flip=flip, fixedCircuit=circuit)
        save_trotter(confirmTs, confirmComp[0], confirmComp[1], trotterN, 10, T, q)

    char = "c"
    dT = extract_dT_value(folder)
    char = "p"
    averages, stds, circuit, costs = getPool(data_arrs, params, dT, False, optimize, flip=flip, fixedCircuit = circuit)
    Ts = np.linspace(trotterT,trotterT + T,len(averages))

    pattern = r"dT-([0-9]+(?:\.[0-9]+)?)"
    match = re.search(pattern, folder)
    saveData(N,n,c,Δ,Ts,q,float(match.group(1)),averages,stds,char)

    filename = f"TEPAI-bonds-N-{N}-n-{n}-T-{T}-q-{q}.csv"
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'z'])
        for xi, yi, zi in zip(Ts, costs[0], costs[1]):
            writer.writerow([xi, yi, zi])

    organize_trotter_tepai()

def organize_trotter_tepai(
    plotting_dir: Path = Path("TE-PAI-noSampling/data/plotting"),
    target_base:  Path = Path("TE-PAI-noSampling/data/trotterThenTEPAI"),
):
    """
    Finds in `plotting_dir`:
      • One or two files matching `lie-N-{N}-T-{T}-q-{q}(…).csv`
      • Exactly one TE-PAI file matching
          N-{N2}-n-{n}-p-{p}-Δ-pi_over_{delta}-T-{T2}-q-{q}-dT-{dt}.csv
      • Any number of trotter-bonds files matching
          trotter-bonds-N-{N}-n-{n}-T-{T}-q-{q}.csv
      • Optionally, a file named `lengths.csv`

    Builds (or replaces) a folder named
      Δ-pi_over-{delta}-q-{q}-N1-{N1}-T1-{T1}-N2-{N2}-p-{p}-T2-{T2}-dt-{dt}

    Moves all lie-file(s), all trotter-bonds files, the TE-PAI file,
    and (if present) `lengths.csv` into that folder, then into `target_base`.
    """
    # patterns
    pat_lie = re.compile(
        r"^lie-N-(?P<N>\d+)-T-(?P<T>[0-9.]+)-q-(?P<q>\d+)(?:-[^.]+)?\.csv$"
    )
    pat_tep = re.compile(
        r"^N-(?P<N2>\d+)-n-(?P<n>\d+)-p-(?P<p>\d+)-Δ-pi_over_(?P<delta>[^-]+)"
        r"-T-(?P<T2>[0-9.]+)-q-(?P<q>\d+)-dT-(?P<dt>[0-9.]+)\.csv$"
    )
    pat_trot = re.compile(
        r"^(?:trotter|TEPAI)-bonds-N-(?P<N>\d+)-n-(?P<n>\d+)-T-(?P<T>[0-9.]+)-q-(?P<q>\d+)(?:-[^.]+)?\.csv$",
        re.IGNORECASE
    )

    lie_runs     = []
    tep_match    = None
    trotter_runs = []
    lengths_file = None

    # scan directory
    for f in plotting_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if (m := pat_lie.match(name)):
            gd = m.groupdict()
            lie_runs.append((f,
                             float(gd["T"]),    # for sorting
                             gd["T"],           # for exact naming
                             gd["N"],
                             gd["q"]))
        elif (m := pat_tep.match(name)):
            tep_match = (f, m.groupdict())
        elif (m := pat_trot.match(name)):
            gd = m.groupdict()
            trotter_runs.append((f, gd["q"]))
        elif name == "lengths.csv":
            lengths_file = f

    # require ≥1 lie-file and exactly 1 TE-PAI
    if len(lie_runs) < 1 or tep_match is None:
        raise FileNotFoundError(
            f"Need at least 1 lie-file + 1 TE-PAI file in {plotting_dir}, "
            f"found {len(lie_runs)} lie and "
            f"{'none' if tep_match is None else 'one'} TE-PAI."
        )

    # collect all q-values for consistency
    q_vals = {q for *_, q in lie_runs} \
           | {tep_match[1]["q"]} \
           | {q for _, q in trotter_runs}
    if len(q_vals) != 1:
        raise ValueError(f"q mismatch among files: {q_vals}")
    q_str = q_vals.pop()

    # pick the lie-run with smallest T
    lie_runs.sort(key=lambda tup: tup[1])
    _, _, T1_str, N1_str, _ = lie_runs[0]

    # TE-PAI params (use raw strings for naming)
    tep_f, tep_g = tep_match
    N2_str    = tep_g["N2"]
    p_str     = tep_g["p"]
    delta_str = tep_g["delta"]
    T2_str    = tep_g["T2"]
    dt_str    = tep_g["dt"]

    # build folder name including delta
    folder_name = (
        f"Δ-pi_over-{delta_str}"
        f"-q-{q_str}"
        f"-N1-{N1_str}-T1-{float(T1_str)}"
        f"-N2-{N2_str}-p-{p_str}-T2-{float(T2_str)}-dt-{dt_str}"
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
    for f, *_ in lie_runs:
        shutil.move(str(f), staging / f.name)

    # move trotter-bonds files
    for f, _ in trotter_runs:
        shutil.move(str(f), staging / f.name)

    # move TE-PAI file
    shutil.move(str(tep_f), staging / tep_f.name)

    # move lengths.csv if found
    if lengths_file is not None:
        shutil.move(str(lengths_file), staging / lengths_file.name)

    # relocate folder to target_base
    shutil.move(str(staging), target)

    msg = (
        f"Moved {len(lie_runs)} lie-file(s), {len(trotter_runs)} trotter-bonds file(s), "
        f"+ 1 TE-PAI file"
    )
    if lengths_file:
        msg += ", + lengths.csv"
    msg += f" into {target}"
    print(msg)

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
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    te_pai = TE_PAI(hamil, q, Δ, dt2, 1000, n_tep)
    tep_len = te_pai.expected_num_gates

    return lie_runs, bond_runs, x_t, y_t, e_t, q, N1, N2, p, T1, dt2, tep_len

def plotTrotterPAI(folder):
    lie_runs, bond_runs, x_t, y_t, e_t, q, N1, N2, p, T1, dt2, tep_len = (getTrotterPai(folder))
    # --- Subplot 1: Trotterization + TE-PAI continuation ---
    n1 = 0
    n2 = 0
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    lie_runs.sort(key=lambda t: t[1])
    (f_small, _, N_small), (f_large, _, N_large) = lie_runs
    # smaller-T trotter
    xs, ys = [], []
    with f_small.open() as fp:
        next(fp)
        for line in fp:
            xi, yi = line.strip().split(',')
            xs.append(float(xi)); ys.append(float(yi))
            n2 += 1
    ax1.plot(xs, ys, linestyle='-', color='darkblue', label=f"Trotterization-N-{N_small}")
    # larger-T trotter
    xl, yl = [], []
    with f_large.open() as fp:
        next(fp)
        for line in fp:
            xi, yi = line.strip().split(',')
            xl.append(float(xi)); yl.append(float(yi))
            n1 += 1
    ax1.plot(xl, yl, linestyle='--', color='gray', label=f"Trotterization-N-{N_large}")
    ax1.errorbar(x_t, y_t, yerr=e_t, fmt='o', linestyle='-', color='tab:blue',
                 label=f"TE-PAI continuation-N-{N2}-p-{p}")
    ax1.set_xlabel("time"); ax1.set_ylabel("x expectation value"); ax1.legend()
    
    _, _, _, circuit1 = trotter(int(N1), 1, T1/n1, q, compare=False, save=False, flip=True)
    _, _, _, circuit2 = trotter(int(N2), 1, dt2, q, compare=False, save=False, flip=True)
    trotter1_len = len(circuit1.gates)
    trotter2_len = len(circuit2.gates)

    # Separate trotter vs TEPAI
    trotter_runs = [br for br in bond_runs if br[1]['type'].lower().startswith('trotter')]
    tepai_runs = [br for br in bond_runs if br[1]['type'].lower().replace('-', '') == 'tepai']
    # Sort trotter by T
    trotter_runs.sort(key=lambda t: float(t[1]['T']))
    # First trotter (small)
    f_s, gd_s = trotter_runs[0];  xs2, ys2, zs2 = [], [], []
    with f_s.open() as fp:
        next(fp)
        for line in fp:
            xi, yi, zi = line.strip().split(',')
            xs2.append(float(xi)); ys2.append(float(yi)); zs2.append(float(zi))

    ys2 = np.array(ys2)
    zs2 = [trotter1_len * i * ys**3 for i, ys in enumerate(ys2)]
    ax3.plot(xs2, zs2, linestyle='-', marker='o', label=f"trotter-duration-N-{gd_s['N']}-n-{gd_s['n']}-T-{gd_s['T']}")
    ax2.plot(xs2, ys2, linestyle='-', marker='o', label=f"trotter-bonds-N-{gd_s['N']}-n-{gd_s['n']}-T-{gd_s['T']}")
    # Second trotter (large)
    f_l, gd_l = trotter_runs[1]; xl2, yl2, zl2 = [], [], []
    with f_l.open() as fp:
        next(fp)
        for line in fp:
            xi, yi, zi = line.strip().split(',')
            xl2.append(float(xi)); yl2.append(float(yi)); zl2.append(float(zi))
    yls = np.array(yl2)
    zl2 = [trotter2_len * i * yls**3 for i, yls in enumerate(yl2)]
    ax3.plot(xl2, zl2, linestyle='--', marker='o', label=f"trotter-duration-N-{gd_l['N']}-n-{gd_l['n']}-T-{gd_l['T']}")
    ax2.plot(xl2, yl2, linestyle='--', marker='o', label=f"trotter-bonds-N-{gd_l['N']}-n-{gd_l['n']}-T-{gd_l['T']}")
    # TEPAI-bonds
    if tepai_runs:
        f_t, gd_t = tepai_runs[0]; xt, yt, zt = [], [], []
        with f_t.open() as fp:
            next(fp)
            for line in fp:
                xi, yi, zi = line.strip().split(',')
                xt.append(float(xi)); yt.append(float(yi)); zt.append(float(zi))
        yt = np.array(yt)
        zt = [tep_len * i * yt**3 for i, yt in enumerate(yt)]
        ax3.plot(xt, zt, linestyle='-', marker='x', color='tab:orange', label=f"TEPAI-costs-N-{gd_t['N']}-n-{gd_t['n']}-T-{gd_t['T']}")
        ax2.plot(xt, yt, linestyle='-', marker='x', color='tab:orange', label=f"TEPAI-bonds-N-{gd_t['N']}-n-{gd_t['n']}-T-{gd_t['T']}")
    plt.show()

def plot_trotter_then_tepai(
    Δ_name: str,
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
    Locate folder q-{q}-N1-{N1}-T1-{T1}-N2-{N2}-p-{p}-T-{T2}-dt-{dt}, then:
      • If 2 lie-*.csv:
         – smaller-T: darkblue solid
         – larger-T: gray dashed
      • If 1 lie-*.csv:
         – that run: darkblue solid
      • TE-PAI: tab:blue errorbars
      • Bond data: two trotter-bonds files plus one TEPAI-bonds file plotted in second subplot
      • Skip first data point for the smaller-T trotter-bonds
      • Skip first data point for the TEPAI-bonds, and shift its x-values by the max x of the first trotter-bonds

    X axis = time, Y axis = x expectation value.
    Title = "Trotterization followed by TE-PAI for {q} qubits for total time {T1+T2}"
    """
    # Build folder path
    folder = base_dir / f"Δ-{Δ_name}-q-{q}-N1-{N1}-T1-{float(T1)}-N2-{N2}-p-{p}-T2-{float(T2)}-dt-{dt}"
    if not folder.is_dir():
        raise FileNotFoundError(f"No such folder: {folder}")

    # Compile filename patterns
    pat_lie = re.compile(r"^lie-N-(?P<N>\d+)-T-(?P<T>[0-9.]+)-q-(?P<q>\d+)(?:-fixedCircuit)?\.csv$")
    pat_tep = re.compile(
        r"^N-(?P<N2>\d+)-n-(?P<n>\d+)-p-(?P<p>\d+)-Δ-pi_over_[^–]+"
        r"-T-(?P<T2>[0-9.]+)-q-(?P<q>\d+)-dT-(?P<dt>[0-9.]+)\.csv$"
    )
    # Match both trotter-bonds and TEPAI-bonds (type captured)
    pat_bonds = re.compile(
        r"^(?P<type>trotter|TE-?PAI)-bonds-N-(?P<N>\d+)-n-(?P<n>\d+)-T-(?P<T>\d+(?:\.\d+)?)-q-(?P<q>\d+)\.csv$",
        re.IGNORECASE
    )

    # Containers for runs
    lie_runs = []
    tep_file = None
    bond_runs = []

    # Scan directory for matching files
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
        raise FileNotFoundError("Expected 3 bond CSV files (2 trotter + 1 TEPAI) in folder")

    # Load TE-PAI data
    x_t, y_t, e_t = [], [], []
    with tep_file.open() as fp:
        next(fp)
        for line in fp:
            xi, yi, erri = line.strip().split(',')
            x_t.append(float(xi)); y_t.append(float(yi)); e_t.append(float(erri))

    # --- Subplot 1: Trotterization + TE-PAI continuation ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    if len(lie_runs) == 1:
        file_s, _, N_s = lie_runs[0]
        x_s, y_s = [], []
        with file_s.open() as fp:
            next(fp)
            for line in fp:
                xi, yi = line.strip().split(',')
                x_s.append(float(xi)); y_s.append(float(yi))
        ax1.plot(x_s, y_s, linestyle='-', color='darkblue', label=f"Trotterization-N-{N_s}")
    else:
        lie_runs.sort(key=lambda t: t[1])
        (f_small, _, N_small), (f_large, _, N_large) = lie_runs
        # smaller-T trotter
        xs, ys = [], []
        with f_small.open() as fp:
            next(fp)
            for line in fp:
                xi, yi = line.strip().split(',')
                xs.append(float(xi)); ys.append(float(yi))
        ax1.plot(xs, ys, linestyle='-', color='darkblue', label=f"Trotterization-N-{N_small}")
        # larger-T trotter
        xl, yl = [], []
        with f_large.open() as fp:
            next(fp)
            for line in fp:
                xi, yi = line.strip().split(',')
                xl.append(float(xi)); yl.append(float(yi))
        ax1.plot(xl, yl, linestyle='--', color='gray', label=f"Trotterization-N-{N_large}")
    ax1.errorbar(x_t, y_t, yerr=e_t, fmt='o', linestyle='-', color='tab:blue',
                 label=f"TE-PAI continuation-N-{N2}-p-{p}")
    ax1.set_xlabel("time"); ax1.set_ylabel("x expectation value"); ax1.legend()

    # Separate trotter vs TEPAI
    trotter_runs = [br for br in bond_runs if br[1]['type'].lower().startswith('trotter')]
    tepai_runs = [br for br in bond_runs if br[1]['type'].lower().replace('-', '') == 'tepai']
    # Sort trotter by T
    trotter_runs.sort(key=lambda t: float(t[1]['T']))
    # First trotter (small)
    f_s, gd_s = trotter_runs[0];  xs2, ys2, zs2 = [], [], []
    with f_s.open() as fp:
        for line in fp:
            xi, yi, zi = line.strip().split(',')
            xs2.append(float(xi)); ys2.append(float(yi)); zs2.append(float(zi))

    ax3.plot(xs2, zs2, linestyle='-', marker='o', label=f"trotter-costs-N-{gd_s['N']}-n-{gd_s['n']}-T-{gd_s['T']}")
    ax2.plot(xs2, ys2, linestyle='-', marker='o', label=f"trotter-bonds-N-{gd_s['N']}-n-{gd_s['n']}-T-{gd_s['T']}")
    # Second trotter (large)
    f_l, gd_l = trotter_runs[1]; xl2, yl2, zl2 = [], [], []
    with f_l.open() as fp:
        next(fp)
        for line in fp:
            xi, yi, zi = line.strip().split(',')
            xl2.append(float(xi)); yl2.append(float(yi)); zl2.append(float(zi))
    ax3.plot(xl2, zl2, linestyle='--', marker='o', label=f"trotter-costs-N-{gd_l['N']}-n-{gd_l['n']}-T-{gd_l['T']}")
    ax2.plot(xl2, yl2, linestyle='--', marker='o', label=f"trotter-bonds-N-{gd_l['N']}-n-{gd_l['n']}-T-{gd_l['T']}")
    # TEPAI-bonds
    if tepai_runs:
        f_t, gd_t = tepai_runs[0]; xt, yt, zt = [], [], []
        with f_t.open() as fp:
            next(fp)
            for line in fp:
                xi, yi, zi = line.strip().split(',')
                xt.append(float(xi)); yt.append(float(yi)); zt.append(float(zi))
        ax3.plot(xt, zt, linestyle='-', marker='x', color='tab:orange', label=f"TEPAI-costs-N-{gd_t['N']}-n-{gd_t['n']}-T-{gd_t['T']}")
        ax2.plot(xt, yt, linestyle='-', marker='x', color='tab:orange', label=f"TEPAI-bonds-N-{gd_t['N']}-n-{gd_t['n']}-T-{gd_t['T']}")

    ax2.set_xlabel("time"); ax2.set_ylabel("x expectation value");
    ax2.set_title(f"Bond data for {q} qubits")
    ax2.legend()

    fig.suptitle(f"Trotterization followed by TE-PAI for {q} qubits for total time {T1+T2}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
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
   
def save_costs_csv(folder_path, Ts, costs, filename='output.csv'):
    """
    Saves Ts and costs into a CSV file in the given folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder where the CSV should be saved.
    Ts : Iterable
        Sequence of x-values.
    costs : sequence of two iterables
        costs[0] → y-values, costs[1] → z-values.
    filename : str, optional
        Name of the CSV file (default 'output.csv').
    """
    # 1. Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # 2. Build the full file path
    file_path = os.path.join(folder_path, filename)

    # 3. Write the CSV
    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # header
        writer.writerow(['x', 'y', 'z'])
        # rows
        for x, y, z in zip(Ts, costs[0], costs[1]):
            writer.writerow([x, y, z])

    print(f"Saved CSV to {file_path}")

def load_costs_csv(folder_path, filename='output.csv'):
    """
    If `<folder_path>/<filename>` exists, read its 'x','y','z' columns
    and return them as (Ts, ys, zs). Otherwise return None.

    Parameters
    ----------
    folder_path : str
        Path to the folder where the CSV might live.
    filename : str, optional
        Name of the CSV file (default 'output.csv').

    Returns
    -------
    tuple of lists (Ts, ys, zs) or None
    """
    file_path = os.path.join(folder_path, filename)
    if not os.path.isfile(file_path):
        print(f"No file named {filename!r} found in folder {folder_path!r}.")
        return None

    Ts, ys, zs = [], [], []
    with open(file_path, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        expected = ['x', 'y', 'z']
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected columns: {reader.fieldnames!r}, expected {expected!r}")
        for row in reader:
            Ts.append(float(row['x']))
            ys.append(float(row['y']))
            zs.append(float(row['z']))

    return Ts, ys, zs

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

def getComplexity(circuit):

    mps = circuit.psi
    costs = mps.contraction_cost(optimize="greedy", output_inds=())

    # 1. Naïve cost (always 128 for N=4):
    #print("Default cost:", mps.contraction_cost())

    # 2. Searched cost via optimize:
    #print("Greedy cost:", mps.contraction_cost(optimize="greedy"))

    # 3. ContractionTree for absolute cost:
    tree = mps.contraction_tree(optimize="greedy")
    #print("Absolute FLOPs:", tree.contraction_cost(log=None))

    # 4. PathInfo for detailed view:
    info = mps.contraction_info(optimize="greedy")
    #print("Flops from PathInfo:", info)

    tree = circuit.psi.contraction_tree(optimize="greedy")
    return circuit.psi.max_bond(), info.naive_cost

def compareComplexity(paths, flip=True):
    bonds = []
    costs = []
    times = []
    labels = []
    lengths = []

    for path in paths:
        data_dict = JSONtoDict(path)
        data_arrs, Ts, params, pool = DictToArr(data_dict, True)
        paramDict = parse_path(path)
        dT = paramDict['dT']
        Δ_name = paramDict['Δ']
        N, _, c, Δ, T, q = params
        q = int(q); N = int(N)
        T = round(float(T), 8)
        Δ = parse_pi_over(Δ)
        labels.append(f"q-{q}-N-{N}-T-{T}-dT-{dT}-Δ-{Δ_name}")

        if load_costs_csv(path, filename='cost.csv') == None:
            averages, stds, circuit, cost = getPool(data_arrs, params, dT, False, False, flip=flip)
            timesteps = len(averages)-1
        else:
            ts, bs, cs = load_costs_csv(path, filename='cost.csv')
            cost = [bs, cs]
            timesteps = len(ts)
        Ts = np.linspace(dT, T, timesteps)
        
        length = []
        circuit_pool, sign_pool = data_arrs[0]
        mean_length = sum(len(el) for el in circuit_pool) / len(circuit_pool) if circuit_pool else 0
        for i in range(len(Ts)):
            length.append(mean_length*(i+1))

        times.append(Ts)
        bonds.append(cost[0])
        costs.append(cost[1])
        lengths.append(length)
        save_costs_csv(path, Ts, cost, filename='cost.csv')

    bonds = np.array(bonds)
    lengths = np.array(lengths)
    durations = 8 * lengths * bonds ** 3

    # Plotting
    fig, axs = plt.subplots(1,3, figsize=(12, 4))
    for bond, cost, time, label, duration in zip(bonds, costs, times, labels, durations):
        if bond is not None:
            axs[0].plot(time, bond, label=f"{label}-bond")
        if cost is not None:
            axs[1].plot(time, cost, label=f"{label}-cost")
        if length is not None:
            axs[2].plot(time, duration, label=f"{label}-duration")
        
        #print(duration[-1]/np.max(durations))    
    
    axs[0].set_title("Max bond size")
    axs[1].set_title("Contraction cost")
    axs[2].set_title("Estimated calculation duration")
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    plt.show()

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

def getCircuit(q, flip=False, mps=True):

    if mps:
        quimb = qtn.CircuitMPS(q, cutoff = 1e-12)
    else:
        quimb = qtn.Circuit(q)

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

def plot_bond_data(folder_path="TE-PAI-noSampling/data/bonds/plot", out_file="bond_growth.png"):
    """
    Plot maximum bond dimension over time for various snapshots and Trotter parameters.
    - Sort curves by their 'n' value
    - Color gradient from green (low n) to red (high n)
    - Single solid line style
    - Legend inside the plot with title "Number of qubits" (full opacity)
    - Annotate horizontal lines at powers of two (from 2^3=8 upwards) on the right
    """
    # Compile filename patterns
    pat_lie = re.compile(r"^lie-bond-N-(?P<N>[^-]+)-T-(?P<T>[^-]+)-q-(?P<q>[^.]+)\.csv$")
    pat_trot = re.compile(r"^trotter-bonds-N-(?P<N>[^-]+)-n-(?P<n>[^-]+)-(?P<T>[^-]+)-q-(?P<q>[^.]+)\.csv$")

    # Collect data entries
    entries = []
    for filename in sorted(os.listdir(folder_path)):
        fp = os.path.join(folder_path, filename)
        m1 = pat_lie.match(filename)
        m2 = pat_trot.match(filename)
        if not (m1 or m2):
            continue
        df = pd.read_csv(fp)
        x = df.iloc[:, 0].tolist()
        y = df.iloc[:, 1].tolist()
        # ensure starts at zero
        if x and x[0] != 0:
            x.insert(0, 0)
            y.insert(0, 0)
        # Determine n value for sorting and labeling
        nval = int(m1.group('q') if m1 else m2.group('n'))
        entries.append({'nval': nval, 'x': x, 'y': y})

    if not entries:
        print(f"No matching CSV files found in {folder_path}")
        return

    # Sort entries by n value
    entries.sort(key=lambda e: e['nval'])

    # Setup plot
    plt.figure(figsize=(8, 6), dpi=225)
    cmap = plt.get_cmap('RdYlGn_r')  # green to red
    num = len(entries)

    # Track global min/max for annotations
    all_x = []
    max_y = 0

    # Plot each curve
    for idx, entry in enumerate(entries):
        color = cmap(idx / max(1, num - 1))
        plt.plot(entry['x'], entry['y'], linestyle='-', linewidth=2,
                 color=color, label=f"n = {entry['nval']}")
        all_x.extend(entry['x'])
        max_y = max(max_y, max(entry['y']))

    # Labels and title
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Maximum Bond Dimension", fontsize=14)
    plt.title("Bond Dimension Growth over Time", fontsize=16)

    # Ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Grid
    #plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Annotate horizontal powers of two from 2^3=8 upwards on the right
    x_max = max(all_x) if all_x else 0
    k = 3
    while 2**k <= max_y:
        y_val = 2**k
        plt.axhline(y=y_val, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        # Position label on right; n=3 slightly left to avoid overlap
        if k == 3:
            x_text = x_max * 1.03
        else:
            x_text = x_max
        plt.text(x_text, y_val * 1.02, rf"$2^{{{k}}}$", color='black', fontsize=10,
                 va='bottom', ha='right')
        k += 1

    # Legend inside with full opacity
    plt.legend(title="Number of qubits", fontsize=10, title_fontsize=12,
               loc='upper left', frameon=True, framealpha=1)

    plt.tight_layout()

    # Save and display
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()

def plot_gate_counts(path, n, bins=10):
    path = strip_trailing_dot_zero(path)
    data_dict = JSONtoDict(path)
    data_arrs, Ts, params, pool = DictToArr(data_dict, True)
    N, _, c, Δ, T, q = params
    q = int(q)
    N = int(N)
    T = round(float(T), 8)
    Δ = parse_pi_over(Δ)
    circuit_pool, sign_pool = data_arrs[0]

    # gather circuit lengths in blocks of size n
    circuit_lengths = [len(c) for c in circuit_pool]
    circuit_lengths = group_sum(circuit_lengths, n)
    experimental_length = np.mean(circuit_lengths)

    # theoretical expected length
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    te_pai = TE_PAI(hamil, q, Δ, T, N, n)
    theoretical_length = te_pai.expected_num_gates
    sigma = np.sqrt(theoretical_length)

    print(f"Theoretical length: {theoretical_length:.4f}, "
          f"Experimental length: {experimental_length:.4f}, "
          f"Rel. diff: {(theoretical_length - experimental_length)/theoretical_length:.4%}")

    # plot
    plt.figure(figsize=(8, 5))
    # histogram
    counts, bins_edges, _ = plt.hist(
        circuit_lengths,
        bins=bins,
        density=True,
        alpha=0.6,
        edgecolor='black',
        label="TE-PAI circuit lengths"
    )

    # normal pdf overlay
    x = np.linspace(bins_edges[0], bins_edges[-1], 1000)
    y = norm.pdf(x, loc=theoretical_length, scale=sigma)
    plt.plot(
        x,
        y,
        'r--',
        linewidth=2,
        label=fr'$\mathcal{{N}}(\nu_\infty,\,\sqrt{{\nu_\infty}})$'
    )

    # theoretical mean line
    plt.axvline(
        theoretical_length,
        color='gray',
        linestyle='dotted',
        linewidth=2,
        label=fr"$\nu_\infty = {theoretical_length:.2f}$"
    )

    plt.xlabel("Circuit length")
    plt.ylabel("Probability density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("TE-PAI-circuit-lengths.png", dpi=300)

def parse_pi_over(text):
    if text.startswith("pi_over_"):
        try:
            denominator = int(text.split("_")[-1])
            return np.pi / denominator
        except ValueError:
            raise ValueError("Invalid denominator in input string.")
    else:
        raise ValueError("Input must be of the form 'pi_over_<integer>'.")

def group_sum(arr, n):
    if len(arr) % n != 0:
        raise ValueError("Array length must be divisible by n.")
    return [sum(arr[i:i+n]) for i in range(0, len(arr), n)]

def draw_circuits(path, flip=True, variants=True):
    # — your existing setup —
    path = strip_trailing_dot_zero(path)
    data_dict = JSONtoDict(path)
    data_arrs, Ts, params, pool = DictToArr(data_dict, True)
    N, _, c, Δ, T, q = params
    q = int(q); N = int(N)
    T = round(float(T), 8)
    Δ = parse_pi_over(Δ)

    circuit_pool, sign_pool = data_arrs[0]
    tags   = ['PSI0', 'Z', 'RZ', 'RZZ', 'RXX', 'RYY', 'H']
    colors = [
        gate_colors[tag.lower()][0]
        if tag.lower() in gate_colors else "#888888"
        for tag in tags
    ]

    # fix to first 9 circuits in a 3x3 grid
    if variants:
        n_plots = 9
        dT = T / n_plots
        fig, axes = plt.subplots(3, 3, figsize=(12, 9))
        axes = axes.flatten()
    else:
        n_plots = 5
        fig, axes = plt.subplots(n_plots, 3, figsize=(15, 15))
        quimb = getCircuit(q, flip=flip, mps=False)

        Ts = np.linspace(0.1, 0.5, n_plots)
        Ns = [1, 4, 7, 10, 13]
        for i,t in enumerate(Ts):
            _, _, _, circuit = trotter(N=Ns[i], n_snapshot=1, T=t, q=q, compare=False, save=False, draw=False, flip=flip, mps=False)
            
            print(circuit.num_gates)

            circuit.psi.draw(
                color=tags,
                custom_colors=colors,
                layout='kamada_kawai',
                ax=axes[i][0]
            )
            axes[i][0].get_legend().remove()


    for i in range(n_plots):
        if variants:
            ax = axes[i]
            i = 10*i
            print(f"length of circuit {i}: {len(circuit_pool[i])}")
            # fresh circuit each time
            quimb = getCircuit(q, flip=flip, mps=False)
            applyGates(quimb, circuit_pool[i])

            quimb.psi.draw(
                color=tags,
                custom_colors=colors,
                layout='kamada_kawai',
                ax=ax
            )
            
        else:
            ax = axes[i][1]
            applyGates(quimb, circuit_pool[i*7])
            print(f"Random length: {quimb.num_gates}")
            quimb.psi.draw(
                color=tags,
                custom_colors=colors,
                layout='kamada_kawai',
                ax=ax
            )

            if i > 0:
                ax.get_legend().remove()

            ax = axes[i][2]
            applyGates(quimb, circuit_pool[i*7])
            quimb.amplitude_rehearse(simplify_sequence='ADCRS')['tn'].draw(
                color=tags,
                custom_colors=colors,
                layout='kamada_kawai',
                ax=ax
            )
            ax.get_legend().remove()


        # only keep legend on the first subplot
        if i > 0:
            leg = ax.get_legend()
            if leg:
                leg.remove()

        # caption as T = (i+1)*dT
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("circuitComparisons.png", dpi=600)
    #plt.show()

def show_otimization(path, indices, flip=True):
    path = strip_trailing_dot_zero(path)
    data_dict = JSONtoDict(path)
    data_arrs, Ts, params, pool = DictToArr(data_dict, True)
    N, _, c, Δ, T, q = params
    q = int(q); N = int(N)
    T = round(float(T), 8)
    Δ = parse_pi_over(Δ)
    circuit_pool, sign_pool = data_arrs[0]
    quimb = getCircuit(q, flip=flip, mps=False)
    for i in range(indices):
        applyGates(quimb, circuit_pool[i])
    qiskit = quimb_to_qiskit(quimb)
    style = {
                "displaycolor": gate_colors,
                "textcolor":      "#222222",    # global text
                "gatetextcolor":  "#FFFFFF",    # fallback gate label color
                "linecolor":      "#666666",    # qubit wire
                "creglinecolor":  "#999999",    # classical wire
                "backgroundcolor": "#FFFFFF",   # canvas
            }
    # 1) make the optimized copy
    native_gates = ['h', 'z', 'rz', 'rxx', 'ryy', 'rzz']
    qc_opt = transpile(qiskit, optimization_level=3, basis_gates=native_gates)

    # 2) set up a tall, wide figure
    fig, axes = plt.subplots(2, 1,
                            figsize=(20, 10),
                            constrained_layout=True)

    # 3a) draw the original
    qiskit.draw(output='mpl',
            ax=axes[0],
            fold=-1,      # disable pagination → one long line
            scale=0.8,
            style=style)

    # 3b) draw the optimized
    qc_opt.draw(output='mpl',
                ax=axes[1],
                fold=-1,
                scale=0.8,
                style=style)
    
    print(compare_optimization(qiskit, qc_opt))

    plt.show()

def compare_optimization(qc, qc_opt):
    # before and after
    before = Counter(qc.count_ops())
    after  = Counter(qc_opt.count_ops())

    # compute net change for every gate in either circuit
    all_gates = set(before) | set(after)
    net_diff = {g: after[g] - before[g] for g in all_gates}

    print("Net change (after - before):")
    for gate, delta in net_diff.items():
        sign = "+" if delta>0 else ""
        print(f"  {gate:6s}: {sign}{delta}")

    total_before = sum(before.values())
    total_after  = sum(after.values())
    return total_before, total_after

def calc_optimization(path, flip=True):
    path = strip_trailing_dot_zero(path)
    data_dict = JSONtoDict(path)
    data_arrs, Ts, params, pool = DictToArr(data_dict, True)
    N, _, c, Δ, T, q = params
    q = int(q); N = int(N)
    T = round(float(T), 8)
    Δ = parse_pi_over(Δ)
    circuit_pool, sign_pool = data_arrs[0]
    quimb = getCircuit(q, flip=flip, mps=False)
    for i in range(len(circuit_pool)):
        applyGates(quimb, circuit_pool[i])

    qiskit = quimb_to_qiskit(quimb)
    qc_opt = transpile(qiskit, optimization_level=3)
    compare_optimization(qiskit, qc_opt)

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