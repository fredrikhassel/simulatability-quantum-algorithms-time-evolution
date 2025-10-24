"""
Miscellaneous helper functions and CLI entry point.
"""
import os
import csv
import shutil
from pathlib import Path
import ast
from qiskit import QuantumCircuit, transpile
import re
import pandas as pd
#from calculations import JSONtoDict, HDF5toDict, DictToArr, getSucessive, getPool, extract_dT_value, trotter
#from plotting import plot_data_from_folder
import numpy as np
import matplotlib.pyplot as plt

# I/O & data‐saving routines
def saveData(N, n_snapshot, circuits_count, delta_name, Ts, q, dT, averages, stds, char, NNN):
    # Define the output directory
    if not NNN:
        output_dir = os.path.join('TE-PAI-noSampling', 'data', 'plotting')
    else:
        output_dir = os.path.join('TE-PAI-noSampling', 'NNN_data', 'plotting')
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

# Lightweight converters & path utilities
def strip_trailing_dot_zero(folder_name):
    if '-T-' in folder_name:
        parts = folder_name.split('-T-')
        head = parts[0]
        tail = parts[1]
        if tail.endswith('.0'):
            tail = tail[:-2]  # remove the .0
        return f"{head}-T-{tail}"
    return folder_name

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

# Simulation orchestrators & CLI entrypoints
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

# Help wrappers around core routines
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



