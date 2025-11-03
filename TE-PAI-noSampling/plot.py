from pathlib import Path
import re, csv, ast, os
import numpy as np
import pandas as pd
from plotting import plot_data_from_folder, plot_bond_data, plot_trotter_then_tepai, plot_gate_counts, plotTrotterPAI, plot_data_two_folders, plotMainCalc2, plotMainCalc3,plotManyCalc2, plotTrotterVsTEPAI, plot_main_calc_simple,plot_main_calc_simple_2x1
from calculations import trotter, parse, trotterThenTEPAI, organize_trotter_tepai, trotterComparison, mainCalc, manyCalc, fullCalc, mainCalc2, save_lengths
from HAMILTONIAN import Hamiltonian
from main import TE_PAI

def TEPAI_from_start(mode,T, N, n, skip_trotter = False,
                     tepaipath = f"TE-PAI-noSampling/NNN_data/circuits/N-100-n-1-p-100-Δ-pi_over_1024-q-10-dT-0.1-T-1.0", 
                     plotpath  = f"TE-PAI-noSampling/NNN_data/fullCalc/Δ-pi_over-1024-q-10-N1-100-T1-1.0-N2-100-p-100-T2-1.0-dt-0.1", 
                     ):
    if mode == 1:
        fullCalc(tepaipath, T=T, N=N, n=n, skip_trotter=skip_trotter)
    if mode == 2:
        plotMainCalc3(plotpath, justLengths=False, aligned=True)
    if mode == 3:
        fullCalc(tepaipath, T=T, N=N, n=n, skip_trotter=skip_trotter)
        plotMainCalc3(plotpath, justLengths=False, aligned=True)

def bootstrap_comparison(mode  = 0,
    tepaipath = f"TE-PAI-noSampling/2D_data/circuits/N-1000-n-1-p-10-Δ-pi_over_1024-q-12-dT-0.1-T-1", 
    trottpath = f"TE-PAI-noSampling/2D_data/plotting/lie-N-1000-T-1.0-q-12.csv",
    trottbond = f"TE-PAI-noSampling/2D_data/plotting/lie-bond-N-1000-T-1.0-q-12.csv",
):
    # Extracting parameters
    H_name = "NNN" if ("NNN_" in tepaipath) else "2D" if ("2D_" in tepaipath) else "SCH"
    N, n_snapshot, p, Δ, q, dT, T = re.search(r"N-(\d+)-n-(\d+)-p-(\d+)-Δ-([^-]+)-q-(\d+)-dT-([^-]+)-T-([^-]+)", tepaipath).groups()
    Δ = np.pi / int(Δ.split('_')[-1]) if 'pi_over' in Δ else float(Δ)
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=int(q))
    if H_name == "SCH":
        hamil = Hamiltonian.spin_chain_hamil(int(q),freqs)
    if H_name == "NNN":
        hamil = Hamiltonian.next_nearest_neighbor_hamil(int(q),freqs)
    if H_name == "2D":
        hamil = Hamiltonian.lattice_2d_hamil(int(q),freqs=freqs)

    # Getting gate-counts
    te_pai = TE_PAI(hamil, int(q), Δ, float(dT), int(N), 1)
    tepai_gates = te_pai.expected_num_gates
    trott_times = np.linspace(0, float(T), int(N))
    n = int(N) // int(n_snapshot)
    trott_gates = sum(len(hamil.get_term(t)) for t in trott_times[:n])

    # Getting bond-dimensions
    times, trott_bonds = zip(*[(float(r['x']), float(r['y'])) for r in csv.DictReader(open(trottbond))])
    trott_bonds = np.array([0]+list(trott_bonds)); times = np.array([0.0]+list(times))
    df = pd.read_csv(tepaipath+"/contraction_costs.csv")
    s = df.loc[0, "Cost"]
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    tepai_bonds = np.array([0]+[float(x) for x in nums])

    # Getting expecation values
    _, trott_vals = np.loadtxt(trottpath, delimiter=',', skiprows=1, unpack=True); trott_vals=np.array([1]+list(trott_vals))
    params = os.path.basename(tepaipath)
    tokens = params.split('-')
    pairs = {}
    if len(tokens) % 2 == 0:
        pairs = {tokens[i]: tokens[i+1] for i in range(0, len(tokens), 2)}
    else:
        raise ValueError("unexpected parameter format: " + params)
    if 'T' in pairs:
        pairs['T'] = f"{float(pairs['T']):.1f}"
    order = ['N', 'n', 'p', 'Δ', 'T', 'q', 'dT']
    filename = '-'.join(f"{k}-{pairs[k]}" for k in order if k in pairs) + ".csv"
    plot_dir = os.path.join(os.path.dirname(os.path.dirname(tepaipath)), "plotting")
    csv_path = os.path.join(plot_dir, filename)
    _, tepai_vals, tepai_stds = np.loadtxt(csv_path, delimiter=',', skiprows=1, unpack=True)

    print(f"TEPAI gates: {tepai_gates}")
    print(f"Trotter gates: {trott_gates}")
    print(f"TEPAI bonds: {tepai_bonds}")
    print(f"Trotter bonds: {trott_bonds}")
    print(f"Times: {times}")
    print("----------------")
    print(f"trott_vals: {trott_vals}")
    print(f"tepai_vals: {tepai_vals}")
    print(f"tepai_stds: {tepai_stds}")

    if mode == 0:
        plot_main_calc_simple(tepai_len_per_dt  = tepai_gates, 
                            trotter_len_per_dt= trott_gates,
                            times             = times,
                            trotter_bonds     = trott_bonds,
                            tepai_bonds       = tepai_bonds,
                            trotter_vals      = trott_vals,
                            tepai_vals        = tepai_vals,
                            tepai_stds        = tepai_stds
                            )
    if mode == 1:
        plot_main_calc_simple_2x1(tepai_len_per_dt  = tepai_gates, 
                            trotter_len_per_dt= trott_gates,
                            times             = times,
                            trotter_bonds     = trott_bonds,
                            tepai_bonds       = tepai_bonds,
                            trotter_vals      = trott_vals,
                            tepai_vals        = tepai_vals,
                            tepai_stds        = tepai_stds
                            )


def Trotter_then_TEPAI(mode, n=10, q=10, tepai_dT=0.1, n1=30, n2=10, N1=300, N2=100, Δ=np.pi/(2**10), NNN=True,
        tepaipath = f"TE-PAI-noSampling/NNN_data/circuits/N-100-n-1-p-100-Δ-pi_over_1024-q-10-dT-0.1-T-1.0", 
        plotpath  = f"TE-PAI-noSampling/NNN_data/trotterThenTEPAI/Δ-pi_over-4096-q-10-N1-300-T1-3.0-N2-100-p-100-T2-4.0-dt-0.1", 
):
    if mode == 1:
        mainCalc2(tepaiPath=tepaipath, finalT1=3, N1=300, n1=40, finalT2=4, confirm=True, flip=True)
    if mode == 2:
        save_lengths(n,q,tepai_dT,n1,n2,N1,N2,Δ,NNN,base_dir=plotpath)
        plotMainCalc3(plotpath, justLengths=False, aligned=False)
    if mode == 3:
        mainCalc2(tepaiPath=tepaipath, finalT1=3, N1=300, n1=40, finalT2=4, confirm=True, flip=True)
        save_lengths(n,q,tepai_dT,n1,n2,N1,N2,Δ,NNN,base_dir=plotpath)
        plotMainCalc3(plotpath, justLengths=False, aligned=False)

def TEPAI_simulation(folder_path="TE-PAI-noSampling/data/plotting"):
    plot_data_from_folder(folder_path)


if __name__ == "__main__":
    bootstrap_comparison(mode = 1)
    #TEPAI_simulation(folder_path="TE-PAI-noSampling/2D_data/plotting")

