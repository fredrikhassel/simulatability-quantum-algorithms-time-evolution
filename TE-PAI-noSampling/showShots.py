import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calculations as calc
import PAI as pai
from HAMILTONIAN import Hamiltonian


# --------------------------- Plotting (agnostic) --------------------------- #
def plot_results(Ts, results, lie_csv_path=None, save_path='results_vs_time.png', title='Measurement vs Time'):
    """
    Plot per-run trajectories + mean (left) and standard error over time (right).
    Ts: 1D array-like of time points (length must match each run's length).
    results: list[list[float]] where each inner list is a run across time.
    lie_csv_path: optional path to a CSV with columns 'x' and 'y' for LIE reference.
    """
    # Robust ragged handling (pad with NaN if needed)
    max_len = max(len(r) for r in results)
    arr = np.full((len(results), max_len), np.nan, dtype=float)
    for i, r in enumerate(results):
        arr[i, :len(r)] = np.asarray(r, dtype=float)

    # Mean and SE across runs per time index
    mean_values = np.nanmean(arr, axis=0)
    n_runs = np.sum(~np.isnan(arr), axis=0)
    std_values = np.nanstd(arr, axis=0, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        se_values = std_values / np.sqrt(n_runs)

    # Time vector defensiveness
    Ts = np.asarray(Ts, dtype=float)
    Ts = Ts[:max_len]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left subplot: runs + mean (+ optional LIE reference) ---
    ax = axes[0]
    for r in arr:
        valid = ~np.isnan(r)
        ax.plot(Ts[valid], r[valid], marker='o', linewidth=0.5, linestyle='--', markersize=2, alpha=0.6)
    ax.plot(Ts[:len(mean_values)], mean_values, '-', linewidth=2, label='Mean of runs')

    # Optional LIE reference
    if lie_csv_path and os.path.exists(lie_csv_path):
        ref = pd.read_csv(lie_csv_path)
        if {'x', 'y'}.issubset(ref.columns):
            ax.plot(ref['x'].values, ref['y'].values, 'k-', linewidth=2, label='LIE reference')

    ax.set_xlabel('Time')
    ax.set_ylabel('Measurement')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small', ncol=2)

    # --- Right subplot: standard error over time ---
    ax2 = axes[1]
    ax2.plot(Ts[:len(se_values)], se_values, '-', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Standard Error')
    ax2.set_title('Standard Error vs Time')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


# -------------------- Compute flow (original logic wrapped) ----------------- #
def compute_and_save(
    folder="TE-PAI-noSampling/data/circuits/N-100-n-1-p-10000-Δ-pi_over_64-q-20-dT-0.2-T-2",
    n_runs_to_plot=100,
    out_dir="TE-PAI-noSampling/data/many-circuits",
    csv_basename=None,
    lie_csv_path=None,
    save_plot_path='results_vs_time.png',
    gam_list=None,
):
    """
    Runs the original computation, saves results to CSV (run, indices, signs, weights, results),
    and produces the plot. Returns (Ts, results, csv_path).
    If gam_list is provided it must have length == len(Ts) and will be used to weight measurements.
    """
    # Load & unpack
    data_dict = calc.JSONtoDict(folder)
    data_arrs, Ts_input, params, pool = calc.DictToArr(data_dict, True)
    N, n, c, Δ, T, q = params
    q = int(q)
    T = float(T)
    dT = calc.extract_dT_value(folder)

    circuit_pool, sign_pool = data_arrs[0]
    n_timesteps = round(T / dT)
    n_circuits = int(len(sign_pool) / n_timesteps)
    indices_all = calc.generate_random_indices(len(circuit_pool), n_circuits, n_timesteps)
    Ts = np.arange(0, T + dT, dT)

    # Validate gam_list if provided
    if gam_list is not None:
        if len(gam_list) != len(Ts):
            raise ValueError(f"gam_list length ({len(gam_list)}) must equal number of measurement times len(Ts) ({len(Ts)}).")

    # Initialize containers
    results = [[] for _ in range(n_runs_to_plot)]
    kept_indices = []
    kept_signs = []
    kept_weights = []

    for run_idx, run_indices in enumerate(indices_all[:n_runs_to_plot]):
        run_indices = [int(ii) for ii in run_indices]
        quimb = calc.getCircuit(q, True)
        circuits = [circuit_pool[idx] for idx in run_indices]
        signs = [sign_pool[idx] for idx in run_indices]
        print(f"Starting run {run_idx + 1} with signs: {signs}")

        # set initial measurement at t=0
        if gam_list is not None:
            results[run_idx].append(gam_list[0])
            weights = [gam_list[0]]
        else:
            results[run_idx].append(1.0)
            weights = [1.0]

        current_sign = +1
        for time_idx, (circuit, sign) in enumerate(zip(circuits, signs)):
            # update sign (stochastic sign flips)
            current_sign = current_sign * sign

            # apply gates and measure
            calc.applyGates(quimb, circuit)
            #bond, cost = calc.getComplexity(quimb)
            measured = calc.measure(quimb, q, None)

            # determine gamma for this measurement: measurement after j-th circuit corresponds to index j+1
            gamma_factor = 1.0 if gam_list is None else gam_list[time_idx + 1]
            weight = current_sign * gamma_factor

            # save weighted measurement
            results[run_idx].append(measured * weight)
            weights.append(weight)

        kept_indices.append(run_indices)
        kept_signs.append(signs)
        kept_weights.append(weights)

    # Save CSV (lists are JSON-encoded strings)
    os.makedirs(out_dir, exist_ok=True)
    if csv_basename is None:
        tail = os.path.basename(folder.rstrip("/\\"))
        csv_basename = f"runs-{tail}.csv"
    csv_path = os.path.join(out_dir, csv_basename)

    rows = []
    for i in range(n_runs_to_plot):
        rows.append({
            "run": i + 1,
            "indices": json.dumps(kept_indices[i]),
            "signs": json.dumps(kept_signs[i]),
            "weights": json.dumps(kept_weights[i]),
            "results": json.dumps(results[i]),
        })
    pd.DataFrame(rows, columns=["run", "indices", "signs", "weights", "results"]).to_csv(csv_path, index=False)
    print(f"Saved runs to: {csv_path}")

    # Plot (agnostic)
    plot_results(Ts, results, lie_csv_path=lie_csv_path, save_path=save_plot_path,
                 title='Measurement vs Time (computed)')

    return Ts, results, csv_path

def get_gam_list(folder):
    """Compute the gam_list for given parameters."""
    FOLDER = "TE-PAI-noSampling/data/circuits/N-100-n-1-p-10000-Δ-pi_over_64-q-20-dT-0.2-T-2"
    params = dict(re.findall(r'([A-Za-zΔ]+)-([A-Za-z0-9_.]+)', FOLDER))

    N = int(params["N"])
    dT = float(params["dT"])
    T = float(params["T"])
    q = int(params["q"])
    n_snap = int(T / dT)
    v = params["Δ"]
    Δ = np.pi / float(v.split('_')[-1]) if v.startswith('pi_over_') else float(v)

    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    steps = np.linspace(0, T, N)
    angles = [[2 * np.abs(coef) * T / N for coef in hamil.coefs(t)] for t in steps]
    n = int(N / n_snap)
    gam_list = [1] + [
            np.prod([pai.gamma(angles[j], Δ) for j in range((i + 1) * n)])
            for i in range(n_snap)
        ]
    return gam_list


# ----------------------- Plot directly from a saved CSV --------------------- #
def _parse_T_and_dT_from_name(path):
    """Attempt to parse T and dT from filename (e.g., ...dT-0.2-T-2...). Returns (dT, T) or (None, None)."""
    name = os.path.basename(path)
    m_dT = re.search(r"dT-([0-9]*\.?[0-9]+)", name)
    m_T = re.search(r"T-([0-9]*\.?[0-9]+)", name)
    dT = float(m_dT.group(1)) if m_dT else None
    T = float(m_T.group(1)) if m_T else None
    return dT, T


def plot_from_csv(csv_path, lie_csv_path=None, save_plot_path='results_vs_time.png'):
    """
    Reads a previously saved CSV of (run, indices, signs, results), reconstructs the runs,
    rebuilds a time axis from filename-encoded dT/T if available (fallback: time steps),
    and calls the common plotting routine.
    """
    df = pd.read_csv(csv_path)
    if not {'run', 'indices', 'signs', 'results'}.issubset(df.columns):
        raise ValueError("CSV must have columns: run, indices, signs, results")

    # Reconstruct results as lists of floats
    results = []
    for s in df['results'].astype(str).tolist():
        arr = json.loads(s)
        results.append([float(x) for x in arr])

    # Time vector from filename if possible
    dT, T = _parse_T_and_dT_from_name(csv_path)
    max_len = max(len(r) for r in results)
    if dT is not None and T is not None:
        # Keep consistent with length from data (T might imply n_timesteps+1)
        Ts = np.arange(0, (max_len) * dT, dT)[:max_len]
    else:
        Ts = np.arange(max_len, dtype=float)  # fallback to time steps

    plot_results(Ts, results, lie_csv_path=lie_csv_path, save_path=save_plot_path,
                 title='Measurement vs Time (from CSV)')



if __name__ == "__main__":
    
    MODE = "compute"   # set to "compute" or "from_csv"
    FOLDER = "TE-PAI-noSampling/data/circuits/N-100-n-1-p-10000-Δ-pi_over_64-q-20-dT-0.2-T-2"
    GAM_LIST = get_gam_list(FOLDER)
    N_RUNS = 30
    OUT_DIR = "TE-PAI-noSampling/data/many-circuits"
    CSV_BASENAME = None  # or e.g., "runs-N-100-...csv"
    LIE_CSV = "TE-PAI-noSampling/data/plotting/lie-N-100-T-2-q-20.csv"
    OUT_PNG = "results_vs_time.png"

    CSV_PATH = "TE-PAI-noSampling/data/many-circuits/runs-old-N-100-n-1-p-10000-Δ-pi_over_64-q-20-dT-0.2-T-2.csv"

    if MODE == "compute":
        compute_and_save(
            folder=FOLDER,
            n_runs_to_plot=N_RUNS,
            out_dir=OUT_DIR,
            csv_basename=CSV_BASENAME,
            lie_csv_path=LIE_CSV,
            save_plot_path=OUT_PNG,
            gam_list=GAM_LIST,
        )
    elif MODE == "from_csv":
        plot_from_csv(
            csv_path=CSV_PATH,
            lie_csv_path=LIE_CSV,
            save_plot_path=OUT_PNG
        )
    else:
        raise ValueError("MODE must be 'compute' or 'from_csv'.")

