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
from scipy.stats import binom
from plotting import calcOverhead
import math
from fractions import Fraction
from typing import Tuple, Optional
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
# --------------------------- Plotting (agnostic) --------------------------- #
def plot_results(Ts, results, q=None, delta=None, tuples=None, res=None, resample_stats=None, lie_csv_path=None, save_path='results_vs_time.png', title='Measurement vs Time'):
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
    # Time vector defensivenesst
    Ts = np.asarray(Ts, dtype=float)
    Ts = Ts[:max_len]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # --- Left subplot: runs + mean (+ optional LIE reference) ---
    ax = axes[0]
    if False:
        for r in arr:
            valid = ~np.isnan(r)
            ax.plot(Ts[valid], r[valid], marker='o', linewidth=0.5, linestyle='--', markersize=2, alpha=0.6, color="gray")
    
    # --- Right subplot: standard error over time ---
    ax2 = axes[1]
    times = Ts[:len(se_values)]
    if q is not None and delta is not None:
        overhead = [calcOverhead(q, T, delta) / np.sqrt(n_runs[0]) for T in times]
        
        ax2.plot(times, overhead, 'r--', linewidth=2, label='Theoretical Overhead / √shots')
        upper = mean_values + overhead
        lower = mean_values - overhead
        ax.fill_between(
            Ts[:len(mean_values)],
            lower,
            upper,
            color='red',
            alpha=0.15,
            zorder=1,
            label='Theoretical uncertainty band',
        )

        # dashed red boundary lines
        ax.plot(Ts[:len(mean_values)], upper, '--', color='red', alpha=0.6, linewidth=1.5, zorder=1)
        ax.plot(Ts[:len(mean_values)], lower, '--', color='red', alpha=0.6, linewidth=1.5, zorder=1)
    ax.errorbar(
        Ts[:len(mean_values)],
        mean_values,
        yerr=np.asarray(se_values, dtype=float),
        fmt='-',
        linewidth=2,
        label='TE-PAI',
        color='tab:green'
    )
    # Optional LIE reference
    if lie_csv_path and os.path.exists(lie_csv_path):
        ref = pd.read_csv(lie_csv_path)
        if {'x', 'y'}.issubset(ref.columns):
            ax.plot(ref['x'].values, ref['y'].values, 'k-', linewidth=2, label='Trotterization')
    if resample_stats is not None:
                m_boot = np.asarray(resample_stats.get('mean', []), dtype=float)
                s_boot = np.asarray(resample_stats.get('std',  []), dtype=float)
                lbl    = resample_stats.get('label', 'Resample (CSV)')
                k = min(len(Ts), len(m_boot), len(s_boot))
                if k > 0:
                    ax.errorbar(Ts[:k], m_boot[:k], yerr=s_boot[:k], fmt='o', markersize=3,
                                capsize=2, alpha=0.9, color="tab:green")
    ax.set_xlabel('Time')
    ax.set_ylabel('Measurement')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small', ncol=2)
    ax2.plot(times, se_values, '-', linewidth=2, label='Standard Error', color="tab:green")
    
    t1 = np.array(Ts[:len(mean_values)])
    t2 = np.array(ref['x'].values)
    interp_mean = np.interp(t2, t1, np.array(mean_values))
    ax2.plot(t2, abs(interp_mean-np.array(ref['y'].values)), label="Absolute error", color="tab:orange")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Standard Error')
    ax2.set_title('Standard Error vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
def _ragged_to_2d(list_of_lists, fill_value=np.nan):
    """
    Convert list-of-lists with variable lengths into a 2D (runs x timesteps) array, padded with NaN.
    """
    max_len = max(len(x) for x in list_of_lists)
    arr = np.full((len(list_of_lists), max_len), fill_value, dtype=float)
    for i, x in enumerate(list_of_lists):
        arr[i, :len(x)] = np.asarray(x, dtype=float)
    return arr
def resample_from_weights_results(weights_2d: np.ndarray,
                                results_2d: np.ndarray,
                                n_bootstrap: int = 10_000,
                                sample_size: int = 1_000,
                                per_pair_rep: int = 10000,
                                random_state: int | None = None):
    """
    Cannon-style resampling adapted to (weights, results) arrays.
    For each timestep t:
    - Build the base vector s_t by taking the observed weighted outcomes at that timestep
        (i.e., results_2d[:, t]; these already equal weight * measured).
    - Replicate each entry per_pair_rep times (matching cannon's 'size=100').
    - Draw (with replacement) `sample_size` values to form one bootstrap sample,
        repeat `n_bootstrap` times, and take the mean per bootstrap sample.
    - Return per-timestep (mean_of_bootstrap_means, std_of_bootstrap_means).
    Notes:
    * This matches the cannon algorithms effect when (c, p) reduces to deterministic
        ±c per observation (as in our stored weighted outcomes). If your 'results' are
        *unweighted*, replace `v = r_t` by `v = w_t * r_t` below.
    """
    rng = np.random.default_rng(random_state)
    n_runs, n_t = results_2d.shape
    means = np.full(n_t, np.nan, dtype=float)
    stds  = np.full(n_t, np.nan, dtype=float)
    for t in range(n_t):
        r_t = results_2d[:, t]
        w_t = weights_2d[:, t]
        mask = (~np.isnan(r_t)) & (~np.isnan(w_t))
        if not np.any(mask):
            continue
        # Observed weighted outcomes at this timestep (already weight * measured in your CSV)
        v = r_t[mask]  # if your 'results' are unweighted, use: v = w_t[mask] * r_t[mask]
        # Cannon-style replication (size=100 per pair)
        s = np.repeat(v, per_pair_rep)
        # Bootstrap: (n_bootstrap × sample_size) indices
        idx = rng.integers(0, s.size, size=(n_bootstrap, sample_size))
        samples = s[idx]
        boot_means = samples.mean(axis=1)
        means[t] = boot_means.mean()
        stds[t]  = boot_means.std(ddof=1)
    return means, stds
# -------------------- Compute flow (original logic wrapped) ----------------- #
def compute_and_save(
    folder="TE-PAI-noSampling/data/circuits/N-100-n-1-p-10000-Δ-pi_over_64-q-20-dT-0.2-T-2",
    n_runs_to_plot=100,
    out_dir="TE-PAI-noSampling/data/many-circuits",
    csv_basename=None,
    lie_csv_path=None,
    save_plot_path='results_vs_time.png',
    gam_list=None,
    plot=True,
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
    tuples  = [[] for _ in range(n_timesteps)]
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
            gamma_factor = gam_list[time_idx+1]
            weight = current_sign * gamma_factor
            # save weighted measurement
            results[run_idx].append(measured*weight)
            weights.append(weight)
            tuples[time_idx].append((weight, measured))
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
    if plot:
        plot_results(Ts, results, tuples=None, lie_csv_path=lie_csv_path, save_path=save_plot_path,
                    title='Measurement vs Time (computed)')
    return Ts, results, csv_path

def compute_and_save_parallel(
    folder="TE-PAI-noSampling/data/circuits/N-100-n-1-p-10000-Δ-pi_over_64-q-20-dT-0.2-T-2",
    n_runs_to_plot=100,
    out_dir="TE-PAI-noSampling/data/many-circuits",
    csv_basename=None,
    lie_csv_path=None,
    save_plot_path='results_vs_time.png',
    gam_list=None,
    max_workers=None,  # optional, None -> default to os.cpu_count()-1
    plot=True,
):
    """
    Parallel version of compute_and_save with progress prints.
    Same outputs/CSV/plotting as the serialized version.
    Returns (Ts, results, csv_path).
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
            raise ValueError(
                f"gam_list length ({len(gam_list)}) must equal number of measurement times len(Ts) ({len(Ts)})."
            )
    # ALL RUNS
    if n_runs_to_plot == None:
        n_runs_to_plot = len(indices_all)
    # Prepare containers (match serialized structure exactly)
    results       = [[] for _ in range(n_runs_to_plot)]
    tuples        = [[] for _ in range(n_timesteps)]
    kept_indices  = [None] * n_runs_to_plot
    kept_signs    = [None] * n_runs_to_plot
    kept_weights  = [None] * n_runs_to_plot
    # Launch workers per run
    selected = [[int(ii) for ii in run_indices] for run_indices in indices_all[:n_runs_to_plot]]
    # Log the same "Starting run ..." lines deterministically before compute
    for run_idx, run_indices in enumerate(selected):
        signs = [sign_pool[idx] for idx in run_indices]
        print(f"Starting run {run_idx + 1} with signs: {signs}", flush=True)
    total = len(selected)
    workers = (os.cpu_count()-1 if max_workers is None else max_workers)
    print(f"[Setup] Launching {total} runs across up to {workers} worker processes "
        f"(timesteps/run={n_timesteps}, dT={dT}).", flush=True)
    start_time = time.time()
    step = max(1, total // 10)  # print about every 10% of runs
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # keep run index so we can store results in order even if completion is out-of-order
        future_to_idx = {
            ex.submit(_compute_single_run, run_indices, q, [circuit_pool[index] for index in run_indices], [sign_pool[index] for index in run_indices], gam_list): i
            for i, run_indices in enumerate(selected)
        }
        completed = 0
        try:
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                run_indices, signs, weights, results_run, tuples_contrib = fut.result()
                # Save per-run artifacts (preserve ordering)
                kept_indices[i] = run_indices
                kept_signs[i]   = signs
                kept_weights[i] = weights
                results[i]      = results_run
                # Aggregate tuples across runs into per-time buckets
                for t_idx, wm in enumerate(tuples_contrib):
                    tuples[t_idx].append(wm)
                # Progress print
                completed += 1
                if (completed % step == 0) or (completed == total):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else float('inf')
                    remaining = total - completed
                    eta = (remaining / rate) if rate > 0 else float('inf')
                    print(f"[Progress] Completed {completed}/{total} runs "
                        f"({completed/total:.0%}). Elapsed {elapsed:.1f}s, ETA {eta:.1f}s.",
                        flush=True)
        except KeyboardInterrupt:
            print("\n[Abort] KeyboardInterrupt received. Cancelling remaining tasks...", flush=True)
            ex.shutdown(cancel_futures=True)
            raise
        except Exception as e:
            print(f"\n[Error] A worker raised an exception: {e}", flush=True)
            ex.shutdown(cancel_futures=True)
            raise
    # Save CSV (lists are JSON-encoded strings), identical schema & filename pattern
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
    print(f"Saved runs to: {csv_path}", flush=True)
    # Plot (agnostic)
    if plot:
        plot_results(Ts, results, tuples, lie_csv_path=lie_csv_path, save_path=save_plot_path,
                    title='Measurement vs Time (computed)')
    total_time = time.time() - start_time
    print(f"[Done] All {total} runs finished in {total_time:.1f}s.", flush=True)
    return Ts, results, csv_path
def _compute_single_run(run_indices, q, circuits, signs, gam_list):
    """
    Compute a single run: apply gates across all timesteps, collect measurements,
    weights, and per-timestep (weight, measured) tuples for aggregation.
    Returns: (run_indices, signs, weights, results_run, tuples_contrib)
    """
    quimb = calc.getCircuit(int(q), True)
    results_run = []
    weights     = []
    tuples_contrib = []  # list of (weight, measured) per time index
    # initial t=0 "measurement"
    if gam_list is not None:
        results_run.append(gam_list[0])
        weights.append(gam_list[0])
    else:
        results_run.append(1.0)
        weights.append(1.0)
    current_sign = +1
    for time_idx, (circuit, sign) in enumerate(zip(circuits, signs)):
        current_sign *= sign
        calc.applyGates(quimb, circuit)
        measured = calc.measure(quimb, int(q), None)
        gamma_factor = gam_list[time_idx + 1] if gam_list is not None else 1.0
        weight = current_sign * gamma_factor
        results_run.append(measured*weight)
        weights.append(weight)
        tuples_contrib.append((weight, measured))
    return run_indices, signs, weights, results_run, tuples_contrib

def get_gam_list(folder):
    """Compute the gam_list for given parameters."""
    params = dict(re.findall(r'([A-Za-zΔ]+)-([A-Za-z0-9_.]+)', folder))
    N = int(params["N"])
    dT = float(params["dT"])
    T = float(params["T"])
    q = int(params["q"])
    n_snap = int(T / dT)
    N = N*n_snap # = 10 so instead we multiply by 5 and got much better results!
    v = params["Δ"]
    Δ = np.pi / float(v.split('_')[-1]) if v.startswith('pi_over_') else float(v)

    #N=int(N/5); T=dT; n_snap=1; 
    print(f"N:{N}, dT:{dT}, T:{T}, q:{q}, n_snap:{n_snap}, v:{v}, Δ:{Δ}")

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


def gam_list_for_stitched_time_indep(folder):
    params = dict(re.findall(r'([A-Za-zΔ]+)-([A-Za-z0-9_.]+)', folder))
    N = int(params["N"])
    dT = float(params["dT"])
    T = float(params["T"])
    q = int(params["q"])
    n_snap = int(T / dT)
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    v = params["Δ"]
    Δ = np.pi / float(v.split('_')[-1]) if v.startswith('pi_over_') else float(v)

    # number of stitched segments, robust to float noise
    n_snap = round(T / dT)
    dT_eff = T / n_snap  # make total exactly T

    N_seg = N

    N_total = n_snap * N_seg
    angles_layer = [2 * abs(c) * T / N_total for c in hamil.coefs(0.0)]
    gamma_per_layer = pai.gamma(angles_layer, Δ)
    gam_list = [1.0] + [gamma_per_layer ** ((i+1) * N_seg) for i in range(n_snap)]
    return gam_list

# ----------------------- Plot directly from a saved CSV --------------------- #
def _parse_T_and_dT_from_name(path: str, all=False) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[float]]:
    """
    Parse dT and T as before, optionally parse q (int) and Δ which is expected as 'pi over N'
    (e.g. 'Δ-pi_over_64' or 'Δ-pi/64'). Returns (dT, T, q, delta).
    """
    name = os.path.basename(path)
    m_dT = re.search(r"dT-([0-9]*\.?[0-9]+)", name)
    m_T  = re.search(r"T-([0-9]*\.?[0-9]+)", name)
    m_q  = re.search(r"\bq-([0-9]+)\b", name)
    # match Δ-<pi_over_N> or Δ-pi/N (accepts Δ or 'delta', case-insensitive)
    m_delta_den = re.search(
        r"(?:Δ|delta)-(?:pi(?:_over_|/|_)?)(\d+)",
        name,
        flags=re.IGNORECASE,
    )
    dT = float(m_dT.group(1)) if m_dT else None
    T  = float(m_T.group(1))  if m_T  else None
    q  = int(m_q.group(1))    if m_q else None
    delta = None
    if m_delta_den:
        try:
            den = int(m_delta_den.group(1))
            if den != 0:
                delta = float(np.pi) / den
        except (ValueError, ZeroDivisionError):
            delta = None
    if not all:
        return dT, T
    return dT, T, q, delta
def plot_from_csv(csv_path, lie_csv_path=None, save_plot_path='results_vs_time.png'):
    """
    Reads a previously saved CSV of (run, indices, signs, results, weights), reconstructs the runs,
    rebuilds a time axis from filename-encoded dT/T if available (fallback: time steps),
    and calls the common plotting routine with an optional resampling overlay.
    """
    df = pd.read_csv(csv_path)
    if not {'run', 'indices', 'signs', 'results'}.issubset(df.columns):
        raise ValueError("CSV must have columns: run, indices, signs, results")
    if 'weights' not in df.columns:
        raise ValueError("CSV must have a 'weights' column for resampling with weights.")
    # Reconstruct per-run lists
    results_ll = []
    for s in df['results'].astype(str).tolist():
        arr = json.loads(s)
        results_ll.append([float(x) for x in arr])
    weights_ll = []
    for s in df['weights'].astype(str).tolist():
        arr = json.loads(s)
        weights_ll.append([float(x) for x in arr])
    gammas = [1.0, 1.1784691677652317, 1.3887895793732774, 1.636645699805052, 1.9287364957758042, 2.2729564930153385, 2.678609146690363, 3.156658291868529, 3.7200244701375227, 4.383934141389263, 5.166331219140593]
    gammas = [1.0, 1.170123452869955, 1.3691888949563058, 1.6021200373974709, 1.87467823007167, 2.1936049635915977, 2.566788614230473, 3.003459556070648, 3.5144184663046483, 4.112303470622327, 4.811902736293693]
    gammas = GAM_LIST
    for i,results in enumerate(results_ll):
        for j,result in enumerate(results):
            results_ll[i][j] = ((result/np.abs(weights_ll[i][j]))*gammas[j])
    # Convert to aligned 2D arrays (runs × timesteps)
    R = _ragged_to_2d(results_ll)  # weighted outcomes per spec (weight * measured)
    W = _ragged_to_2d(weights_ll)
    # Time vector from filename if possible
    dT, T, q, delta = _parse_T_and_dT_from_name(csv_path, all=True)
    max_len = R.shape[1]
    if dT is not None and T is not None:
        Ts = np.arange(0, max_len * dT, dT)[:max_len]
    else:
        Ts = np.arange(max_len, dtype=float)
    # Cannon-style resampling on the CSV-backed arrays
    #boot_mean, boot_std = resample_from_weights_results(W, R,n_bootstrap=10_000,sample_size=10_000,per_pair_rep=100,random_state=None)
    # Standard plot + overlay the errorbar from resampling
    plot_results(
        Ts, results_ll, q=q, delta=delta, lie_csv_path=lie_csv_path, save_path=save_plot_path,
        #resample_stats={'mean': boot_mean, 'std': boot_std, 'label': 'Resample (CSV)'},
        title='Measurement vs Time (from CSV)',
    )
if __name__ == "__main__":
    
    # Config
    MODE            = "compute"   # "compute" or "from_csv"
    #FOLDER          = "TE-PAI-noSampling/data/circuits/N-100-n-1-p-10000-Δ-pi_over_256-q-20-dT-0.2-T-2"
    FOLDER          = "TE-PAI-noSampling/data/circuits/N-100-n-1-p-1000-Δ-pi_over_256-q-20-dT-0.5-T-5"
    #FOLDER          = "TE-PAI-noSampling/data/circuits/N-100-n-1-p-100000-Δ-pi_over_64-q-20-dT-0.2-T-2"

    GAM_LIST        = get_gam_list(FOLDER)
    SHOTS           = 100

    print([float(g) for g in GAM_LIST])

    OUT_DIR         = "TE-PAI-noSampling/data/many-circuits"
    CSV_BASENAME    = None
    #LIE_CSV         = "TE-PAI-noSampling/data/plotting/lie-N-1000-T-2-q-20.csv"
    LIE_CSV         = "TE-PAI-noSampling/data/plotting/lie-N-1000-T-5-q-20.csv"
    OUT_PNG         = "results_vs_time.png"
    CSV_PATH        = "TE-PAI-noSampling/data/many-circuits/runs-N-100-n-1-p-100000-Δ-pi_over_64-q-20-dT-0.2-T-2.csv"
    #CSV_PATH        = "TE-PAI-noSampling/data/many-circuits/runs-N-100-n-1-p-10000-Δ-pi_over_256-q-20-dT-0.2-T-2.csv"
    MAX_WORKERS     = None
    N_RUNS          = None

    if MODE == "compute":
        compute_and_save_parallel(
            folder=FOLDER,
            n_runs_to_plot=N_RUNS,
            out_dir=OUT_DIR,
            csv_basename=CSV_BASENAME,
            lie_csv_path=LIE_CSV,
            save_plot_path=OUT_PNG,
            gam_list=GAM_LIST,
            max_workers=MAX_WORKERS,
            plot=False,
        )
    elif MODE == "from_csv":
        plot_from_csv(
            csv_path=CSV_PATH,
            lie_csv_path=LIE_CSV,
            save_plot_path=OUT_PNG
        )
    else:
        raise ValueError("MODE must be 'compute' or 'from_csv'.")