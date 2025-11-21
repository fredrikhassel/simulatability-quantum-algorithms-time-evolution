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
from typing import Tuple, Optional, NamedTuple, List
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# API For TEPAI-execution
class MeasurementResults(NamedTuple):
    """Container for per-timestep measurement stats and metadata."""
    Ts: np.ndarray
    unweighted_runs: List[List[float]]
    unweighted_mean: np.ndarray
    unweighted_std: np.ndarray
    weighted_runs: List[List[float]]
    weighted_mean: np.ndarray
    weighted_std: np.ndarray
    csv_path: str

def _load_runs_csv(csv_path: str) -> tuple[list[list[float]], list[list[float]], np.ndarray]:
    """Load weighted runs and weights from a saved CSV and rebuild the time axis."""
    df = pd.read_csv(csv_path)
    if not {'results', 'weights'}.issubset(df.columns):
        raise ValueError("CSV must include 'results' and 'weights' columns.")
    weighted_runs = [[float(x) for x in json.loads(s)] for s in df['results'].astype(str)]
    weights_runs  = [[float(x) for x in json.loads(s)] for s in df['weights'].astype(str)]
    R = _ragged_to_2d(weighted_runs)
    dT, T, _, _ = _parse_T_and_dT_from_name(csv_path, all=True)
    max_len = R.shape[1]
    Ts = np.arange(0, max_len * dT, dT)[:max_len] if (dT is not None and T is not None) else np.arange(max_len, dtype=float)
    return weighted_runs, weights_runs, Ts

def _runs_to_unweighted(weighted_runs: list[list[float]], weights_runs: list[list[float]]) -> list[list[float]]:
    """Convert weighted runs to unweighted runs by per-step division, robust to zero weights."""
    out: list[list[float]] = []
    for wrun, wts in zip(weighted_runs, weights_runs):
        run = []
        for r, w in zip(wrun, wts):
            run.append(r / w if w != 0 else np.nan)
        out.append(run)
    return out

def _mean_std_ragged(ll: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-timestep mean and std over ragged runs, using ddof=1 where count>=2 else 0."""
    A = _ragged_to_2d(ll)
    means = np.nanmean(A, axis=0)
    cnts  = np.sum(~np.isnan(A), axis=0).astype(float)
    diffs = A - means
    diffs[np.isnan(diffs)] = 0.0
    ss    = np.sum(diffs * diffs, axis=0)
    denom = np.where(cnts >= 2, cnts - 1.0, cnts)  # 1->0 (std=0); 0->0 (std=nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        vars_ = np.where(denom > 0, ss / denom, np.nan)
    stds = np.sqrt(vars_)
    stds = np.where(cnts == 1, 0.0, stds)
    return means, stds

def compute_and_save_parallel_ext(
    folder: str,
    n_runs_to_plot: int | None,
    out_dir: str,
    csv_basename: str | None = None,
    gam_list: list[float] | None = None,
    max_workers: int | None = None,
    max_bond: int | None = None,
) -> tuple[np.ndarray, str]:
    """Parallel compute identical CSV artifacts as the original, with configurable max_bond."""
    data_dict = calc.JSONtoDict(folder)
    data_arrs, Ts_input, params, pool = calc.DictToArr(data_dict, True)
    N, n, c, Δ, T, q = params
    q = int(q); T = float(T)
    dT = calc.extract_dT_value(folder)
    circuit_pool, sign_pool = data_arrs[0]
    n_timesteps = round(T / dT)
    n_circuits  = int(len(sign_pool) / n_timesteps)
    indices_all = calc.generate_random_indices(len(circuit_pool), n_circuits, n_timesteps)
    Ts = np.arange(0, T + dT, dT)

    if gam_list is not None and len(gam_list) != len(Ts):
        raise ValueError(f"gam_list length ({len(gam_list)}) must equal len(Ts) ({len(Ts)}).")

    if n_runs_to_plot is None:
        n_runs_to_plot = len(indices_all)

    kept_indices  = [None] * n_runs_to_plot
    kept_signs    = [None] * n_runs_to_plot
    kept_weights  = [None] * n_runs_to_plot
    results       = [[] for _ in range(n_runs_to_plot)]
    tuples        = [[] for _ in range(n_timesteps)]

    selected = [[int(ii) for ii in run_indices] for run_indices in indices_all[:n_runs_to_plot]]
    for run_idx, run_indices in enumerate(selected):
        signs = [sign_pool[idx] for idx in run_indices]
        print(f"Starting run {run_idx + 1} with signs: {signs}", flush=True)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(
                _compute_single_run,
                run_indices,
                q,
                [circuit_pool[index] for index in run_indices],
                [sign_pool[index]   for index in run_indices],
                gam_list,
                max_bond if (max_bond is not None) else 2,
            ): i
            for i, run_indices in enumerate(selected)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            run_indices, signs, weights, results_run, tuples_contrib = fut.result()
            kept_indices[i] = run_indices
            kept_signs[i]   = signs
            kept_weights[i] = weights
            results[i]      = results_run
            for t_idx, wm in enumerate(tuples_contrib):
                tuples[t_idx].append(wm)

    os.makedirs(out_dir, exist_ok=True)
    if csv_basename is None:
        tail = os.path.basename(folder.rstrip("/\\"))
        csv_basename = f"runs-{tail}.csv"
    csv_path = os.path.join(out_dir, csv_basename)

    rows = [{
        "run": i + 1,
        "indices": json.dumps(kept_indices[i]),
        "signs":   json.dumps(kept_signs[i]),
        "weights": json.dumps(kept_weights[i]),
        "results": json.dumps(results[i]),
    } for i in range(n_runs_to_plot)]
    pd.DataFrame(rows, columns=["run", "indices", "signs", "weights", "results"]).to_csv(csv_path, index=False)
    print(f"Saved runs to: {csv_path}", flush=True)
    return Ts, csv_path

def compute_or_load_measurement_stats(
    folder: str,
    out_dir: str,
    gam_list: list[float] | None = None,
    hamil: Hamiltonian | None = None,
    csv_basename: str | None = None,
    n_runs: int | None = None,
    max_workers: int | None = None,
    max_bond: int | None = None,
    use_parallel: bool = True,
    allow_cached: bool = True,
) -> MeasurementResults:
    """Compute or load runs CSV, then return unweighted & weighted runs with per-timestep mean/std."""
    if csv_basename is None:
        tail = os.path.basename(folder.rstrip("/\\"))
        csv_basename = f"runs-{tail}.csv"
    csv_path = os.path.join(out_dir, csv_basename)

    if allow_cached and os.path.exists(csv_path):
        weighted_runs, weights_runs, Ts = _load_runs_csv(csv_path)
    else:
        if gam_list is None:
            gam_list = get_gam_list(folder, hamil=hamil)
        if use_parallel:
            Ts, csv_path = compute_and_save_parallel_ext(
                folder=folder,
                n_runs_to_plot=n_runs,
                out_dir=out_dir,
                csv_basename=csv_basename,
                gam_list=gam_list,
                max_workers=max_workers,
                max_bond=max_bond,
            )
        else:
            Ts, _, _ = compute_and_save(
                folder=folder,
                n_runs_to_plot=n_runs if n_runs is not None else 100,
                out_dir=out_dir,
                csv_basename=csv_basename,
                lie_csv_path=None,
                save_plot_path='results_vs_time.png',
                gam_list=gam_list,
                plot=False,
            )
        weighted_runs, weights_runs, Ts = _load_runs_csv(csv_path)

    unweighted_runs = _runs_to_unweighted(weighted_runs, weights_runs)
    uw_mean, uw_std = _mean_std_ragged(unweighted_runs)
    w_mean,  w_std  = _mean_std_ragged(weighted_runs)

    return MeasurementResults(
        Ts=Ts,
        unweighted_runs=unweighted_runs,
        unweighted_mean=uw_mean,
        unweighted_std=uw_std,
        weighted_runs=weighted_runs,
        weighted_mean=w_mean,
        weighted_std=w_std,
        csv_path=csv_path,
    )






# --------------------------- Plotting (agnostic) --------------------------- #
def plot_results(
    Ts,
    results,
    resample_stats=None,
    lie_csv_path=None,
    save_path='results_vs_time.png',
    title='Measurement vs Time',
    error_mode=1,
    q=None,
    delta=None,
    plot_runs=True,
    run_alpha=0.12,
    plot_median_line=False,
):
    """Plot run mean (left) and per-shot RMS error vs Trotter with optional individual-run traces."""
    max_len = max(len(r) for r in results)
    arr = np.full((len(results), max_len), np.nan, dtype=float)
    for i, r in enumerate(results):
        arr[i, :len(r)] = np.asarray(r, dtype=float)

    mean_values = np.nanmean(arr, axis=0)
    std_values = np.nanstd(arr, axis=0, ddof=1)

    Ts = np.asarray(Ts, dtype=float)[:max_len]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    if plot_runs:
        for i in range(arr.shape[0]):
            run = arr[i, :len(Ts)]
            valid = ~np.isnan(run)
            if np.any(valid):
                ax.plot(Ts[valid], run[valid], linewidth=0.8, alpha=run_alpha, zorder=0, color='gray')

    ax.plot(Ts[:len(mean_values)], mean_values, '-', linewidth=2, label='TE-PAI', color='tab:green')
    ax.errorbar(Ts[:len(mean_values)], mean_values, yerr=np.asarray(std_values, float),
                fmt='none', ecolor='tab:green', alpha=0.6, capsize=2)

    if plot_median_line:
        median_values = np.nanmedian(arr, axis=0)
        ax.plot(Ts[:len(median_values)], median_values, '-', linewidth=1.0, alpha=0.35, color='gray', label='Median (runs)')

    ref = None
    if lie_csv_path and os.path.exists(lie_csv_path):
        tmp = pd.read_csv(lie_csv_path)
        if {'x', 'y'}.issubset(tmp.columns):
            ref = tmp[['x', 'y']].dropna()
            ax.plot(ref['x'].values, ref['y'].values, 'k-', linewidth=2, label='Trotterization')

    if resample_stats is not None:
        m_boot = np.asarray(resample_stats.get('mean', []), dtype=float)
        s_boot = np.asarray(resample_stats.get('std', []), dtype=float)
        k = min(len(Ts), len(m_boot), len(s_boot))
        if k > 0:
            ax.errorbar(Ts[:k], m_boot[:k], yerr=s_boot[:k], fmt='o', markersize=3,
                        capsize=2, alpha=0.9, color="tab:green")

    if q is not None and delta is not None:
        times = Ts[:len(mean_values)]
        overhead = np.array([calcOverhead(q, T, delta) for T in times], dtype=float)
        upper = mean_values[:len(times)] + overhead
        lower = mean_values[:len(times)] - overhead
        ax.fill_between(times, lower, upper, color='red', alpha=0.15, zorder=1, label='Theoretical uncertainty band (per shot)')
        ax.plot(times, upper, '--', color='red', alpha=0.6, linewidth=1.5, zorder=1)
        ax.plot(times, lower, '--', color='red', alpha=0.6, linewidth=1.5, zorder=1)

    ax.set_xlabel('Time')
    ax.set_ylabel('Measurement')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small', ncol=2)

    ax2 = axes[1]
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Error (per shot)')
    ax2.set_title('RMS Error vs Trotter and Overhead')
    ax2.grid(True, alpha=0.3)

    if ref is not None:
        trotter_on_ts = np.interp(
            Ts,
            np.asarray(ref['x'].values, dtype=float),
            np.asarray(ref['y'].values, dtype=float),
            left=np.nan, right=np.nan
        )

        rss = np.zeros_like(Ts, dtype=float)
        for i in range(arr.shape[0]):
            run = arr[i, :len(Ts)]
            m = (~np.isnan(run)) & (~np.isnan(trotter_on_ts))
            d = run[m] - trotter_on_ts[m]
            rss[m] += d * d

        rms_per_shot = np.sqrt(rss) / np.sqrt(arr.shape[0])
        valid = ~np.isnan(rms_per_shot)
        ax2.plot(Ts[valid], rms_per_shot[valid], '-', linewidth=2, color='tab:green', label='RMS error vs Trotter')

    if q is not None and delta is not None:
        times = Ts[:len(mean_values)]
        overhead = np.array([calcOverhead(q, T, delta) for T in times], dtype=float)
        ax2.plot(times, overhead, '--', linewidth=2, color='tab:red', label='Overhead (per shot)')

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
    max_bond=None,
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
            ex.submit(_compute_single_run, run_indices, q, [circuit_pool[index] for index in run_indices], [sign_pool[index] for index in run_indices], gam_list, max_bond): i
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
        if max_bond is None:
            csv_basename = f"runs-{tail}.csv"
        else:
            csv_basename = f"runs-{tail}-X-{max_bond}.csv"
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

def _compute_single_run(run_indices, q, circuits, signs, gam_list, max_bond=2):
    """
    Compute a single run: apply gates across all timesteps, collect measurements,
    weights, and per-timestep (weight, measured) tuples for aggregation.
    Returns: (run_indices, signs, weights, results_run, tuples_contrib)
    """
    quimb = calc.getCircuit(int(q), True, max_bond=max_bond)
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
        print(f"Running {time_idx+1} out of {len(circuits)}")
        current_sign *= sign
        calc.applyGates(quimb, circuit)
        measured = calc.measure(quimb)
        gamma_factor = gam_list[time_idx + 1] if gam_list is not None else 1.0
        weight = current_sign * gamma_factor
        results_run.append(measured*weight)
        weights.append(weight)
        tuples_contrib.append((weight, measured))
    return run_indices, signs, weights, results_run, tuples_contrib

def get_gam_list(folder, params=None, hamil=None):
    """Compute the gam_list for given parameters."""
    if params is None:
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
    if hamil is None:
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
    MODE            = "from_csv"   # "compute" or "from_csv"
    #FOLDER          = "TE-PAI-noSampling/data/circuits/N-100-n-1-p-10000-Δ-pi_over_256-q-20-dT-0.2-T-2"
    #FOLDER          = "TE-PAI-noSampling/data/circuits/N-100-n-1-p-1000-Δ-pi_over_256-q-20-dT-0.5-T-5"
    #FOLDER          = "TE-PAI-noSampling/data/circuits/N-100-n-1-p-1000-Δ-pi_over_128-q-20-dT-0.5-T-5"
    FOLDER          = "TE-PAI-noSampling/data/circuits/N-100-n-1-p-10000-Δ-pi_over_1024-q-20-dT-0.5-T-5.0"
    #FOLDER          = "TE-PAI-noSampling/data/circuits/N-100-n-1-p-100000-Δ-pi_over_64-q-20-dT-0.2-T-2"
    #FOLDER          = "TE-PAI-noSampling/data/circuits/N-100-n-1-p-1000-Δ-pi_over_256-q-20-dT-1.0-T-10.0"
    #FOLDER          = "TE-PAI-noSampling/data/circuits/N-100-n-1-p-100-Δ-pi_over_4096-q-20-dT-0.5-T-5.0"
    
    GAM_LIST        = get_gam_list(FOLDER)

    OUT_DIR         = "TE-PAI-noSampling/data/many-circuits"
    CSV_BASENAME    = None
    LIE_CSV         = "TE-PAI-noSampling/data/plotting/lie-N-1000-T-5.0-q-20.csv"
    #LIE_CSV         = "TE-PAI-noSampling/data/plotting/lie-N-100-T-5.0-q-20.csv"

    OUT_PNG         = "results_vs_time.png"
    #CSV_PATH        = "TE-PAI-noSampling/data/many-circuits/runs-N-100-n-1-p-100000-Δ-pi_over_64-q-20-dT-0.2-T-2.csv"
    #CSV_PATH        = "TE-PAI-noSampling/data/many-circuits/runs-N-100-n-1-p-10000-Δ-pi_over_256-q-20-dT-0.2-T-2.csv"
    #CSV_PATH        = "TE-PAI-noSampling/data/many-circuits/runs-N-100-n-1-p-1000-Δ-pi_over_256-q-20-dT-0.5-T-5.csv"
    CSV_PATH        = "TE-PAI-noSampling/data/many-circuits/runs-N-100-n-1-p-10000-Δ-pi_over_1024-q-20-dT-0.5-T-5.0.csv"
    #CSV_PATH        = "TE-PAI-noSampling/data/many-circuits/runs-N-100-n-1-p-1000-Δ-pi_over_128-q-20-dT-0.5-T-5.csv"
    #CSV_PATH        = "TE-PAI-noSampling/data/many-circuits/runs-N-100-n-1-p-100-Δ-pi_over_128-q-20-dT-0.5-T-5.csv"
    #CSV_PATH        = "TE-PAI-noSampling/data/many-circuits/runs-N-100-n-1-p-100-Δ-pi_over_4096-q-20-dT-0.5-T-5.0.csv"
   
    MAX_WORKERS     = None
    N_RUNS          = None
    MAX_BOND        = None

    #FOLDER          = "TE-PAI-noSampling/Truncation/N-100-n-1-p-100-Δ-pi_over_1024-q-20-dT-0.1-T-1.0"
    #LIE_CSV         = "TE-PAI-noSampling/Truncation/Lie-N-100-T-1-q-20-X-0.csv"
    #CSV_PATH        = "TE-PAI-noSampling/data/many-circuits/runs-N-100-n-1-p-100-Δ-pi_over_1024-q-20-dT-0.1-T-1.0.csv"
    #GAM_LIST = np.ones(11)

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
            plot=True,
            max_bond = MAX_BOND
        )
    elif MODE == "from_csv":
        plot_from_csv(
            csv_path=CSV_PATH,
            lie_csv_path=LIE_CSV,
            save_plot_path=OUT_PNG
        )
    else:
        raise ValueError("MODE must be 'compute' or 'from_csv'.")