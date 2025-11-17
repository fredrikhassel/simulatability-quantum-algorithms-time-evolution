
import csv
import numpy as np
import calculations as calc
import quimb as qu
import pandas as pd
from HAMILTONIAN import Hamiltonian
from showShots import get_gam_list, compute_and_save_parallel
import os, json, platform
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory as _btf
from circuitGeneratorPool import generate
from TROTTER import Trotter

# Helpers

def get_hamil(H,q,j=0.1):
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)

    #freqs = np.loadtxt("hamil_coef.csv", delimiter=",")

    if H == "SCH":
        hamil = Hamiltonian.spin_chain_hamil(q, freqs, j=j)
    elif H == "NNN":
        hamil = Hamiltonian.next_nearest_neighbor_hamil(q, freqs,j1=j[0], j2=j[1])
    elif H == "2D":
        hamil = Hamiltonian.lattice_2d_hamil(q, freqs=freqs, J=j)
    else:
        raise ValueError(f"Unknown H type: {H}")
    return hamil


def get_resource_estimation_folder(params: dict) -> str:
    """Return path to a matching resource estimation folder (relative to ./RE), creating it if missing."""
    base_dir = os.path.join(os.path.dirname(__file__), "RE")
    os.makedirs(base_dir, exist_ok=True)

    folder_name = "-".join(f"{k}-{v}" for k, v in sorted(params.items()))
    folder_path = os.path.join(base_dir, folder_name)

    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def parse_resource_estimation_folder(path: str) -> dict:
    """Parse a resource estimation folder path into its parameter dictionary."""
    folder_name = os.path.basename(os.path.normpath(path))
    if folder_name == "RE":
        return {}

    parts = folder_name.split("-")
    if len(parts) % 2 != 0:
        raise ValueError(f"Invalid folder name format: {folder_name}")

    return {parts[i]: parts[i + 1] for i in range(0, len(parts), 2)}


def plot(params, last=False):
    folder_path = get_resource_estimation_folder(params)
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=False)

    # Lists to store stats for the second plot
    Ns = []
    mean_errors = []
    std_errors = []

    # --- First subplot: main data ---
    ax_main = axes[0]
    ax_err = axes[1]

    # Load the Cannon reference
    cannon_data = None
    for file in os.listdir(folder_path):
        if file.startswith("cannon_") and file.endswith(".csv"):
            data = pd.read_csv(os.path.join(folder_path, file))
            x = data.iloc[:, 0].astype(float)
            y = 2 * data.iloc[:, 1].astype(float) - 1
            cannon_data = (x, y)
            ax_main.plot(x, y, marker='o', color="black", label="Cannon")

    if cannon_data is None:
        raise ValueError("No cannon_*.csv file found in folder.")
    x_ref, y_ref = cannon_data

    last_diff = []
    # --- Iterate over Trotter runs ---
    for file in os.listdir(folder_path):
        if file.startswith("runs_trotter") and file.endswith(".csv"):
            data = pd.read_csv(os.path.join(folder_path, file))
            cols = data.columns.str.strip()
            try:
                times = cols[1:].astype(float)
            except Exception:
                raise ValueError(f"Could not parse timestep column names: {cols[1:]}")

            for _, row in data.iterrows():
                N = int(row.iloc[0])
                vals = row.iloc[1:].astype(float)
                y_trotter = 2 * vals - 1

                # Plot the trotter trajectory
                ax_main.plot(times, y_trotter, marker='o', label=f"N={N}")

                # Interpolate cannon reference to these times (if necessary)
                y_ref_interp = np.interp(times, x_ref, y_ref)

                # Compute difference and its stats
                diff = np.abs(y_trotter - y_ref_interp)
                last_diff.append(diff[-1])
                mean_diff = np.mean(diff)
                std_diff = np.std(diff)

                Ns.append(N)
                mean_errors.append(mean_diff)
                std_errors.append(std_diff)

    # --- Configure main plot ---
    ax_main.set_xlabel("time")
    ax_main.set_ylabel("value")
    ax_main.legend()
    ax_main.set_title("Trotter vs. Cannon trajectories")

    # --- Second subplot: mean ± std error vs N ---
    if not last:
        ax_err.errorbar(Ns, mean_errors, yerr=std_errors, fmt='o-', capsize=5, color='tab:red')
    else:
        ax_err.plot(Ns, last_diff, color="tab:red")
    ax_err.set_xlabel("Trotter steps (N)")
    ax_err.set_ylabel("Mean |Trotter - Cannon| ± std")
    ax_err.set_title("Trotterization error vs N")
    ax_err.grid(True)

    plt.tight_layout()
    plt.show()


# Trotterization
def perform_trotter(trotter, q, X, progress=True):
    circ = calc.getCircuit(q, max_bond=X)
    measurements = [calc.measure(circ)]


    for i, gates in  enumerate(trotter):
        if progress:
            print(f"Starting part {i+1}/{len(trotter)} with gates {len(gates)}")
        calc.applyGates(circ, gates)
        measurements.append(calc.measure(circ))
    print(measurements)
    return measurements

def make_cannon(params: dict, progress=False, compare=0) -> str:

    folder = get_resource_estimation_folder(params)
    T = float(params["T"])
    N = int(params["C"])
    n_snap = params.get("n_snap", 10)
    q = int(params["q"])
    Htype = params.get("H")
    hamil = get_hamil(Htype,q,params["j"])

    trotter_gates = Trotter(hamil=hamil, numQs=q, T=T, N=N, n_snapshot=n_snap, Δ_name=None, c=None).run(getGates=True)
    measurements = perform_trotter(trotter_gates, q, params["X"], True)
    snap_times = np.linspace(0, T, n_snap+1)
    snap_times = np.asarray(snap_times).ravel()
    measurements = np.asarray(measurements).ravel()
    data = np.column_stack((snap_times, measurements))
    csv_path = os.path.join(folder, "cannon_trotter.csv")
    np.savetxt(csv_path, data, delimiter=",", header="snap_time,measurement", comments="")

    if compare > 0:
        trotter2_gates = Trotter(hamil=hamil, numQs=q, T=T, N=compare, n_snapshot=n_snap, Δ_name=None, c=None).run(getGates=True)
        measurements2 = perform_trotter(trotter2_gates, q, params["X"], False)
        measurements2 = np.asarray(measurements2).ravel()
        data2 = np.column_stack((snap_times, measurements2))
        csv_path2 = os.path.join(folder, "compare_trotter.csv")
        np.savetxt(csv_path2, data2, delimiter=",", header="snap_time,measurement", comments="")

# Resource estimation

def estimate_trotter_resources(params: dict, N_grid, epsilon: float, progress=False) -> str:
    """For each cannon time t>0, find the smallest N (nondecreasing across t) s.t. |trotter_first(H,t,N)-cannon(t)|<=epsilon;
    write x,y,N to trotter-e-{epsilon}.csv, and a full grid of |Δ| vs N to full-e-{epsilon}.csv."""
    folder = get_resource_estimation_folder(params)
    cannon = os.path.join(folder, "cannon_trotter.csv")
    if not os.path.exists(cannon):
        alts = [f for f in os.listdir(folder) if f.startswith("trotter") and f.endswith(".csv")]
        if not alts:
            raise FileNotFoundError("No cannon CSV found.")
        cannon = os.path.join(folder, sorted(alts)[0])
    q = int(params["q"])
    H = params.get("H")
    T = params.get("T")
    hamil = get_hamil(H,q, params["j"])

    data = np.loadtxt(cannon, delimiter=",", skiprows=1)
    x = np.asarray(data[:, 0]).ravel()
    y = np.asarray(data[:, 1]).ravel()

    trott_vals = []
    runs_path = os.path.join(folder, f"runs_trotter.csv")
    # ensure timesteps are floats and a list
    times = [float(t) for t in x]
    with open(runs_path, "w", newline="") as f:
        writer = csv.writer(f)
        # header: first column label 'N', then timesteps
        writer.writerow(["N"] + times)

        for i, N in enumerate(N_grid):
            if progress:
                print(f"Performing Trotterization with N={N}")

            trotter = Trotter(hamil=hamil, N=int(N), T=T, numQs=q, n_snapshot=10, c=None, Δ_name=None)
            # perform_trotter should return an iterable of values with same length as times
            values = perform_trotter(trotter.run(getGates=True), q, params["X"])
            #values = perform_trotter(trotter.run2(), q, params["X"])
            # convert to floats
            vals = [float(v) for v in values]

            # validate lengths
            if len(vals) != len(times):
                raise ValueError(f"Length mismatch: got {len(vals)} values for N={N} but {len(times)} timesteps")

            # write row: N then the values (no list brackets, proper CSV)
            writer.writerow([float(N)] + vals)

    trott_eps = [[] for _ in trott_vals]
    full_path = os.path.join(folder, f"N-runs.csv")
    with open(full_path, "w") as f:
        f.write(f"N,vals")
        for i,trott in enumerate(trott_vals):
            for j,val in enumerate(trott):
                trott_eps[i].append(2*(np.abs(val-y[j])))
            f.write(f"{N_grid[i]},{[float(e) for e in trott_eps[i]]}\n")


    


def per_shot_se_from_csv(csv_path: str, lie_csv_path: str, folder) -> np.ndarray:
    """Compute per-timestep per-shot standard error vs a canonical (x,y) curve using CSV runs and weights."""
    gammas = get_gam_list(folder)
    df = pd.read_csv(csv_path)
    if not {'results', 'weights'}.issubset(df.columns):
        raise ValueError("CSV must have columns: results, weights")

    results_ll, weights_ll = [], []
    for s in df['results'].astype(str):
        results_ll.append(np.asarray(json.loads(s), dtype=float))
    for s in df['weights'].astype(str):
        weights_ll.append(np.asarray(json.loads(s), dtype=float))

    weighted_ll = []
    for r, w in zip(results_ll, weights_ll):
        n = min(len(r), len(w), len(gammas))
        r, w, g = r[:n], w[:n], np.asarray(gammas[:n], dtype=float)
        denom = np.where(np.abs(w) > 0.0, np.abs(w), np.nan)
        weighted_ll.append((r / denom) * g)

    max_len = max(len(x) for x in weighted_ll)
    R = np.full((len(weighted_ll), max_len), np.nan, dtype=float)
    for i, row in enumerate(weighted_ll):
        R[i, :len(row)] = row

    lie_df = pd.read_csv(lie_csv_path)
    y = (lie_df['y'].to_numpy() if 'y' in lie_df.columns else lie_df.iloc[:, 1].to_numpy()).astype(float)
    y = y[:R.shape[1]]

    diffs = R[:, :len(y)] - y[None, :]
    return np.sqrt(np.nanmean(diffs**2, axis=0))


def plot_rmse_three(per_shot_errors, timesteps, epsilon: float, N_max: int = 10_000,
                    figsize=(12, 3.4), eps_x: float = 0.65, show=True):
    """Plot three RMSE-vs-shots subplots (y = σ_per_shot / sqrt(N_s)), draw ε lines and intersection points;
    Also print the N_s required to reach ε for each timestep (even if beyond plotting range)."""
    import numpy as np, matplotlib.pyplot as plt

    if len(per_shot_errors) != 3 or len(timesteps) != 3:
        raise ValueError("Provide exactly three per-shot errors and three timesteps.")

    Ns = np.arange(1, N_max + 1, dtype=float)
    fig, axes = plt.subplots(1, 3, sharey=False, figsize=figsize)

    for ax, se, t in zip(axes, per_shot_errors, timesteps):
        se = float(se) if se is not None else np.nan
        if np.isfinite(se) and se > 0:
            rmse = se / np.sqrt(Ns)
            ax.plot(Ns, rmse, color="tab:green", lw=3)

            # Compute N* for reaching epsilon
            N_star = int(np.ceil((se / epsilon) ** 2)) if epsilon > 0 else None

            # --- Print N_star regardless of whether it's plotted ---
            if N_star is not None:
                print(f"t = {t:>6}:  Required N_s to reach ε = {epsilon:g} is {N_star:,}")

            # --- Plot point only if within range ---
            if N_star is not None and 1 <= N_star <= N_max:
                ax.scatter([N_star], [epsilon], s=40, zorder=3,
                           color="tab:green", edgecolors="black")

        ax.axhline(epsilon, color="black", ls="--", lw=1.5)
        ymax = max(float(se) if np.isfinite(se) else 0.0, float(epsilon))
        ax.set_ylim(0, 1.1 * ymax if ymax > 0 else 1)

        # --- ε annotation ---
        ypad = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        trans = _btf(ax.transAxes, ax.transData)
        ax.text(eps_x, epsilon + ypad, r"$\epsilon$ = " + f"{epsilon:g}",
                transform=trans, ha="left", va="bottom")

        ax.text(0.98, 0.95, f"t = {t}", transform=ax.transAxes,
                ha="right", va="top")
        ax.text(0.1, 0.95, rf"$\sigma_s$ = {se:.2g}", transform=ax.transAxes,
                ha="left", va="top")
        ax.set_xlabel(r"$N_s$")

    fig.supylabel("RMSE")
    fig.tight_layout()

    if show:
        plt.show()
    else:
        return fig, axes


def plot_full_epsilon_three(params: dict, timesteps, epsilon: float = 0.1, figsize=(12, 3.4), eps_x: float = 0.65, show: bool = True):
    "Plot three subplots of attained error vs N from any full-e-*.csv; draw user ε line and highlight the first point with error ≤ ε."
    folder = get_resource_estimation_folder(params)
    eps_tag = f"{float(epsilon):g}"
    preferred = os.path.join(folder, f"full-e-{eps_tag}.csv")
    candidates = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("full-e-") and f.endswith(".csv")]
    if not candidates:
        raise FileNotFoundError(f"No full-e-*.csv files found in {folder}.")
    csv_path = preferred if os.path.exists(preferred) else max(candidates, key=os.path.getmtime)

    df = pd.read_csv(csv_path)
    if not {"x", "N", "epsilon_attained"}.issubset(df.columns):
        raise ValueError("CSV must have columns: x,N,epsilon_attained")
    if len(timesteps) != 3:
        raise ValueError("Provide exactly three timesteps to plot.")

    fig, axes = plt.subplots(1, 3, sharey=False, figsize=figsize)
    xs_unique = np.asarray(sorted(df["x"].unique()))
    for ax, t_req in zip(axes, timesteps):
        if np.any(np.isclose(xs_unique, t_req, rtol=1e-9, atol=1e-12)):
            x_match = xs_unique[np.isclose(xs_unique, t_req, rtol=1e-9, atol=1e-12)][0]
        else:
            x_match = xs_unique[np.argmin(np.abs(xs_unique - t_req))]

        sub = df[df["x"].values == x_match].copy().sort_values("N")
        Ns = sub["N"].to_numpy(float)
        errs = sub["epsilon_attained"].to_numpy(float)

        ax.plot(Ns, errs, color="tab:green", lw=3)

        mask = np.where(np.isfinite(errs) & (errs <= float(epsilon)))[0]
        if mask.size:
            i0 = int(mask[0])
            ax.scatter([Ns[i0]], [errs[i0]], s=40, zorder=3, color="tab:green", edgecolors="black")

        ax.axhline(epsilon, color="black", ls="--", lw=1.5)

        ymax = np.nanmax([np.nanmax(errs) if errs.size else 0.0, float(epsilon)])
        ax.set_ylim(0.0, 1.1 * ymax if ymax > 0 else 1.0)

        ypad = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        trans = _btf(ax.transAxes, ax.transData)
        ax.text(eps_x, float(epsilon) + ypad, r"$\epsilon$ = " + f"{epsilon:g}", transform=trans, ha="left", va="bottom")

        ax.text(0.98, 0.95, f"t = {x_match:g}", transform=ax.transAxes, ha="right", va="top")
        ax.set_xlabel(r"$N$")

    fig.supylabel(r"Error")
    fig.tight_layout()
    if show:
        plt.show()
    else:
        return fig, axes

# TE-PAI

def gen_good_circuits(params, Delta, p, n_timesteps=10.0, N = 100):
    
    out = get_resource_estimation_folder(params)

    q       = params["q"]
    Delta   = Delta
    T       = params["T"]
    dT      = T / n_timesteps
    N       = N
    p       = p
    H       = params["H"]
    j       = params["j"]
    params  = (q,Delta,T,dT,N,p,H)

    n_workers = 4 if platform.system() == "Windows" else None

    generate(params, n_workers=n_workers, j=j, out=out)


def run_good_circuits(params, Delta, n_timesteps=10.0, N = 100):
    
    folder = get_resource_estimation_folder(params)
    T = float(params["T"])

    ps = {
        "N": N,
        "T": T,
        "dT": T/n_timesteps,
        "q": params["q"],
        "Δ": "pi_over_"+str(2**Delta)
    }

    n_workers = 4 if platform.system() == "Windows" else None

    sub = next(f for f in os.listdir(folder) if f.startswith("N"))
    tepai_folder = os.path.join(folder, sub)

    gam_list = get_gam_list(folder, params=ps)
    compute_and_save_parallel(tepai_folder, max_workers=n_workers, n_runs_to_plot=None, gam_list=gam_list, out_dir=folder)


plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    })

if __name__ == "__main__":
    
    #test_trots()

    params = {
    "q"     : 20,
    "H"     : "SCH",
    "j"     : 0.1,
    "T"     : 5,
    "C"     : 5000,
    "X"     : 0
    }

    #params2 = params.copy()
    #params2["C"] = 100
    #make_cannon(params, progress=True, compare=100)
    #test_cannon(params2, get_resource_estimation_folder(params))
    #gen_good_circuits(params, Delta=8, p=10)
    #run_good_circuits(params, Delta=8)
    #plot_full_epsilon_three(params, [1.5, 3.0, 5.0], epsilon=0.01, eps_x=0.2)
    plot_rmse_three([0.59, 1.05, 1.43], [1.5, 3.0, 5.0], epsilon=1e-3, N_max=1000)
    #se = per_shot_se_from_csv(
    #    csv_path = "TE-PAI-noSampling/data/many-circuits/runs-N-100-n-1-p-1000-Δ-pi_over_256-q-20-dT-0.5-T-5.csv",
    #    lie_csv_path = "TE-PAI-noSampling/data/plotting/lie-N-1000-T-5-q-20.csv",
    #    folder = "TE-PAI-noSampling/data/circuits/N-100-n-1-p-1000-Δ-pi_over_128-q-20-dT-0.5-T-5"
    #)
    #make_cannon(params, progress=True)
    #estimate_trotter_resources(params, N_grid=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], epsilon=0.1, progress=True)
    #plot(params, True)
    #make_cannon(params, progress=True)
    #get_resource_estimation_folder(params)
