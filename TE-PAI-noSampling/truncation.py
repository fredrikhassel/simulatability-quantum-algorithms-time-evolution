from pathlib import Path
import argparse, json, os, time
from typing import Dict, Optional, Tuple, List, re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from HAMILTONIAN import Hamiltonian
from calculations import getCircuit, applyGates, measure
from concurrent.futures import ProcessPoolExecutor, as_completed
from showShots import _compute_single_run, get_gam_list, compute_or_load_measurement_stats, gam_list_for_stitched_time_indep, _parse_T_and_dT_from_name, _ragged_to_2d


def aggregate_runs_with_weights(runs, signs, gam_list):
    """Aggregate runs with per-timestep sign*weight; auto-prepend 1 if one element short."""
    runs_arr = np.asarray(runs, dtype=float)
    signs_arr = np.asarray(signs, dtype=float)
    gam = np.asarray(gam_list, dtype=float)

    T_runs, T_signs, T_gam = runs_arr.shape[1], signs_arr.shape[1], gam.shape[0]
    max_T = max(T_runs, T_signs, T_gam)
    if T_signs < max_T:
        signs_arr = np.hstack([np.ones((signs_arr.shape[0], 1)), signs_arr])
    if T_runs < max_T:
        runs_arr = np.hstack([np.ones((runs_arr.shape[0], 1)), runs_arr])
    if T_gam < max_T:
        gam = np.concatenate([[1.0], gam])
    if not (runs_arr.shape == signs_arr.shape and gam.shape[0] == runs_arr.shape[1]):
        raise ValueError("Array length mismatch after padding correction.")

    weighted = runs_arr * signs_arr * gam[np.newaxis, :]
    y_mean = weighted.mean(axis=0)
    R = weighted.shape[0]
    y_sem = weighted.std(axis=0, ddof=1) / np.sqrt(R) if R > 1 else np.zeros_like(y_mean)
    return y_mean, y_sem


def _fmt_float(x: float) -> str:
    """Return compact filesystem-friendly float string."""
    return ("%g" % x).replace(".", "_")


def _ensure_dir(p: Path) -> None:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def _chunk_bounds(total: int, chunks: int) -> np.ndarray:
    """Integer boundaries splitting range(total) into `chunks` parts."""
    return np.linspace(0, total, chunks + 1, dtype=int)


def _build_hamiltonian(params: Dict):
    """Build spin-chain Hamiltonian from params or TE-PAI meta file if provided."""
    q = int(params["q"]) 
    rng = np.random.default_rng(int(params.get("seed", 0)))

    freqs = params.get("freqs")
    meta_dir = params.get("tepai_circuits_dir")
    if meta_dir and freqs is None:
        try:
            meta_path = None
            p = Path(meta_dir)
            if p.exists() and p.is_dir():
                for f in sorted(p.iterdir()):
                    if f.name.startswith("hamil_meta") and f.is_file():
                        meta_path = f
                        break
            if meta_path is not None:
                import csv, json as _json
                with open(meta_path, 'r', encoding='utf-8') as fh:
                    reader = csv.reader(fh)
                    rows = list(reader)
                freqs_val = None
                for name, val in rows:
                    if name.strip().lower() == 'freqs':
                        freqs_val = val
                        break
                if freqs_val is None and rows:
                    freqs_val = rows[0][1]
                if freqs_val is not None:
                    try:
                        freqs = np.asarray(_json.loads(freqs_val), dtype=float)
                    except Exception:
                        try:
                            freqs = np.fromstring(freqs_val.strip(' []'), sep=',')
                        except Exception:
                            freqs = None
        except Exception:
            freqs = None

    if freqs is None:
        freqs = rng.uniform(-1, 1, size=q)
    return Hamiltonian.spin_chain_hamil(q, freqs, j=float(params.get("j", 1.0)))


def _build_trotter_step_gates(hamil, T: float, N: int) -> List[List[tuple]]:
    """Per-step gate lists for first-order Lie–Trotter over 0→T with N steps."""
    times = np.linspace(0.0, T, N)
    terms = [hamil.get_term(t) for t in times]
    return [[(pauli, 2 * coef * T / N, ind) for (pauli, ind, coef) in terms[i]] for i in range(N)]


def _measure_over_steps(circ, step_gates: List[List[tuple]], num_points: int, include_t0: bool, T: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply step-gates in chunks; after each chunk record measurement and bond dimension."""
    bounds = _chunk_bounds(len(step_gates), num_points)
    times = np.linspace(0.0 if include_t0 else T / num_points, T, num_points + (1 if include_t0 else 0))
    vals, bonds = [], []
    if include_t0:
        vals.append(measure(circ)); bonds.append(int(circ.psi.max_bond()))
    for k in range(num_points):
        s, e = bounds[k], bounds[k + 1]
        if e > s:
            applyGates(circ, [g for step in step_gates[s:e] for g in step])
        vals.append(measure(circ)); bonds.append(int(circ.psi.max_bond()))
    return times, np.asarray(vals), np.asarray(bonds)


def compute_or_read_trotter_untruncated(params: Dict) -> Tuple[Path, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load untruncated Trotter CSV or simulate; always save to out_dir; return (path, times, values, bonds)."""
    out_dir = Path(params.get("out_dir", "TE-PAI-noSampling/Truncation")); _ensure_dir(out_dir)
    q, T = int(params["q"]), float(params["T"])
    N = int(params.get("N_trotter", params.get("N", 100)))
    provided_csv = params.get("trotter_csv")

    save_path = out_dir / f"Lie-N-{N}-T-{_fmt_float(T)}-q-{q}-X-0.csv"

    if provided_csv:
        path = Path(provided_csv)
        data = np.loadtxt(path, delimiter=",")
        x, y = data[:, 0], data[:, 1]
        # Always (re)save into our canonical location
        np.savetxt(save_path, np.column_stack([x, y]), delimiter=",", fmt="%.10f")
        return save_path, x, y, None

    hamil = _build_hamiltonian(params)
    circ = getCircuit(q, flip=True, max_bond=None)
    steps = _build_trotter_step_gates(hamil, T=T, N=N)
    x, y, b = _measure_over_steps(circ, steps, num_points=10, include_t0=False, T=T)
    np.savetxt(save_path, np.column_stack([x, y]), delimiter=",", fmt="%.10f")
    return save_path, x, y, b


def compute_trotter_truncated(params: Dict) -> Tuple[Path, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Simulate truncated-bond Trotter; return (path, times, values, bonds)."""
    out_dir = Path(params.get("out_dir", "TE-PAI-noSampling/Truncation")); _ensure_dir(out_dir)
    q, T = int(params["q"]), float(params["T"])
    N = int(params.get("N_trotter", params.get("N", 100)))
    X = int(params.get("max_bond", 16))

    save_path = out_dir / f"Lie-N-{N}-T-{_fmt_float(T)}-q-{q}-X-{X}.csv"
    hamil = _build_hamiltonian(params)
    circ = getCircuit(q, flip=True, max_bond=X)
    steps = _build_trotter_step_gates(hamil, T=T, N=N)
    x, y, b = _measure_over_steps(circ, steps, num_points=10, include_t0=False, T=T)
    np.savetxt(save_path, np.column_stack([x, y]), delimiter=",", fmt="%.10f")
    return save_path, x, y, b


def run_tepai(params: Dict) -> Tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate TE-PAI runs (or load from CSV); always save to out_dir; return (path, x, mean, sem)."""
    out_dir = Path(params.get("out_dir", "simulatability-quantum-algorithms-time-evolution/TE-PAI-noSampling/Truncation")); _ensure_dir(out_dir)
    X = int(params.get("tepai_max_bond", params.get("max_bond", 16))) # max bond dimension
    tepai_csv = params.get("tepai_csv")                               # None if no csv exists
    circuits_dir = params.get("tepai_circuits_dir")                   # Folder containing circuits and signs
    hamil = _build_hamiltonian(params)                                # Hamiltonian
    gam_list = get_gam_list(folder=circuits_dir, hamil=hamil)         # GAM_LIST

    res = compute_or_load_measurement_stats(
    folder=circuits_dir,
    out_dir=out_dir,
    gam_list=gam_list,
    hamil=hamil,               # or a Hamiltonian if you want us to build gam_list for you
    n_runs=None,              # None -> all generated runs
    max_workers=4,         # defaults to cpu_count()-1 in ProcessPoolExecutor
    max_bond=2,               # set maximum bond dimension
    use_parallel=True,        # set False to use the original serial path
    allow_cached=True,        # read CSV if it already exists
    )

    return res.csv_path

def test_trotter_only(params: Dict) -> None:
    """Run untruncated & truncated Trotter once; plot expectations and bond dimensions."""
    out_dir = Path(params.get("out_dir", "TE-PAI-noSampling/Truncation")); _ensure_dir(out_dir)
    _, t_u, y_u, b_u = compute_or_read_trotter_untruncated(params)
    _, t_t, y_t, b_t = compute_trotter_truncated(params)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax0, ax1 = axes
    ax0.plot(t_u, y_u, color="black", label="Trotter (X=∞)")
    ax0.plot(t_t, y_t, linestyle="--", color="gray", label=f"Trotter (X={int(params.get('max_bond', 16))})")
    ax0.set_xlabel("Time"); ax0.set_ylabel("Expectation value"); ax0.set_title("Expectation vs Time")
    ax0.legend(); ax0.grid(True, alpha=0.25)

    if b_u is not None:
        ax1.plot(t_u, b_u, color="black", label="X=∞ bond dim")
    if b_t is not None:
        ax1.plot(t_t, b_t, linestyle="--", color="gray", label=f"X={int(params.get('max_bond', 16))} bond dim")
    ax1.set_xlabel("Time"); ax1.set_ylabel("Bond dimension"); ax1.set_title("Bond dimension vs Time")
    ax1.legend(); ax1.grid(True, alpha=0.25)

    fig_path = out_dir / f"trotter-only-with-bonds-N-{params.get('N_trotter', params.get('N', 100))}-T-{_fmt_float(params['T'])}-q-{params['q']}.png"
    if bool(params.get("show", True)):
        plt.show()
    fig.savefig(fig_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

def plot_all(params: Dict, trotter_csv: Path, trotter_trunc_csv: Path, tepai_csv: Path,
             save_path: Optional[Path] = None, show: bool = True, weighted: bool = True) -> None:
    """Plot Trotter, truncated Trotter, and TE-PAI (from runs CSV or legacy 3-col CSV)."""
    def _looks_like_runs_csv(p: Path) -> bool:
        """Return True if CSV has runs schema."""
        try:
            with open(p, "r", encoding="utf-8") as fh:
                head = fh.readline().lower()
            return ("results" in head and "weights" in head) or head.startswith("run,")
        except Exception:
            return False

    def _ragged_to_2d(ll):
        """Pad ragged list-of-lists with NaN."""
        max_len = max(len(x) for x in ll)
        A = np.full((len(ll), max_len), np.nan, float)
        for i, x in enumerate(ll):
            A[i, :len(x)] = np.asarray(x, float)
        return A

    def _mean_se_ragged(ll):
        """Compute per-step mean and standard error."""
        A = _ragged_to_2d(ll)
        mean = np.nanmean(A, axis=0)
        n = np.sum(~np.isnan(A), axis=0).astype(float)
        diffs = A - mean
        diffs[np.isnan(diffs)] = 0.0
        ss = np.sum(diffs * diffs, axis=0)
        denom = np.where(n >= 2, n - 1.0, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            std = np.sqrt(ss / denom)
            se = std / np.sqrt(n)
        se = np.nan_to_num(se, nan=0.0, posinf=0.0, neginf=0.0)
        return mean, se

    def _parse_dt(path_str: str) -> float:
        """Extract dT from filename; fallback to 1.0."""
        import re as _re
        bname = os.path.basename(path_str)
        m = _re.search(r"(?:^|[-_])dT-([0-9]*\.?[0-9]+)", bname)
        if not m:
            m = _re.search(r"(?:^|[-_])dt-([0-9]*\.?[0-9]+)", bname)
        return float(m.group(1)) if m else 1.0

    def _load_tepai_from_runs_csv(p: Path, use_weighted: bool):
        """Return (x, mean, SE) from runs CSV."""
        df = pd.read_csv(p)
        if not {"results", "weights"}.issubset(df.columns):
            raise ValueError("TE-PAI CSV missing 'results'/'weights' columns.")
        runs_w   = [[float(x) for x in json.loads(s)] for s in df["results"].astype(str)]
        runs_wts = [[float(x) for x in json.loads(s)] for s in df["weights"].astype(str)]
        if use_weighted:
            runs = runs_w
        else:
            runs = [[(r / w) if w != 0 else np.nan for r, w in zip(rw, ww)]
                    for rw, ww in zip(runs_w, runs_wts)]
        y_mean, y_se = _mean_se_ragged(runs)
        dt = _parse_dt(str(p))
        x = np.arange(len(y_mean), dtype=float) * dt
        return x, y_mean, y_se

    trotter_xy = np.loadtxt(trotter_csv, delimiter=",")
    trotter_trunc_xy = np.loadtxt(trotter_trunc_csv, delimiter=",")
    x0, y0 = trotter_xy[:, 0], trotter_xy[:, 1]
    x1, y1 = trotter_trunc_xy[:, 0], trotter_trunc_xy[:, 1]

    if _looks_like_runs_csv(tepai_csv):
        xt, yt, zt = _load_tepai_from_runs_csv(tepai_csv, use_weighted=weighted)
        label = "TE-PAI (weighted)" if weighted else "TE-PAI (unweighted)"
    else:
        tepai_xyz = np.loadtxt(tepai_csv, delimiter=",")
        xt, yt, zt = tepai_xyz[:, 0], tepai_xyz[:, 1], tepai_xyz[:, 2]
        label = "TE-PAI"

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x0, y0, color="black", label="Trotter (X=∞)")
    ax.plot(x1, y1, linestyle="--", color="gray", label=f"Trotter (X={int(params.get('max_bond', 16))})")
    ax.errorbar(xt, yt, yerr=zt, fmt="o-", linewidth=1.2, label=label, color="tab:green")
    ax.set_xlabel("Time"); ax.set_ylabel("Expectation value"); ax.set_title("Tensor-network Trotter vs. truncated and TE-PAI")
    ax.legend(); ax.grid(True, alpha=0.25)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)

def run_experiment(params: Dict) -> None:
    """Compute/load curves, run TE-PAI aggregation, then plot all."""
    out_dir = Path(params.get("out_dir", "TE-PAI-noSampling/Truncation")); _ensure_dir(out_dir)
    trott_path, _, _, _ = compute_or_read_trotter_untruncated(params)
    trott_trunc_path, _, _, _ = compute_trotter_truncated(params)
    tepai_path = run_tepai(params)

    fig_path = out_dir / f"comparison-N-{params.get('N_trotter', params.get('N', 100))}-T-{_fmt_float(params['T'])}-q-{params['q']}.png"
    plot_all(params, trott_path, trott_trunc_path, tepai_path, save_path=fig_path, show=bool(params.get("show", True)))


def _load_params(path: Optional[str]) -> Dict:
    """Load JSON params from file or return {}."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _merge_overrides(base: Dict, overrides: Dict) -> Dict:
    """Overlay CLI overrides onto base params and return merged mapping."""
    merged = dict(base)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged

def run_trotter_truncated_and_plot(params: Dict) -> Path:
    """Run truncated Trotterization, save CSV, and plot expectation/bond vs time."""
    csv_path, t, y, b = compute_trotter_truncated(params)

    X = int(params.get("max_bond", 16))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax0, ax1 = axes

    ax0.plot(t, y, linestyle="--", color="gray", label=f"Trotter (X={X})")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Expectation value")
    ax0.set_title("Truncated Trotter: Expectation vs Time")
    ax0.legend()
    ax0.grid(True, alpha=0.25)

    if b is not None:
        ax1.plot(t, b, linestyle="--", color="gray", label=f"X={X} bond dim")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Bond dimension")
    ax1.set_title("Truncated Trotter: Bond dimension vs Time")
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    if bool(params.get("show", True)):
        plt.show()
    plt.close(fig)

    return csv_path


def main() -> None:
    """CLI: parse args, build params, run the experiment."""
    parser = argparse.ArgumentParser(description="Benchmark TN Trotter vs. truncation vs. TE-PAI")
    parser.add_argument("--params", type=str, default=None, help="Path to JSON params file")
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--T", type=float, default=None)
    parser.add_argument("--N_trotter", type=int, default=None)
    parser.add_argument("--N_tepai", type=int, default=None)
    parser.add_argument("--max_bond", type=int, default=None)
    parser.add_argument("--tepai_max_bond", type=int, default=None)
    parser.add_argument("--delta", type=int, default=None)
    parser.add_argument("--num_circuits", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--j", type=float, default=None)
    parser.add_argument("--trotter_csv", type=str, default=None)
    parser.add_argument("--tepai_circuits_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--tepai_csv", type=str, default=None)
    args = parser.parse_args()

    base = _load_params(args.params)
    overrides = {
        "q": args.q, "T": args.T, "N_trotter": args.N_trotter, "N_tepai": args.N_tepai,
        "max_bond": args.max_bond, "tepai_max_bond": args.tepai_max_bond,
        "delta": args.delta, "num_circuits": args.num_circuits, "seed": args.seed,
        "j": args.j, "trotter_csv": args.trotter_csv, "tepai_circuits_dir": args.tepai_circuits_dir,
        "out_dir": args.out_dir, "show": args.show,
    }
    params = _merge_overrides(base, overrides)

    missing = [k for k in ("q", "T") if k not in params or params[k] is None]
    if missing:
        raise SystemExit(f"Missing required params: {missing}")
    run_experiment(params)

if __name__ == "__main__":
    #main()
    #quit()
    params = { "q": 20, "T": 5.0, "N_trotter": 100, "max_bond": 16, "seed": 0, "j": 1.0, "show": True, 
            "trotter_csv": "TE-PAI-noSampling/Truncation/Lie-N-100-T-1-q-20-X-0.csv",
            "out_dir": "TE-PAI-noSampling/Truncation",
            }
    print(run_trotter_truncated_and_plot(params))
    #test_trotter_only(params)
