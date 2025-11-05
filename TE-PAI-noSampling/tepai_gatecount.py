from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any
import numpy as np
import math
from HAMILTONIAN import Hamiltonian  # used in __main__ example; harmless import

__all__ = [
    "precompute_angle_lists",
    "log_overhead_from_delta",
    "overhead_from_delta",
    "epsilon_from_delta",
    "gate_count_from_delta",
    "DeltaSearchResult",
    "find_delta_for_accuracy",
]

# ---------------------------- Internal caches ----------------------------
# Keyed by (id(hamil), float(T), int(N))
_ANGLE_CACHE: Dict[Tuple[int, float, int], Dict[str, Any]] = {}

def _cache_key(hamil, T: float, N: int) -> Tuple[int, float, int]:
    return (id(hamil), float(T), int(N))

# ---------------------------- Helpers (public API preserved) ----------------------------

def _safe_log_abs(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, fill_value=np.nan, dtype=float)
    zero_mask = (x == 0)
    nz = ~zero_mask
    out[nz] = np.log(np.abs(x[nz]))
    out[zero_mask] = -np.inf
    return out

def precompute_angle_lists(hamil, T: float, N: int) -> List[np.ndarray]:
    """
    Build angle lists exactly as in your TE-PAI snippet:
      angles(t_j) = [ 2 * |coef_k(t_j)| * T / N  for k ]
    Adds module-level caching and also precomputes a flattened, nonzero-only array
    for fast evaluations later (used internally by find_delta_for_accuracy).
    """
    key = _cache_key(hamil, T, N)
    cached = _ANGLE_CACHE.get(key)
    if cached is not None:
        return cached["angle_lists"]

    steps = np.linspace(0.0, float(T), int(N))
    angle_lists: List[np.ndarray] = []
    for t in steps:
        coefs = np.asarray(list(hamil.coefs(t)), dtype=complex)
        angles = 2.0 * np.abs(coefs) * float(T) / float(N)
        angle_lists.append(np.asarray(angles, dtype=float))

    # Precompute a flattened array of all NONZERO angles once (zeros contribute 1)
    nonzero_chunks = []
    total_nonzero = 0
    for a in angle_lists:
        nz = a != 0.0
        if np.any(nz):
            nonzero_chunks.append(a[nz].astype(float, copy=False))
            total_nonzero += int(np.count_nonzero(nz))
    angles_all = np.concatenate(nonzero_chunks) if nonzero_chunks else np.empty((0,), dtype=float)

    # Cache L1 norm once if handy downstream
    try:
        l1_norm_T = float(hamil.l1_norm(T))
    except Exception:
        l1_norm_T = None  # still safe; we can call later if needed

    _ANGLE_CACHE[key] = {
        "angle_lists": angle_lists,
        "angles_all": angles_all,           # 1D nonzero angles
        "total_nonzero": total_nonzero,     # count of nonzero angles
        "l1_norm_T": l1_norm_T,
    }
    return angle_lists

def log_overhead_from_delta(angle_lists: List[np.ndarray], delta: float) -> float:
    """
    log(|g|) with:
      g = Π_j gamma(angles_j, Δ)
      gamma(angles, Δ) = Π_k [ cos(sign(θ_k)*Δ/2 - θ_k) / cos(sign(θ_k)*Δ/2) ]
    Preserves original semantics for external callers.
    (find_delta_for_accuracy uses an optimized path below.)
    """
    if math.isclose(math.cos(delta / 2.0), 0.0, abs_tol=1e-14):
        return float("+inf")

    total_log = 0.0
    for angles in angle_lists:
        sgn = np.sign(angles)
        denom = np.cos(sgn * (delta / 2.0))
        if np.any(np.isclose(denom, 0.0, atol=1e-14)):
            return float("+inf")
        num = np.cos(sgn * (delta / 2.0) - angles)
        total_log += float(np.sum(_safe_log_abs(num) - _safe_log_abs(denom)))
    return total_log

def overhead_from_delta(angle_lists: List[np.ndarray], delta: float) -> float:
    lg = log_overhead_from_delta(angle_lists, delta)
    if np.isneginf(lg):
        return 0.0
    if np.isposinf(lg):
        return float("inf")
    if lg > 700:  # ~exp(709) overflows double
        return float("inf")
    return float(np.exp(lg))

def epsilon_from_delta(angle_lists: List[np.ndarray], delta: float, shots: float) -> float:
    """ε(Δ) = |g(Δ)| / sqrt(N_shots)."""
    g = overhead_from_delta(angle_lists, delta)
    if not np.isfinite(g):
        return float("inf")
    return g / math.sqrt(float(shots))

def gate_count_from_delta(hamil, T: float, delta: float) -> float:
    """Gate count: ((3 - cos Δ) / sin Δ) * hamil.l1_norm(T)."""
    s = math.sin(delta)
    if math.isclose(s, 0.0, abs_tol=1e-14):
        return float("inf")
    # Use cached l1_norm(T) if available
    key = _cache_key(hamil, T, 0)  # N doesn't affect l1_norm(T), but keep signature stable
    l1 = None
    for (hid, TT, NN), payload in _ANGLE_CACHE.items():
        if hid == id(hamil) and TT == float(T):
            l1 = payload.get("l1_norm_T")
            break
    if l1 is None:
        l1 = float(hamil.l1_norm(T))
    return ((3.0 - math.cos(delta)) / s) * l1

@dataclass
class DeltaSearchResult:
    delta_opt: float
    gate_count: float
    epsilon: float
    overhead_g: float
    feasible: bool

# ---------------------------- Optimized internal evaluators ----------------------------

def _prep_fast_buffers(hamil, T: float, N: int):
    """Return (angles_all_nonzero, total_nonzero, l1_norm_T) from the cache."""
    key = _cache_key(hamil, T, N)
    payload = _ANGLE_CACHE.get(key)
    if payload is None:
        precompute_angle_lists(hamil, T, N)
        payload = _ANGLE_CACHE[key]
    return payload["angles_all"], payload["total_nonzero"], payload["l1_norm_T"]

def _log_overhead_fast(angles_all: np.ndarray, total_nonzero: int, delta: float) -> float:
    """
    Fast path using the fact that θ>=0 in construction => sign(θ)=1 for nonzero θ.
    log|g| = sum_{θ>0} log|cos(Δ/2 - θ)| - total_nonzero * log|cos(Δ/2)|
    """
    c = math.cos(delta / 2.0)
    if math.isclose(c, 0.0, abs_tol=1e-14):
        return float("+inf")
    # Denominator contribution
    log_denom = total_nonzero * math.log(abs(c))
    # Numerator across all nonzero angles
    if angles_all.size == 0:
        log_num = 0.0
    else:
        num_vals = np.cos((delta * 0.5) - angles_all)
        # allow zeros -> -inf
        with np.errstate(divide="ignore", invalid="ignore"):
            log_num = float(np.sum(np.log(np.abs(num_vals), dtype=float)))
    return log_num - log_denom

def _eps_g_from_delta_fast(angles_all: np.ndarray, total_nonzero: int, delta: float, shots: float) -> Tuple[float, float]:
    lg = _log_overhead_fast(angles_all, total_nonzero, delta)
    if np.isposinf(lg):
        return float("inf"), float("inf")
    if np.isneginf(lg):
        g = 0.0
    elif lg > 700:
        g = float("inf")
    else:
        g = float(np.exp(lg))
    eps = g / math.sqrt(float(shots)) if np.isfinite(g) else float("inf")
    return eps, g

# ---------------------------- Main search (descending Δ, cached) ----------------------------

def find_delta_for_accuracy(
    hamil,
    T: float,
    N: int,
    n_snap: int,           # kept for API compatibility; not used directly here
    shots: float,
    eps_target: float,
    delta_bounds: Tuple[float, float] = (1e-5, math.pi - 1e-5),
    coarse_samples: int = 256,
    refine_iters: int = 30,
    eps_slack: float = 1e-12,
) -> DeltaSearchResult:
    """
    Find the LARGEST Δ in `delta_bounds` satisfying ε(Δ) <= eps_target.
    Now tries the biggest Δ first, scans downward, and uses caching plus a fast evaluator.
    """
    if shots <= 0:
        raise ValueError("shots must be positive.")
    if eps_target <= 0:
        raise ValueError("eps_target must be positive.")
    d_lo, d_hi = float(delta_bounds[0]), float(delta_bounds[1])
    if not (0.0 < d_lo < d_hi < math.pi):
        raise ValueError("delta_bounds must satisfy 0 < d_lo < d_hi < π.")

    # Ensure cached buffers exist
    angle_lists = precompute_angle_lists(hamil, T, N)  # preserves external behavior
    angles_all, total_nonzero, l1_norm_T = _prep_fast_buffers(hamil, T, N)

    # Small per-call cache for (eps, g) by delta
    eval_cache: Dict[float, Tuple[float, float]] = {}
    def eval_eps_g(delta: float) -> Tuple[float, float]:
        # Round to avoid duplicate keys from FP noise in bisection
        key = round(float(delta), 15)
        if key in eval_cache:
            return eval_cache[key]
        val = _eps_g_from_delta_fast(angles_all, total_nonzero, key, shots)
        eval_cache[key] = val
        return val

    # 1) Try the upper bound first (fast path)
    eps_hi, g_hi = eval_eps_g(d_hi)
    if np.isfinite(eps_hi) and eps_hi <= eps_target + eps_slack:
        # Upper bound feasible -> it's already the largest allowed Δ
        gates = ((3.0 - math.cos(d_hi)) / math.sin(d_hi)) * (l1_norm_T if l1_norm_T is not None else float(hamil.l1_norm(T)))
        return DeltaSearchResult(
            delta_opt=d_hi,
            gate_count=gates,
            epsilon=eps_hi,
            overhead_g=g_hi,
            feasible=True,
        )

    # 2) Descending coarse scan until we hit the first feasible Δ
    deltas = np.linspace(d_lo, d_hi, int(coarse_samples))
    deltas = deltas[::-1]  # descending: start near d_hi
    best_min_eps = float("inf")

    prev_delta = d_hi
    prev_eps = eps_hi
    lo = None  # feasible
    hi = None  # infeasible

    for i, d in enumerate(deltas[1:], start=1):  # we've already tested d_hi
        eps_d, _ = eval_eps_g(d)
        if np.isfinite(eps_d):
            best_min_eps = min(best_min_eps, eps_d)
        # We are moving downward. If current d is feasible and previous was infeasible, we bracketed.
        if (eps_d <= eps_target + eps_slack) and not (np.isfinite(prev_eps) and prev_eps <= eps_target + eps_slack):
            lo, hi = d, prev_delta
            break
        prev_delta, prev_eps = d, eps_d

    if lo is None:
        # Never found a feasible point
        raise RuntimeError(
            "No Δ in the provided bounds can achieve the target accuracy. "
            f"Best coarse ε was {best_min_eps:.3e} (target {eps_target:.3e}). "
            "Consider increasing shots, loosening ε, or adjusting N."
        )

    # 3) Refine via bisection between (lo feasible) and (hi infeasible) to maximize Δ
    for _ in range(int(refine_iters)):
        mid = 0.5 * (lo + hi)
        eps_mid, _ = eval_eps_g(mid)
        if np.isfinite(eps_mid) and eps_mid <= eps_target + eps_slack:
            lo = mid   # feasible; push rightwards
        else:
            hi = mid   # infeasible; pull leftwards

    delta_opt = lo
    eps_val, g_val = eval_eps_g(delta_opt)
    # Use cached l1_norm(T) if available
    l1 = l1_norm_T if l1_norm_T is not None else float(hamil.l1_norm(T))
    gates = ((3.0 - math.cos(delta_opt)) / math.sin(delta_opt)) * l1

    return DeltaSearchResult(
        delta_opt=delta_opt,
        gate_count=gates,
        epsilon=eps_val,
        overhead_g=g_val,
        feasible=True,
    )

# ---------------------------- Example (unchanged) ----------------------------
if __name__ == "__main__":
    T = 1.0
    N = 1000
    n_snap = 1
    shots = 1_000_000
    eps_target = 0.01

    q = 20
    rng       = np.random.default_rng(0)
    freqs     = rng.uniform(-1, 1, size=q)
    sch_hamil = Hamiltonian.spin_chain_hamil(q, freqs, j=0.1)

    res = find_delta_for_accuracy(
        hamil=sch_hamil,
        T=T,
        N=N,
        n_snap=n_snap,
        shots=shots,
        eps_target=eps_target,
    )

    print("Δ* =", res.delta_opt)
    print("g(Δ*) =", res.overhead_g)
    print("ε(Δ*) =", res.epsilon)
    print("gate count =", res.gate_count)

    from main import TE_PAI
    tepai = TE_PAI(sch_hamil, sch_hamil.nqubits, res.delta_opt, T, N, n_snap)
    print(tepai.expected_num_gates)
    print(tepai.gamma/math.sqrt(shots))  # should match ε(Δ*)
