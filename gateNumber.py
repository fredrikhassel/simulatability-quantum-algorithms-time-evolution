import numpy as np
import matplotlib.pyplot as plt

# Define T range and l values
T = np.linspace(0, 10, 500)
ls = range(8, 13)

# Prepare figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- First plot: expected number of gates ---
for l in ls:
    Δ = np.pi / (2**l)
    # original formula: ν = (1/sin Δ) (3 − cos Δ) T
    nu = (1 / np.sin(Δ)) * (3 - np.cos(Δ)) * T
    ax1.plot(T, nu, label=f"$l={l}$")
ax1.set_xlabel("$T$")
ax1.set_ylabel(r"$\nu_\infty(T)$")
ax1.set_yscale('log')
ax1.set_title(r"Expected \# gates ($N\to\infty$)")
ax1.legend(loc='lower right', fontsize='small')

# --- Second plot: overhead vs time ---
for l in ls:
    Δ = np.pi / (2**l)
    print(np.tan(Δ/2))
    # overhead = exp[ 2 * ||c̄||_1 * T * tan(Δ/2) ], here ||c̄||_1 = 1
    overhead = np.exp(2 * T * 120*np.tan(Δ/2))
    ax2.plot(T, overhead, label=f"$l={l}$")
ax2.set_xlabel("$T$")
ax2.set_ylabel("Overhead")
ax2.set_yscale('log')
ax2.set_title("Circuit overhead vs time")
ax2.legend(loc='upper left', fontsize='small')

plt.tight_layout()
plt.show()
