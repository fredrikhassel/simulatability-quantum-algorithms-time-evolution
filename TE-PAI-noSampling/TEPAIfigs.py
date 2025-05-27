import numpy as np
from HAMILTONIAN import Hamiltonian
from TROTTER import Trotter
from main import TE_PAI
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines

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

def main():
    # ——— Common Parameters ———
    n = 10
    q = 10
    d = np.pi / (2**7)
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    print(hamil.l1_norm(0.5))
    print(hamil.l1_norm(1))
    print(hamil.l1_norm(1.5))

    return

    T_values = [0.5, 1.0, 1.5]
    cmap = plt.get_cmap('tab10')

    # Define N ranges
    N1 = np.arange(10, 200, 20)
    N2 = np.arange(200, 2600, 200)

    # Pre-compute data and asymptotes
    results = []
    for T in T_values:
        # expected gates vs N1
        exp_gates = []
        asym_gate = None
        for N in N1:
            te_pai = TE_PAI(hamil, q, d, T, N, n)
            exp_gates.append(te_pai.rea_expect_num_gates)
            if asym_gate is None:
                asym_gate = te_pai.expected_num_gates

        # overhead vs N2
        gamma_vals = []
        asym_over = None
        for N in N2:
            te_pai = TE_PAI(hamil, q, d, T, N, n)
            gamma_vals.append(te_pai.gamma)
            if asym_over is None:
                asym_over = te_pai.overhead

        results.append({
            'T': T,
            'exp_gates': np.array(exp_gates),
            'asym_gate': asym_gate,
            'gamma_vals': np.array(gamma_vals),
            'asym_over': asym_over
        })

    # Determine y-limits to align the maximum asymptote line across subplots
    margin = 1.15
    y1_max = max(r['asym_gate'] for r in results) * margin
    y2_max = max(r['asym_over'] for r in results) * margin

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot data
    for i, r in enumerate(results):
        # subplot 1: expected gates
        N1_plot = np.insert(N1, 0, 0)
        gates_plot = np.insert(r['exp_gates'], 0, 0)
        ax1.plot(N1_plot, gates_plot,
                 linewidth=1.5, color=cmap(i), label=f'T = {r["T"]}', alpha=0.6)
        ax1.hlines(r['asym_gate'], N1_plot[0], N1_plot[-1],
                   linestyles='--', linewidth=2, color='black')

        # subplot 2: measurement overhead
        N2_plot = np.insert(N2, 0, 0)
        gamma_plot = np.insert(r['gamma_vals'], 0, 0)
        ax2.plot(N2_plot, gamma_plot,
                 linewidth=1.5, color=cmap(i), alpha=0.6)
        ax2.hlines(r['asym_over'], N2_plot[0], N2_plot[-1],
                   linestyles='--', linewidth=2, color='black')

    # Final styling
    ax1.set_xlabel('N')
    ax1.set_ylabel('Number of gates')
    ax1.set_ylim(0, y1_max)
    ax1.grid(True)

    ax2.set_xlabel('N')
    ax2.set_ylabel('Measurement overhead')
    ax2.set_ylim(0, y2_max)
    ax2.grid(True)

    # build legend entries
    handles, labels = ax1.get_legend_handles_labels()
    # create a dummy (invisible) handle for the “title”
    title_handle = mlines.Line2D([], [], linestyle='None', label='Total time $T$')
    # prepend it
    handles.insert(0, title_handle)
    labels.insert(0, 'Total time $T$')

    # draw a single-row legend underneath
    fig.legend(handles, labels,
            loc='lower center',
            ncol=len(handles),
            frameon=False)

    plt.subplots_adjust(bottom=0.3)  # or bump even more if needed
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()