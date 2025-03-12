import gc
import json
import os
from matplotlib import pyplot as plt
import numpy as np
import re

# Import the local TE-PAI classes
from HAMILTONIAN import Hamiltonian
from main import TE_PAI
from TROTTER import Trotter
def compare():
    q = 4
    Δ = np.pi / (2**10)
    T = 0.1
    dT = 0.01
    N = 1000
    Ts = np.arange(0,T,dT)
    # Setting up Hamiltonian
    rng = np.random.default_rng(0)
    freqs = rng.uniform(-1, 1, size=q)
    hamil = Hamiltonian.spin_chain_hamil(q, freqs)
    # Setting up TE-PAI
    te_pai = TE_PAI(hamil, q, Δ, dT, N, 1)

    # Calculating expected number of gates
    v_inf = []
    v_inf.append(te_pai.expected_num_gates)
    print(f"For each {dT} timestep we expect {round(v_inf[0])} gates.")
    n_steps = round(T/dT)
    n_gates = n_steps*v_inf[0]
    print(f"For {n_steps} timesteps the pool approach will therefore be expected to use {round(n_gates)} gates per run.")

    dT = 0.01
    Ts = np.arange(0,T+dT,dT)
    v_inf[0] = TE_PAI(hamil, q, Δ, dT, N, 1).expected_num_gates
    for T in Ts[2:]:
        te_pai = TE_PAI(hamil, q, Δ, T, N, 1)
        v_inf.append(te_pai.expected_num_gates)
        print(f"For T_f = {T} the varying approach will be expected to use {round(v_inf[-1])} gates.")
    print(f"For {n_steps} timesteps the varying approach will therefore be expected to use {round(np.sum(v_inf))} gates per run.")
    ratio = np.sum(v_inf) / n_gates
    print(f"So in the pool approach is expected to use the same number of gates if allowed to run {round(ratio)} as many runs as the varying approach.")

def plot_gate_difference():
    # Define the value of n
    n = 25

    # Define the range of N values
    N_values = np.arange(1, 21)  # N from 1 to 20
    T_values = [N * 0.1 for N in N_values]

    # Compute the sums
    S1_values = N_values * n
    S2_values = n * (N_values * (N_values + 1)) / 2

    # Plot the sums
    plt.figure(figsize=(8, 5))
    plt.plot(T_values, S1_values, label=r'Pool approach', marker='o', linestyle='-', color='b')
    plt.plot(T_values, S2_values, label=r'Sucessive approach', marker='s', linestyle='--', color='r')

    # Labels and title
    plt.xlabel('T')
    plt.ylabel('Number of gates')
    plt.title(f'Comparison of Number of gates required')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

plot_gate_difference()