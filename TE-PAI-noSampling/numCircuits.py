import gc
import json
import os
import numpy as np
import re

# Import the local TE-PAI classes
from HAMILTONIAN import Hamiltonian
from main import TE_PAI
from TROTTER import Trotter

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