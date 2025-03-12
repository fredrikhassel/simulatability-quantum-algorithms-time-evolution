from pyquest import Register, unitaries
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import quimb.tensor as qtn
from scipy.linalg import expm

# Define Trotter Step
def trotter_step(n, circ, dt, J, h, periodic=True):
    """Perform one Trotter step with PyQuest and Quimb"""
    interaction_angle = -2 * J * dt
    self_angle = -2 * h * dt

    for k in range(n - 1):
        circ.apply_gate('RZZ', interaction_angle, k, k + 1)
    if periodic:
        circ.apply_gate('RZZ', interaction_angle, n - 1, 0)
    
    for k in range(n):
        circ.apply_gate('RX', self_angle, k)

# Variables
n = 6  # Number of qubits
N = 100  # Number of timesteps
T = 10  # Total time
dt = T / N  # Timestep size
J = 0.2  # Interaction strength
h = 1.2  # Magnetic field coupling

# Construct the magnetization operator
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

def kron_n(*ops):
    result = np.array([1], dtype=complex)
    for op in ops:
        result = np.kron(result, op)
    return result

magnetizationOperator = sum(kron_n(*[Z if j == i else I for j in range(n)]) for i in range(n))

# Trotterization loop
circ1 = qtn.Circuit(n)
mag_Quimb = []
for step in range(N):
    trotter_step(n, circ1, dt, J, h, periodic=False)
    state_vector_Quimb = circ1.psi.to_dense()
    mag_Quimb.append(np.vdot(state_vector_Quimb, magnetizationOperator @ state_vector_Quimb).real)

# Trotterization with fewer steps
N2 = 50
dt2 = T / N2
circ2 = qtn.Circuit(n)
mag_Quimb2 = []
for step in range(N2):
    trotter_step(n, circ2, dt2, J, h, periodic=False)
    state_vector_Quimb = circ2.psi.to_dense()
    mag_Quimb2.append(np.vdot(state_vector_Quimb, magnetizationOperator @ state_vector_Quimb).real)

# Exact Hamiltonian Construction
Hamiltonian = np.zeros((2**n, 2**n), dtype=complex)
for i in range(n - 1):
    ZZ = kron_n(*[Z if j == i or j == i + 1 else I for j in range(n)])
    Hamiltonian += J * ZZ
for i in range(n):
    Hx = kron_n(*[np.array([[0, 1], [1, 0]]) if j == i else I for j in range(n)])
    Hamiltonian += h * Hx

eigvals_exact, _ = np.linalg.eigh(Hamiltonian)
print("Exact Eigenvalues:", eigvals_exact)

# Exact Evolution
init = np.zeros(2**n, dtype=complex)
init[0] = 1.0
state_evolution = [init]
hMag = []

for t in range(N):
    evolved_state = expm(-1j * dt * Hamiltonian) @ state_evolution[-1]
    state_evolution.append(evolved_state)
    hMag.append(np.vdot(evolved_state, magnetizationOperator @ evolved_state).real)

# Exact evolution for N2 steps
state_evolution2 = [init]
hMag2 = []
for t in range(N2):
    evolved_state = expm(-1j * dt2 * Hamiltonian) @ state_evolution2[-1]
    state_evolution2.append(evolved_state)
    hMag2.append(np.vdot(evolved_state, magnetizationOperator @ evolved_state).real)

# Plotting
x_data = np.linspace(0, T, N)
x_data2 = np.linspace(0, T, N2)
fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(x_data, hMag, label="Exact magnetization", color="tab:blue")
ax[0].scatter(x_data, mag_Quimb, label=f"Trotter {N} steps", color="tab:red", s=5)
ax[0].scatter(x_data2, mag_Quimb2, label=f"Trotter {N2} steps", color="tab:orange", s=5, marker="x")
ax[0].legend()
ax[0].set_title("Trotterization vs Exact Hamiltonian Evolution")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Magnetization")
ax[1].plot(x_data, np.array(hMag) - np.array(mag_Quimb), label=f"Error {N} steps", color="tab:red")
ax[1].plot(x_data2, np.array(hMag2) - np.array(mag_Quimb2), label=f"Error {N2} steps", color="tab:orange")
ax[1].legend()
plt.show()
