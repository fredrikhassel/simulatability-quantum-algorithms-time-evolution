from pyquest import Register, unitaries, operators
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from spinChain import trotter_step

# Variables
n = 6  # Number of qubits
N = 50  # Number of timesteps
T = 10  # Total time
dt = T / N  # Timestep size
J = 0.2  # Interaction strength
h = 1.2  # Magnetic field coupling
targets = [2, 3]  # Target qubits
alpha = np.pi / 8  # Magnetic field angle of incidence

# Initialize the quantum register
reg = Register(n)
# Initialize the state vector to represent the desired initial state
init = np.zeros(2**n, dtype=complex)
# Set the initial state as needed (e.g., all qubits in the |0> state)
init[0] = 1.0

# Create a copy of the register
copyReg = reg.copy()

# Construct the magnetization and spin average operators
magnetizationOperator = sum(unitaries.Z([i]).as_matrix(n) for i in range(n))
spinAverageOperator = sum(unitaries.Z([i], [i + 1]).as_matrix(n) for i in range(n - 1)) / (n - 1)

# Time evolution data storage
mag = []
hMag = []
spin = []
hSpin = []

# Trotterization
for step in range(N):
    trotter_step(n, reg, dt, J, h, alpha)
    state_vector = reg[:]
    mag.append(np.vdot(state_vector, magnetizationOperator @ state_vector).real)
    spin.append(np.vdot(state_vector, spinAverageOperator @ state_vector).real)

# Initialize the Hamiltonian as a zero matrix
dim = 2 ** n
Hamiltonian = np.zeros((dim, dim), dtype=complex)

# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

# Function to construct the Kronecker product for operator on specific qubits
def kron_n(*ops):
    result = np.array([1], dtype=complex)
    for op in ops:
        result = np.kron(result, op)
    return result

# Construct interaction terms
for i in range(n - 1):
    # Interaction between qubit i and i+1
    XX = kron_n(*[X if j == i or j == i + 1 else I for j in range(n)])
    YY = kron_n(*[Y if j == i or j == i + 1 else I for j in range(n)])
    Hamiltonian += J * (XX + YY)

# Construct magnetic field terms
for i in range(n):
    # Magnetic field on qubit i
    Hx = kron_n(*[X if j == i else I for j in range(n)])
    Hy = kron_n(*[Y if j == i else I for j in range(n)])
    Hamiltonian += h * (np.cos(alpha) * Hx + np.sin(alpha) * Hy)

print(Hamiltonian)

# Initialize the state evolution
state_evolution = [init]
for t in np.linspace(0, T, N - 1):
    # Apply the exponentiated Hamiltonian to the current state
    evolved_state = sc.linalg.expm(-1j * dt * Hamiltonian) @ state_evolution[-1]
    state_evolution.append(evolved_state)

# Calculate observables for the exact solution
for state in state_evolution:
    hMag.append(np.vdot(state, magnetizationOperator @ state).real)
    hSpin.append(np.vdot(state, spinAverageOperator @ state).real)

# Generate x-axis values as timesteps from 0 to T
x_data = np.linspace(0, T, N)

# Plotting
fig, ax = plt.subplots(3, 1, figsize=(9, 5), sharex=True)
ax[1].scatter(x_data, mag, label="Magnetization (Trotter)", color="r")
ax[1].plot(x_data, hMag, label="Magnetization (Exact)", color="r")
ax[2].scatter(x_data, spin, label="Spin Average (Trotter)", color="b")
ax[2].plot(x_data, hSpin, label="Spin Average (Exact)", color="b")
ax[2].set_xlabel('Time')

for i in range(3):
    ax[i].grid(True)
    ax[i].legend()

plt.tight_layout()
plt.show()
