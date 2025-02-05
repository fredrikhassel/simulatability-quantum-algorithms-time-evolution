from pyquest import Register, unitaries, operators
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import quimb.tensor as qtn
from quimb.gen.operators import pauli, cX, Rz
from scipy.linalg import logm

def trotter_step_first_order_heisenberg(
        n: int, circ, dt: float,
        Jx: float, Jy: float, Jz: float,
        hx: float, hy: float, hz: float,
        periodic: bool = True):
    """
    First-order Trotterized step for the general Heisenberg spin chain model.

    Parameters:
    - n: Number of qubits (spins).
    - circ: The quantum circuit object (e.g., from Quimb or similar library).
    - dt: Time step.
    - Jx, Jy, Jz: Coupling strengths for XX, YY, and ZZ terms.
    - hx, hy, hz: Local magnetic field strengths in X, Y, and Z directions.
    - periodic: Whether to use periodic boundary conditions.
    """

    # Interaction terms: XX, YY, ZZ
    for k in range(n - 1):  # Loop over pairs of qubits
        # XX interaction
        circ.apply_gate('RXX', -2 * Jx * dt, k, k + 1)
        # YY interaction
        circ.apply_gate('RYY', -2 * Jy * dt, k, k + 1)
        # ZZ interaction
        circ.apply_gate('RZZ', -2 * Jz * dt, k, k + 1)

    if periodic:  # Periodic boundary condition
        # XX interaction
        circ.apply_gate('RXX', -2 * Jx * dt, n - 1, 0)
        # YY interaction
        circ.apply_gate('RYY', -2 * Jy * dt, n - 1, 0)
        # ZZ interaction
        circ.apply_gate('RZZ', -2 * Jz * dt, n - 1, 0)

    # Local magnetic field terms: Apply RX, RY, and RZ gates for each site
    for k in range(n):
        if hx != 0:
            circ.apply_gate('RX', -2 * hx * dt, k)  # X-field
        if hy != 0:
            circ.apply_gate('RY', -2 * hy * dt, k)  # Y-field
        if hz != 0:
            circ.apply_gate('RZ', -2 * hz * dt, k)  # Z-field

    return circ

# Variables
n = 6  # Number of qubits
N = 50  # Number of timesteps
T = 10  # Total time
dt = T / N  # Timestep size
Jx = 0.1
Jy = 0.1
Jz = 0.2
hx = 1.2
hy = 0.5
hz = 0.5

# Initialize the quantum registers
circ = qtn.Circuit(n)

# Initialize the state vector to represent the desired initial state
init = np.zeros(2**n, dtype=complex)
# Set the initial state as needed (e.g., all qubits in the |0> state)
init[0] = 1.0

# Construct the magnetization and spin average operators
magnetizationOperator = sum(unitaries.Z([i]).as_matrix(n) for i in range(n))

# Time evolution data storage
mag = []
hMag = []

# Trotterization
for step in range(N):
    trotter_step_first_order_heisenberg(n, circ, dt, Jx, Jy, Jz, hx, hy, hz, periodic=False)
    state_vector_Quimb = circ.psi.to_dense()
    mag.append(np.vdot(state_vector_Quimb, magnetizationOperator @ state_vector_Quimb).real)

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
    ZZ = kron_n(*[Z if j == i or j == i + 1 else I for j in range(n)])
    Hamiltonian += Jz * ZZ + Jy * YY + Jx * XX

# Construct magnetic field terms
for i in range(n):
    # Magnetic field on qubit i
    Hx = kron_n(*[X if j == i else I for j in range(n)])
    Hy = kron_n(*[Y if j == i else I for j in range(n)])
    Hz = kron_n(*[Z if j == i else I for j in range(n)])
    Hamiltonian += hx * Hx + hy * Hy + hz * Hz

# Initialize the state evolution
state_evolution = [init]
for t in np.linspace(0, T, N - 1):
    # Apply the exponentiated Hamiltonian to the current state
    evolved_state = sc.linalg.expm(-1j * dt * Hamiltonian) @ state_evolution[-1]
    state_evolution.append(evolved_state)

# Calculate observables for the exact solution
for state in state_evolution[1:]:
    hMag.append(np.vdot(state, magnetizationOperator @ state).real)

# Hamiltonian simulation
binary = '0'*n

# Bitstring -> MPS
psi0 = qtn.MPS_computational_state(binary)

ZZ = pauli('Z') & pauli('Z')
XX = pauli('X') & pauli('X')
YY = pauli('Y') & pauli('Y')

H_mpo = qtn.LocalHam1D(n, H2=Jz*ZZ + Jy*YY + Jx*XX, H1=hx*pauli('X') + hy*pauli('y') + hz*pauli('Z'), cyclic=False)

# Setting up TEBD object
tebd = qtn.TEBD(psi0, H_mpo, dt=dt*10)
tebd.split_opts['cutoff'] = 1e-10

# Observables
zMags = []
# Sampling TEBD at times
for psit in tebd.at_times(np.linspace(0,T,N), order=2):
    psi_t = psit
    zMag = 0
    for j in range(0,n):
        zMag += psi_t.gate(Z, j).H @ psi_t
    zMags.append(np.real(zMag))

# Generate x-axis values as timesteps from 0 to T
x_data = np.linspace(0, T, N)

# Recreate the figure from scratch
fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True, clear=True)

# Plotting as before
x_data = np.linspace(0, T, N)
ax[0].plot(x_data[:-1], hMag, label="Exact magnetization", color="tab:blue")
ax[0].scatter(x_data, mag, label="Trotter magnetization", color="tab:blue", s=5,)

# Plot the rescaled data
ax[1].plot(x_data[:-1], hMag, color="tab:red", label="Exact magnetization")
ax[1].scatter(x_data[:-1], zMags[1:], s=5, color="tab:red", label="Trotter magnetization (Quimb TEBD)")

magResidual = [[],[]]
mag_Quimb_TEBD = zMags[1:]
for i in range(len(hMag)):
    magResidual[0].append(mag[i]-hMag[i])
    magResidual[1].append(mag_Quimb_TEBD[i]-hMag[i])

ax[2].plot(x_data[:-1], magResidual[0], label="Quimb trotterization residual", color="tab:blue")
ax[2].plot(x_data[:-1], magResidual[1], label="Quimb TEBD residual", color="tab:red")

for i in range(2):
    ax[i].legend()
plt.suptitle("Comparison of trotterization vs exact exponentiated Hamiltonian", fontsize=14)
ax[0].set_title(f'n: {n} N: {N} T: {T}', fontsize=10)

plt.show()

print(np.mean(np.abs(magResidual[0])))
print(np.mean(np.abs(magResidual[1])))
print(np.mean(np.abs(magResidual[1]))/np.mean(np.abs(magResidual[0])))