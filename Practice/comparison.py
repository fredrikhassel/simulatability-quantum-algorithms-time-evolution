from pyquest import Register, unitaries, operators
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import quimb.tensor as qtn
from quimb.gen.operators import pauli
from scipy.linalg import logm

def trotter_step(n: int, reg: Register, dt: float, J: float, h: float, factor: float =2, periodic: bool = True):
    """ 1 trottized step performed on the given register """

    # Trotterized state
    U_trotter = np.eye(2**n, dtype=complex)

    # Function to apply the Trotterized time evolution operator
    interaction_angle = -factor * J * dt
    self_angle = -factor * h * dt

    for k in range(n-1):  # Loop over pairs of qubits
        # Apply the interaction between qubit k and k+1
        X = unitaries.X(k, controls=k + 1)
        reg.apply_operator(X)
        U_trotter = X.as_matrix(n) @ U_trotter

        # Apply the Rz rotation on qubit k
        Rz = unitaries.Rz(k, interaction_angle)
        reg.apply_operator(Rz)
        U_trotter = Rz.as_matrix(n) @ U_trotter

        # Apply the interaction again
        reg.apply_operator(X)
        U_trotter = X.as_matrix(n) @ U_trotter

    if periodic:
        #Periodic boundary condition
        X = unitaries.X(n-1, controls=0)
        reg.apply_operator(X)
        U_trotter = X.as_matrix(n) @ U_trotter
        Rz = unitaries.Rz(n-1, interaction_angle)
        U_trotter = Rz.as_matrix(n) @ U_trotter
        reg.apply_operator(Rz)
        reg.apply_operator(X)
        U_trotter = Rz.as_matrix(n) @ U_trotter

    # Magnetic field
    for k in range(n):
        Rx = unitaries.Rx([k], self_angle)
        reg.apply_operator(Rx)
        U_trotter = Rx.as_matrix(n) @ U_trotter

    # Calculating the effective Hamiltonian
    H_eff = 1j / dt * logm(U_trotter)
    
    # eigvals_trotter, eigvecs_trotter
    return np.linalg.eigh(H_eff)

# Variables
n = 6  # Number of qubits
N = 1000  # Number of timesteps
T = 10  # Total time
dt = T / N  # Timestep size
J = 0.2  # Interaction strength
h = 1.2  # Magnetic field coupling
target = 0
alpha = np.pi / 8  # Magnetic field angle of incidence

# Initialize the quantum register
reg = Register(n)

# Initialize the state vector to represent the desired initial state
init = np.zeros(2**n, dtype=complex)
# Set the initial state as needed (e.g., all qubits in the |0> state)
init[0] = 1.0

# Construct the magnetization and spin average operators
magnetizationOperator = sum(unitaries.Z([i]).as_matrix(n) for i in range(n))

# Time evolution data storage
mag = []
hMag = []
spin = []
hSpin = []

# Trotterization
eigvals_trotter, eigvecs_trotter = None, None # trotterization eigenvalues and eigenvectors
for step in range(N):
    if step == 0:
        trotter_step(n, reg, dt, J, h)
    else:
        eigvals_trotter, eigvecs_trotter = trotter_step(n, reg, dt, J, h, periodic=False)

    state_vector = reg[:]
    mag.append(np.vdot(state_vector, magnetizationOperator @ state_vector).real)

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
    ZZ = kron_n(*[Z if j == i or j == i + 1 else I for j in range(n)])
    Hamiltonian += J * ZZ

# Construct magnetic field terms
for i in range(n):
    # Magnetic field on qubit i
    Hx = kron_n(*[X if j == i else I for j in range(n)])
    Hamiltonian += h * Hx

# Compare the hamiltonians:
eigvals_exact, eigvecs_exact = np.linalg.eigh(Hamiltonian)
print("Exact Eigenvalues:", eigvals_exact)
print("Trotterized Eigenvalues:", eigvals_trotter)

# Initialize the state evolution
state_evolution = [init]
for t in np.linspace(0, T, N - 1):
    # Apply the exponentiated Hamiltonian to the current state
    evolved_state = sc.linalg.expm(-1j * dt * Hamiltonian) @ state_evolution[-1]
    state_evolution.append(evolved_state)

# Calculate observables for the exact solution
for state in state_evolution[1:]:
    hMag.append(np.vdot(state, magnetizationOperator @ state).real)

# Quimb hamiltonian simulation
binary = '0'*n

# Bitstring -> MPS
psi0 = qtn.MPS_computational_state(binary)

ZZ = pauli('Z') & pauli('Z')
H_mpo = qtn.LocalHam1D(n, H2=J*ZZ, H1=h*pauli('X'), cyclic=False)

# Setting up TEBD object
tebd = qtn.TEBD(psi0, H_mpo)
tebd.split_opts['cutoff'] = 1e-10

# Observables
mz_t_j = []  # z-magnetization
# Sampling TEBD at times
for psit in tebd.at_times(np.linspace(0,T,N), tol=1e-3):
    # Local observables
    mz_j = []
    info = {"cur_orthog": None}  # Current orthogonality center
    # Adding current datapoints using orthogonality center
    for j in range(1,n):
        mz_j.append(psit.magnetization(j, info=info))
    mz_t_j.append(mz_j)

# Generate x-axis values as timesteps from 0 to T
x_data = np.linspace(0, T, N)

# Recreate the figure from scratch
fig, ax = plt.subplots(4, 1, figsize=(9, 6), sharex=True, clear=True)

# Plotting as before
x_data = np.linspace(0, T, N)
ax[0].scatter(x_data, mag, label="Trotter magnetization (pyQuEST)", color="tab:blue", marker="x")
ax[0].plot(x_data[:-1], hMag, label="Exact magnetization", color="tab:green")

# Calculate the scaling factor based on the first point
mz_t_j_sum = []
for m in mz_t_j:
    mz_t_j_sum.append(np.sum(m))

scaling_factor = mz_t_j_sum[0] / hMag[0]

# Rescale hMag using the scaling factor
hMag_rescaled = [value * scaling_factor for value in hMag]

# Plot the rescaled data
ax[1].scatter(x_data[:-1], mz_t_j_sum[:-1], marker="x", color="tab:blue", label="Trotter magnetization (Quimb)")
ax[1].plot(x_data[:-1], hMag_rescaled, color="tab:green", label="Exact magnetization *SCALED")

magResidual = [[],[]]
for i in range(len(hMag)):
    magResidual[0].append(mag[i]-hMag[i])
    magResidual[1].append(hMag_rescaled[i]-mz_t_j_sum[i])

ax[2].plot(x_data[:-1], magResidual[0], label="pyQuEST residual")
ax[3].plot(x_data[:-1], magResidual[1], label="Quimb residual")

for i in range(4):
    ax[i].legend()
plt.suptitle("Comparison of trotterization vs exact exponentiated Hamiltonian", fontsize=14)
ax[0].set_title(f'n: {n} N: {N} T: {T}', fontsize=10)

plt.show()