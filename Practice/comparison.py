from pyquest import Register, unitaries, operators
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import quimb.tensor as qtn
from quimb.gen.operators import pauli, cX, Rz
from scipy.linalg import logm

def trotter_step(n: int, obj, dt: float, J: float, h: float, r: int, factor: float =2, periodic: bool = True, getHam: bool = False):
    """ 1 trottized step performed with PyQuest and Quimb """
    reg = obj[0]  # For PyQuest
    circ = obj[1] # For Quimb

    # Trotterized state
    U_trotter = np.eye(2**n, dtype=complex)

    # Function to apply the Trotterized time evolution operator
    interaction_angle = -factor * J * dt
    self_angle = -factor * h * dt

    for k in range(n-1):  # Loop over pairs of qubits
        # Apply the interaction between qubit k and k+1
        X = unitaries.X(k, controls=k + 1)
        reg.apply_operator(X)
        #circ.apply_gate('CX', k, k + 1, gate_round=r)

        # Apply the Rz rotation on qubit k
        Rz = unitaries.Rz(k, interaction_angle)
        reg.apply_operator(Rz)
        #circ.apply_gate('RZ', interaction_angle, k, gate_round=r)        

        # Apply the interaction again
        reg.apply_operator(X)
        #circ.apply_gate('CX', k, k + 1, gate_round=r)

        circ.apply_gate('RZZ', interaction_angle, k, k+1)

        if(getHam): # Generating efective Hamiltonian
            U_trotter = X.as_matrix(n) @ U_trotter
            U_trotter = Rz.as_matrix(n) @ U_trotter
            U_trotter = X.as_matrix(n) @ U_trotter

    if periodic:
        #Periodic boundary condition
        X = unitaries.X(n-1, controls=0)
        reg.apply_operator(X)
        Rz = unitaries.Rz(n-1, interaction_angle)
        reg.apply_operator(Rz)
        reg.apply_operator(X)

        #circ.apply_gate('CX', n - 1, 0, gate_round=r)
        #circ.apply_gate('RZ', interaction_angle, n - 1, gate_round=r)
        #circ.apply_gate('CX', n - 1, 0, gate_round=r)
        circ.apply_gate('RZZ', interaction_angle, n-1, 0)

        if(getHam):
            U_trotter = X.as_matrix(n) @ U_trotter
            U_trotter = Rz.as_matrix(n) @ U_trotter
            U_trotter = Rz.as_matrix(n) @ U_trotter

    # Magnetic field
    for k in range(n):
        Rx = unitaries.Rx([k], self_angle)
        reg.apply_operator(Rx)
        circ.apply_gate('RX', self_angle, k, gate_round=r)        

        if(getHam):
            U_trotter = Rx.as_matrix(n) @ U_trotter

    # Calculating the effective Hamiltonian
    H_eff = 1j / dt * logm(U_trotter)
    
    # eigvals_trotter, eigvecs_trotter
    return np.linalg.eigh(H_eff) # Identity if getHam = False

def trotter_step_second_order(n: int, circ, dt: float, J: float, h: float, r: int, periodic: bool = True):
    """
    Second-order Trotterized step performed with Quimb
    """

    # Half-step for local magnetic field term (RX gates)
    half_self_angle = -h * dt
    for k in range(n):
        circ.apply_gate('RX', half_self_angle, k)

    # Full-step for interaction term (RZZ gates)
    interaction_angle = -2 * J * dt
    for k in range(n - 1):  # Loop over pairs of qubits
        circ.apply_gate('RZZ', interaction_angle, k, k + 1)

    if periodic:  # Periodic boundary condition
        circ.apply_gate('RZZ', interaction_angle, n - 1, 0)

    # Another half-step for local magnetic field term (RX gates)
    for k in range(n):
        circ.apply_gate('RX', half_self_angle, k)

    return circ
 
# Variables
n = 6  # Number of qubits
N = 100  # Number of timesteps
T = 10  # Total time
dt = T / N  # Timestep size
J = 0.2  # Interaction strength
h = 1.2  # Magnetic field coupling

# Initialize the quantum registers
reg = Register(n)
circ = qtn.Circuit(n)
circ2 = qtn.Circuit(n)

# Initialize the state vector to represent the desired initial state
init = np.zeros(2**n, dtype=complex)
# Set the initial state as needed (e.g., all qubits in the |0> state)
init[0] = 1.0

# Construct the magnetization and spin average operators
magnetizationOperator = sum(unitaries.Z([i]).as_matrix(n) for i in range(n))

# Time evolution data storage
mag_PyQuEST = []
mag_Quimb = []
mag_Quimb2 = []
hMag = []

# Trotterization
eigvals_trotter, eigvecs_trotter = None, None # trotterization eigenvalues and eigenvectors
for step in range(N):
    if step == 0:
        eigvals_trotter, eigvecs_trotter = trotter_step(n, [reg, circ], dt, J, h, r=step, periodic=False, getHam = True)
    else:
        trotter_step(n, [reg, circ], dt, J, h, r=step, periodic=False, getHam = False)
    trotter_step_second_order(n, circ2, dt, J, h, r=step, periodic=False)
    
    state_vector_PyQuEST = reg[:]
    state_vector_Quimb = circ.psi.to_dense()
    state_vector_Quimb2 = circ2.psi.to_dense()
    mag_PyQuEST.append(np.vdot(state_vector_PyQuEST, magnetizationOperator @ state_vector_PyQuEST).real)
    mag_Quimb.append(np.vdot(state_vector_Quimb, magnetizationOperator @ state_vector_Quimb).real)
    mag_Quimb2.append(np.vdot(state_vector_Quimb2, magnetizationOperator @ state_vector_Quimb2).real)

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

# Hamiltonian simulation
binary = '0'*n

# Bitstring -> MPS
psi0 = qtn.MPS_computational_state(binary)

ZZ = pauli('Z') & pauli('Z')

H_mpo = qtn.LocalHam1D(n, H2=J*ZZ, H1=h*pauli('X'), cyclic=False)

# Setting up TEBD object
tebd = qtn.TEBD(psi0, H_mpo, dt=dt)
tebd.split_opts['cutoff'] = 1e-10

# Observables
zMags = []
# Sampling TEBD at times
for psit in tebd.at_times(np.linspace(0,T,N)):
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
ax[0].scatter(x_data, mag_PyQuEST, label="Trotter magnetization (pyQuEST)", color="tab:blue", s=5,)
ax[0].scatter(x_data, mag_Quimb, label="Trotter magnetization (Quimb circuit)", color="tab:red", s=5)
ax[0].scatter(x_data, mag_Quimb2, label="Trotter magnetization (Quimb circuit 2)", color="tab:green", s=5)

# Plot the rescaled data
ax[1].plot(x_data[:-1], hMag, color="tab:red", label="Exact magnetization")
ax[1].scatter(x_data[:-1], zMags[1:], s=5, color="tab:red", label="Trotter magnetization (Quimb TEBD)")

magResidual = [[],[],[],[]]
mag_Quimb_TEBD = zMags[1:]
for i in range(len(hMag)):
    magResidual[0].append(mag_PyQuEST[i]-hMag[i])
    magResidual[1].append(mag_Quimb[i]-hMag[i])
    magResidual[2].append(mag_Quimb2[i]-hMag[i])
    magResidual[3].append(mag_Quimb_TEBD[i]-hMag[i])

ax[2].plot(x_data[:-1], magResidual[0], label="pyQuEST residual", color="tab:blue")
ax[2].plot(x_data[:-1], magResidual[1], label="Quimb circuit residual", color="tab:blue")
ax[2].plot(x_data[:-1], magResidual[2], label="Quimb circuit 2 residual", color="tab:green")
ax[2].plot(x_data[:-1], magResidual[3], label="Quimb TEBD residual", color="tab:red")

print(np.max(magResidual[1]))
print(np.max(magResidual[2]))
print(np.max(magResidual[2])/np.max(magResidual[1]))

for i in range(3):
    ax[i].legend()
plt.suptitle("Comparison of trotterization vs exact exponentiated Hamiltonian", fontsize=14)
ax[0].set_title(f'n: {n} N: {N} T: {T}', fontsize=10)

plt.show()