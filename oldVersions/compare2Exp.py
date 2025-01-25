from pyquest import Register, unitaries, operators
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from spinChain import trotter_step

# Variables
n = 6 # Number of qubits
N = 50 # Number of timesteps
T = 10 # Total time
dt = T / N # Timestep size
J = 0.2 # Interaction strength
h = 1.2 # Magnetic field coupling
targets = [2,3] # 1 cubit
alpha = np.pi/8 # Magnetic field angle of incidence
factor = 2

# Initialize the quantum register and a copy
reg = Register(n)
init = np.zeros(2**n, dtype=complex) # Initial state
for target in targets:
    reg.apply_operator(unitaries.X(target))  # Flip target Qubit to 1
    init[target] = 1
copyReg = reg.copy()

# Creating a circuit for our operators
magnetizationOperator = 0
spinAverageOperator = 0
for i in range(n):
    magnetizationOperator += unitaries.Z([i]).as_matrix(n)
for i in range(n-1):
    spinAverageOperator += unitaries.Z([i],[i+1]).as_matrix(n)
# Averaging
spinAverageOperator = spinAverageOperator/(n-1)

# Time evolution loop
mag   = []
hMag  = []
spin  = []
hSpin = []
ham   = []
hHam  = []

# Trotterization
for step in range(N):
    trotter_step(n, reg, dt, J, h, alpha)
    state_vector = reg[:]
    mag.append(np.vdot(state_vector, magnetizationOperator @ state_vector).real)
    spin.append(np.vdot(state_vector, spinAverageOperator @ state_vector).real)

# Exact solution
Hamiltonian = trotter_step(n, Register(n), dt, J, h, alpha)
print(Hamiltonian)
state_evolution = [init]
for t in np.linspace(0, T, N-1):
    evolved_state = sc.linalg.expm(-1j*dt*Hamiltonian) @ state_evolution[-1]
    state_evolution.append(evolved_state)

for state in state_evolution:
    hMag.append(np.vdot(state, magnetizationOperator @ state).real)
    hSpin.append(np.vdot(state, spinAverageOperator @ state).real)

# Generate x-axis values as timesteps from 0 to T
x_data = np.linspace(0, T, N)

fig, ax = plt.subplots(3, 1, figsize=(9, 5), sharex=True)  # Create subplots
#ax[0].scatter(x_data, ham, label="Hamiltonian", color="g")
#ax[0].plot(x_data, hHam, label="Exact", color="g")
ax[1].scatter(x_data, mag, label="Magnetization", color="r")
ax[1].plot(x_data, hMag, label="Exact", color="r")
ax[2].scatter(x_data, spin, label="Spin Average", color="b")
ax[2].plot(x_data, hSpin, label="Exact", color="b")
ax[2].set_xlabel('Time')

for i in range(3):
    ax[i].grid(True)
    ax[i].legend()

# Display the figure
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
