import numpy as np
from pyquest import Register, unitaries, operators
import scipy as sc
import matplotlib.pyplot as plt

# Variables
N     = 10 # Circuit depth
n     = 5   # Qubit number
T     = 1.0 # total time
dt    = T/N # timestep
J     = 1   # Assuming constant J
target= 2   # Which qubit to measure

# Pauli matrix vector dot product
def pauliProduct(k,l):
    X = np.matmul(unitaries.X(k).as_matrix(n), unitaries.X(l).as_matrix(n))
    Y = np.matmul(unitaries.Y(k).as_matrix(n), unitaries.Y(l).as_matrix(n))
    Z = np.matmul(unitaries.Z(k).as_matrix(n), unitaries.Z(l).as_matrix(n))
    return X+Y+Z

# \00000>
reg  = Register(n)
init = np.zeros(2**n, dtype=complex)

# \00100>
reg.apply_operator(unitaries.X(target))

# Storing measurements for printing
measurements= [reg.prob_of_all_outcomes([target])]
U = []

# Time evolution
for step in range(N):
    # Initializing hamiltonian & variables
    H = 0
    t = (step+1) * dt

    for k in range(n - 1):  # Assuming periodic boundary, can wrap around
        mat = pauliProduct(k,k+1) * J
        H   += mat
        reg.apply_operator(operators.MatrixOperator(targets=range(n),matrix=(mat)))

    # Boundary condition
    mat = pauliProduct(n-1,1) * J
    H   += mat
    reg.apply_operator(operators.MatrixOperator(targets=range(n),matrix=(mat)))

    # Storing measurement
    reg /= np.sqrt(reg.total_prob)
    measurements.append(reg.prob_of_all_outcomes([target]))
    U.append(sc.linalg.expm(-1j*t*H))

# Array to store state evolution
state_evolution = [init]

# Apply each U in the time-evolution operator list
for U_t in U:
    evolved_state = U_t @ state_evolution[-1]
    state_evolution.append(evolved_state)

# Calculate an observable (e.g., total Z expectation)
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)
index = target
operators = [I if i != index else Z for i in range(n)] # I @ I ... @ Z ... @ I @ I
Z_full = operators[0]
for op in operators[1:]:
    Z_full = np.kron(Z_full, op)

# Calculating the expectation value of the Z state for each step in the time evolution
z_expectations = [np.real(np.conj(state).T @ Z_full @ state) for state in state_evolution]

# Generate x-axis values as timesteps from 0 to T
x_data = np.linspace(0, T, N+1)
y_data = list(y[1] for y in measurements)

# Create the figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 1 row, 2 columns

# First plot: Z Expectation Value
axes[0].plot(range(len(z_expectations)), z_expectations, linestyle='-')
axes[0].set_xlabel('Time step')
axes[0].set_ylabel('Z Expectation Value')
axes[0].set_title('Time Evolution of Z Expectation Value')
axes[0].grid(True)

# Second plot: Probability vs. Timesteps
axes[1].plot(x_data, y_data, linestyle='-', label='Probability (%)')
axes[1].set_xlabel('Timesteps (up to 1T)')
axes[1].set_ylabel('Probability (%)')
axes[1].set_title('Normalized prob of all outcomes target 3')
axes[1].legend()
axes[1].grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()