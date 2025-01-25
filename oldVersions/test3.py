import numpy as np
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

# Variables
n = 7  # Number of qubits
N = 100  # Number of timesteps
T = 1  # Total time
dt = T / N  # Timestep size

# Interaction strength
def J(t):
    return np.cos(20 * np.pi * t)

# Generate random local field strengths
def genOmega(n):
    return np.random.uniform(-1, 1, n)

omega = genOmega(n)

# Initialize the quantum circuit
circuit = QuantumCircuit(n)

# Initial state: |0001000>
circuit.x(3)  # Flip the 4th qubit to 1

# Trotterized time evolution
for step in range(N):
    t = (step + 1) * dt
    
    # H1: Apply Z rotations (local terms)
    for k in range(n):
        circuit.rz(-omega[k] * dt, k)
    
    # H2: Apply interaction terms (nearest neighbors)
    for k in range(n - 1):  # Assuming open boundary conditions
        interaction_angle = -2 * J(t) * dt
        circuit.cx(k, k + 1)
        circuit.rz(interaction_angle, k + 1)
        circuit.cx(k, k + 1)

# Circuit visualization
circuit.draw("mpl")

# Measure Z expectation value
from qiskit.quantum_info import Statevector

# Get the final statevector
state = Statevector.from_instruction(circuit)

# Calculate the Z expectation value
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)
index = 3  # Observe the 4th qubit
operators = [I if i != index else Z for i in range(n)]
Z_full = operators[0]
for op in operators[1:]:
    Z_full = np.kron(Z_full, op)

z_expectation = np.real(np.conj(state.data).T @ Z_full @ state.data)
print(f"Z Expectation Value: {z_expectation}")

# Plot the circuit
circuit.draw("mpl")

