import quimb as qu
import quimb.tensor as qtn
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# Variables
n = 2 # Number of qubits
N = 60 # Number of timesteps
T = 5 # Total time
dt = T / N # Timestep size
J = 0.2 # Interaction strength
h = 1.2 # Magnetic field coupling
target = 1 # 1 cubit
alpha = np.pi/8 # Magnetic field angle of incidence

# Initializing 2-qubit circuit
circ = qtn.Circuit(n)

# X gate on target qubit
circ.apply_gate('X', target)

# Function to apply the Trotterized time evolution operator
def trotter_tensor(n, circ, dt, r, J=1, h=1, alpha=0, factor=2):
    interaction_angle = -factor * J * dt
    self_angle = -factor * h * dt

    for k in range(n-1):  # Loop over pairs of qubits
        circ.apply_gate('CX', k, k + 1, gate_round=r)
        circ.apply_gate('RZ', interaction_angle, k, gate_round=r)
        circ.apply_gate('CX', k, k + 1, gate_round=r)
    
    # Periodic boundary condition
    circ.apply_gate('CX', n-1, 0, gate_round=r)
    circ.apply_gate('RZ', interaction_angle, n-1, gate_round=r)
    circ.apply_gate('CX', n-1, 0, gate_round=r)

    # Magnetic field
    for k in range(n):
        circ.apply_gate('RX', self_angle, k, gate_round=r)

y_data = [] 
for step in range(N):
    trotter_tensor(n, circ, dt, r=step, J=J, h=h, alpha=alpha)
    y_data.append(np.abs(circ.amplitude('01')))

# Generate x-axis values as timesteps from 0 to T
x_data = np.linspace(0, T, N)

# Plot the probabilities over time
plt.figure(figsize=(8, 5))
plt.plot(x_data,y_data)
plt.grid(True)
plt.legend()
plt.show()