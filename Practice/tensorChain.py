import quimb as qu
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt

# Variables
n = 2  # Number of qubits
N = 60  # Number of timesteps
T = 1.3  # Total time
dt = T / N  # Timestep size
J = 0.2  # Interaction strength
h = 1.2  # Magnetic field coupling
target = 1  # Target qubit
alpha = np.pi / 8  # Magnetic field angle of incidence

# Initialize the tensor network circuit
circ = qtn.Circuit(n)

# Apply X gate to the target qubit
circ.apply_gate('X', target)

# Function to apply the Trotterized time evolution operator
def trotter_tensor(n, circ, dt, r, J=1, h=1, alpha=0, factor=2):
    interaction_angle = -factor * J * dt
    self_angle = -factor * h * dt

    # Two-qubit interactions
    for k in range(n - 1):
        circ.apply_gate('CX', k, k + 1, gate_round=r)
        circ.apply_gate('RZ', interaction_angle, k, gate_round=r)
        circ.apply_gate('CX', k, k + 1, gate_round=r)

    # Periodic boundary condition
    circ.apply_gate('CX', n - 1, 0, gate_round=r)
    circ.apply_gate('RZ', interaction_angle, n - 1, gate_round=r)
    circ.apply_gate('CX', n - 1, 0, gate_round=r)

    # Single-qubit magnetic field rotations
    for k in range(n):
        circ.apply_gate('RX', self_angle, k, gate_round=r)

# Storage for probabilities
probabilities = []

# Time evolution loop
for step in range(N):
    trotter_tensor(n, circ, dt, r=step, J=J, h=h, alpha=alpha)
    state_vector = circ.psi.to_dense()
    probabilities.append(np.abs(state_vector)**2)

# Generate x-axis values as timesteps from 0 to T
x_data = np.linspace(0, T, N)

# Generate state labels
def generate_labels(n_qubits):
    """Generate binary labels for the states of n_qubits."""
    return [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]

labels = generate_labels(n)

# Plot the probabilities over time
plt.figure(figsize=(8, 5))
for i in range(2**n):
    plt.plot(x_data, [prob[i] for prob in probabilities], label=labels[i])
plt.xlabel('Time')
plt.ylabel('Amplitude squared')
plt.suptitle('Trotterized Time Evolution (Quimb)', fontsize=14)
plt.title('n: '+str(n)+" N: "+str(N)+" T: "+str(T), fontsize=14)
plt.grid(True)
plt.legend()
plt.show()