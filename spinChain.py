from pyquest import Register, unitaries
import numpy as np
import matplotlib.pyplot as plt

# Qubit labels
def generate_labels(n_qubits):
    """
    Generate an array of labels for n qubits where the least-significant bit corresponds to qubits[0].
    
    Args:
        n_qubits (int): Number of qubits.

    Returns:
        list: List of binary string labels in the correct order.
    """
    num_states = 2**n_qubits  # Total number of states
    labels = []
    
    for k in range(num_states):
        # Convert the index to a binary string and pad with leading zeros
        label = format(k, f'0{n_qubits}b')
        labels.append(label)
    
    return labels

# Function to apply the Trotterized time evolution operator
def trotter_step(n, reg, dt, J=1, h=1, alpha=0, factor=2):
    interaction_angle = -factor * J * dt
    self_angle = -factor * h * dt
    hamiltonian = 0

    for k in range(n-1):  # Loop over pairs of qubits
        # Apply the interaction between qubit k and k+1
        X = unitaries.X(k, controls=k + 1)
        reg.apply_operator(X)
        hamiltonian += X.as_matrix(n)
        # Apply the Rz rotation on qubit k
        Rz = unitaries.Rz(k, interaction_angle)
        reg.apply_operator(Rz)
        hamiltonian += Rz.as_matrix(n)
        # Apply the interaction again
        reg.apply_operator(X)
        hamiltonian += X.as_matrix(n)

    # Periodic boundary condition
    reg.apply_operator(unitaries.X(n-1, controls=0))
    hamiltonian += unitaries.X(n-1, controls=0).as_matrix(n)

    reg.apply_operator(unitaries.Rz(n-1, interaction_angle))
    hamiltonian += unitaries.Rz(n-1, interaction_angle).as_matrix(n)
    
    reg.apply_operator(unitaries.X(n-1, controls=0))
    hamiltonian += unitaries.X(n-1, controls=0).as_matrix(n)

    # Magnetic field
    for k in range(n):
        Rx = unitaries.Rx([k], self_angle)
        reg.apply_operator(Rx)
        hamiltonian += Rx.as_matrix(n)
    
    return hamiltonian

# Variables
n = 2 # Number of qubits
N = 60 # Number of timesteps
T = 1.3 # Total time
dt = T / N # Timestep size
J = 0.2 # Interaction strength
h = 1.2 # Magnetic field coupling
target = 1 # 1 cubit
alpha = np.pi/8 # Magnetic field angle of incidence

# Initialize the quantum register
reg = Register(n)
reg.apply_operator(unitaries.X(target-1))  # Flip target Qubit to 1

# Storage for probabilities over time
probabilities = [reg.prob_of_all_outcomes(range(n))]

# Time evolution loop
for step in range(N):
    trotter_step(n, reg, dt, J, h, alpha)  # Apply Trotter step
    probabilities.append(reg.prob_of_all_outcomes(range(n)))

# Generate x-axis values as timesteps from 0 to T
x_data = np.linspace(0, T, N + 1)
# Generate state vector labels
labels = generate_labels(n)

# Plot the probabilities over time
plt.figure(figsize=(8, 5))
for i in range(2**n):
    plt.plot(x_data, [y[i] for y in probabilities], label=labels[i])
plt.xlabel('Time')
plt.ylabel('Amplitude squared')
plt.title('n:  '+str(n)+" N: "+str(N)+" T: "+str(T), fontsize=14)
plt.suptitle('Trotterized Time Evolution (PyQuest)')
plt.grid(True)
plt.legend()
plt.show()