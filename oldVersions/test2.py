from pyquest import Register, unitaries, operators
import numpy as np
import matplotlib.pyplot as plt

# Variables
n = 7   # Number of qubits
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

# Initialize the quantum register
reg = Register(n)

# Initial state: |0001000>
reg.apply_operator(unitaries.X(3))  # Flip the 4th qubit to 1

# Function to apply the Trotterized time evolution operator
def trotter_step(reg, omega, t, dt, n):
    # Apply H1: Local Z rotations
    for k in range(n):
        angle = -omega[k] * dt
        reg.apply_operator(unitaries.Rz(k, angle))
    
    # Apply H2: Interaction terms (nearest neighbors)
    for k in range(n - 1):  # Assuming open boundary conditions
        print(J(t))
        interaction_angle = -2 * J(t) * dt
        reg.apply_operator(unitaries.X([k, k + 1]))
        reg.apply_operator(unitaries.Rx(k + 1, interaction_angle))
        reg.apply_operator(unitaries.X([k, k + 1]))

    # Normalize the state
    reg /= np.sqrt(reg.total_prob)

# Storage for probabilities over time
probabilities = [reg.prob_of_all_outcomes([3])]

# Time evolution loop
for step in range(N):
    t = (step + 1) * dt
    trotter_step(reg, omega, t, dt, n)
    probabilities.append(reg.prob_of_all_outcomes([3]))

# Generate x-axis values as timesteps from 0 to T
x_data = np.linspace(0, T, N + 1)

# Plot the probabilities over time
plt.figure(figsize=(8, 5))
plt.plot(x_data, probabilities, label=['Probability of 4th Qubit in |0>', 'Probability of 4th Qubit in |1>'])
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Trotterized Time Evolution')
plt.grid(True)
plt.legend()
plt.show()
