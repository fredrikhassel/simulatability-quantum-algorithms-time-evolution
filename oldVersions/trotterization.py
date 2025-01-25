from pyquest import Register, unitaries, operators
import numpy as np
import matplotlib.pyplot as plt

# Variables
n = 2   # Number of qubits
N = 100  # Number of timesteps
T = 1  # Total time
dt = T / N  # Timestep size
J = 1

# Initialize the quantum register
reg = Register(n)

# Initial state: |01>
reg.apply_operator(unitaries.X(1))  # Flip the first qubit to 1

# Function to apply the Trotterized time evolution operator
def trotter_step(reg, dt):
    interaction_angle = -2 * J * dt

    for k in range(n - 1):  # Assuming open boundary conditions

        reg.apply_operator(unitaries.Z([k], [k + 1]))
        reg.apply_operator(unitaries.Rz(k, interaction_angle))
        reg.apply_operator(unitaries.Z([k], [k + 1]))

    # boundary condition
    reg.apply_operator(unitaries.Z([n-1], [1]))
    reg.apply_operator(unitaries.Rz(k, interaction_angle))
    reg.apply_operator(unitaries.Z([n-1], [1]))

# Storage for probabilities over time
probabilities = [reg.prob_of_all_outcomes([1])]

# Time evolution loop
for step in range(N):
    t = (step + 1) * dt
    trotter_step(reg,dt)
    probabilities.append(reg.prob_of_all_outcomes([1]))

# Generate x-axis values as timesteps from 0 to T
x_data = np.linspace(0, T, N + 1)

# Plot the probabilities over time
plt.figure(figsize=(8, 5))
plt.plot(x_data, probabilities, label=['Probability of first Qubit in |0>', 'Probability of first Qubit in |1>'])
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Trotterized Time Evolution')
plt.grid(True)
plt.legend()
plt.show()
