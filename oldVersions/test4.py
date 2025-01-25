from pyquest import Register, unitaries
import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 5   # Number of spins (qubits)
N = 100  # Number of timesteps
T = 1  # Total simulation time
dt = T / N  # Timestep size

# Interaction strength
def J(t):
    return np.cos(10 * np.pi * t)

# Local magnetic fields
def genOmega(n):
    return np.random.uniform(-1, 1, n)

omega = genOmega(n)

# Initialize the quantum register
reg = Register(n)

# Set the initial state: |1010101>
for i in range(n):
    if i % 2 == 0:
        reg.apply_operator(unitaries.X(i))

# Function to apply a Trotterized time evolution step
def trotter_step(reg, omega, t, dt, n):
    # Apply local Z rotations (H1)
    for k in range(n):
        angle = -omega[k] * dt
        reg.apply_operator(unitaries.Rz(k, angle))
    
    # Apply nearest-neighbor interactions (H2)
    for k in range(n - 1):  # Assuming open boundary conditions
        interaction_angle = -2 * J(t) * dt
        reg.apply_operator(unitaries.X([k, k + 1]))
        reg.apply_operator(unitaries.Rz(k + 1, interaction_angle))
        reg.apply_operator(unitaries.X([k, k + 1]))
    
    # Normalize the state
    reg /= np.sqrt(reg.total_prob)

# Storage for Z expectation values over time
z_expectations = np.zeros((N + 1, n))

# Compute initial Z expectation values
for k in range(n):
    z_expectations[0, k] = reg.prob_of_all_outcomes([k])[0]

# Time evolution loop
for step in range(N):
    t = (step + 1) * dt
    trotter_step(reg, omega, t, dt, n)
    # Compute Z expectation values for each spin
    for k in range(n):
        z_expectations[step + 1, k] = reg.prob_of_all_outcomes([k])[0]

# Generate x-axis values (time)
time = np.linspace(0, T, N + 1)

# Plot the Z expectation values over time
plt.figure(figsize=(10, 6))
for k in range(n):
    plt.plot(time, z_expectations[:, k], label=f'Spin {k + 1}')
plt.xlabel('Time')
plt.ylabel('Z Expectation Value')
plt.title('1D Spin Chain: Z Expectation Values Over Time')
plt.legend()
plt.grid(True)
plt.show()
