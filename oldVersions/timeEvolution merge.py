from pyquest import Register, Circuit, unitaries, gates, operators
import numpy as np
import matplotlib.pyplot as plt


def J(t):
    return np.cos(20*np.pi*t)

# N values uniformly random in -1 to 1 range
def genOmega(N):
    return np.random.uniform(-1, 1, N)

# Variables
N     = 10  # Circuit depth
n     = 7   # Qubit number
omega = genOmega(n)
T     = 1   # total time
dt    = T/N # timestep
delta = 2e-7 * np.pi

# Gates
def Z(target):
    return unitaries.Rz(target, delta)
def sigmaDotProduct(k,l):
    X = np.matmul(unitaries.X(k).as_matrix(n), unitaries.X(l).as_matrix(n))
    Y = np.matmul(unitaries.Y(k).as_matrix(n), unitaries.Y(l).as_matrix(n))
    Z = np.matmul(unitaries.Z(k).as_matrix(n), unitaries.Z(l).as_matrix(n))
    return X+Y+Z

measurements = []

# \0000000>
reg = Register(n)

# \0001000>
reg.apply_operator(unitaries.X(3))

measurements.append(reg.prob_of_all_outcomes([3]))

# Time evolution
for step in range(N):
    
    t = step * dt
    
    # Apply omega_k Z_k terms
    for k in range(n):
        reg.apply_operator(operators.MatrixOperator(targets=[0,1,2,3,4,5,6], matrix=(Z(k).as_matrix(n) * omega[k])))  # Scaled Z rotation for each qubit based on omega

    # Apply J(t) sigmaDotProduct terms
    for k in range(n - 1):  # Assuming periodic boundary, can wrap around
        reg.apply_operator(operators.MatrixOperator(targets=[0,1,2,3,4,5,6],matrix=(sigmaDotProduct(k, k + 1) * J(t))))
    
    measurements.append(reg.prob_of_all_outcomes([3]))

# Generate x-axis values as timesteps from 0 to T
x_data = np.linspace(0, T, N+1)
y_data = list(y[0] for y in measurements)
print(y_data)
# Plotting the data
plt.figure(figsize=(8, 5))
plt.plot(x_data, y_data, marker='o', linestyle='-', color='b', label='Probability (%)')

# Adding labels and title
plt.xlabel('Timesteps (up to 1T)')
plt.ylabel('Probability (%)')
plt.title('Probability vs. Timesteps')
plt.legend()

# Displaying the plot
plt.grid(True)
plt.show()
