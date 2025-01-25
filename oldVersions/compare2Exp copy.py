from pyquest import Register, unitaries, Circuit
import numpy as np
import matplotlib.pyplot as plt
from spinChain import trotter_step

# Variables
n = 6 # Number of qubits
N = 1000 # Number of timesteps
T = 10 # Total time
dt = T / N # Timestep size
J = 0.2 # Interaction strength
h = 1.2 # Magnetic field coupling
targets = [2,3] # 1 cubit
alpha = np.pi/8 # Magnetic field angle of incidence

# Initialize the quantum register
reg = Register(n)
for target in targets:
    reg.apply_operator(unitaries.X(target))  # Flip target Qubit to 1

# Creating a circuit for our operators
magnetizationOperators = []
spinAverageOperators = []
for i in range(n):
    magnetizationOperators.append(unitaries.Z([i]))
for i in range(n-1):
    spinAverageOperators.append(unitaries.Z([i],[i+1]))
magCirc = Circuit(magnetizationOperators)
spinCirc = Circuit(spinAverageOperators)

# Time evolution loop
for step in range(N):
    trotter_step(reg, dt, J, h, alpha)
    copy1 = reg.copy()
    copy2 = reg.copy()
    copy1.apply_circuit(magCirc)
    copy2.apply_circuit(spinCirc)
    


