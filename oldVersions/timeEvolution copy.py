from pyquest import Register, Circuit, unitaries, gates, operators
import numpy as np

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
    X = unitaries.X(k).as_matrix(n)@unitaries.X(l).as_matrix(n)
    Y = unitaries.Y(k).as_matrix(n)@unitaries.Y(l).as_matrix(n)
    Z = unitaries.Z(k).as_matrix(n)@unitaries.Z(l).as_matrix(n)
    return X+Y+Z

# \0000000>
reg = Register(n)

# \0001000>
midX = unitaries.X(3)

# Iterable of gates
circIterable = [midX, gates.M(range(7))]

# Time evolution
for step in range(N):
    t = step * dt
    # Apply omega_k Z_k terms
    for k in range(n):
        circIterable.append(operators.MatrixOperator(targets=[0,1,2,3,4,5,6], matrix=(Z(k).as_matrix(n) * omega[k])))  # Scaled Z rotation for each qubit based on omega

    # Apply J(t) sigmaDotProduct terms
    for k in range(n - 1):  # Assuming periodic boundary, can wrap around
        circIterable.append(operators.MatrixOperator(targets=[0,1,2,3,4,5,6],matrix=(sigmaDotProduct(k, k + 1) * J(t))))
    
    # Measuring
    circIterable.append(gates.M(range(7)))

# Initializing circuit with gates
cir = Circuit(circIterable)

# Applying circuit and getting output
measurements = reg.apply_circuit(cir)

for measurement in measurements:
    print(measurement)
