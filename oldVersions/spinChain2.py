import numpy as np
from pyquest import Register, unitaries, operators

def get_hamiltonian(L, J, h, alpha=0):

    hamiltonian = 0

    # ZZ interaction
    for i in range(0,L-1):
        hamiltonian += unitaries.Z(i,i+1).as_matrix(L)*J
    
    # Magnetic field contribution
    for i in range(0,L):
        hamiltonian += unitaries.Rz(i, -h * np.sin(alpha)).as_matrix(L)
        hamiltonian += unitaries.Rx(i, -h * np.cos(alpha)).as_matrix(L)

    return hamiltonian

H = get_hamiltonian(L=2, J=0.2, h=1.0, alpha=np.pi / 8)

final_time = 1.6




