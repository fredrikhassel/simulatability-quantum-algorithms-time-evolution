from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy import integrate
from typing import Callable, List, Tuple
from itertools import product

"""This module contains the Hamiltonian class from the TE-PAI paper."""

@dataclass
class Hamiltonian:
    nqubits: int
    terms: List[Tuple[str, List[int], Callable[[float], float]]]

    def __post_init__(self):
        print("The number of qubit:" + str(self.nqubits))
        print("Number of terms in the Hamiltonian:" + str(len(self.terms)))

    @staticmethod
    def spin_chain_hamil(n, freqs, j=0.1):
        def J(t):
            return j#np.cos(20 * t * np.pi)#0.1#
        terms = [
            (gate, [k, (k + 1) % n], J)
            for k, gate in product(range(n), ["XX", "YY", "ZZ"])
        ]
        terms += [("Z", [k], lambda t, k=k: freqs[k]) for k in range(n)]
        return Hamiltonian(n, terms)
    
    @staticmethod
    def next_nearest_neighbor_hamil(n, freqs):
        def J1(t):
            return 0.1  #np.cos(20 * t * np.pi)
        def J2(t):
            return 0.05 #np.cos(10 * t * np.pi)
        terms = [
            (gate, [k, (k + 1) % n], J1)
            for k, gate in product(range(n), ["XX", "YY", "ZZ"])
        ]
        terms += [
            (gate, [k, (k + 2) % n], J2)
            for k, gate in product(range(n), ["XX", "YY", "ZZ"])
        ]
        terms += [("Z", [k], lambda t, k=k: freqs[k]) for k in range(n)]
        return Hamiltonian(n, terms)

    def get_term(self, t):
        return [(term[0], term[1], term[2](t)) for term in self.terms]

    def coefs(self, t: float):
        return [term[2](t) for term in self.terms]

    def l1_norm(self, T: float):
        fn = lambda t: np.linalg.norm(self.coefs(t), 1)
        return integrate.quad(fn, 0, T, limit=100)[0]

    def __len__(self):
        return len(self.terms)
