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
    def next_nearest_neighbor_hamil(n, freqs, j1=0.1, j2=0.05):
        def J1(t):
            return j1  #np.cos(20 * t * np.pi)
        def J2(t):
            return j2 #np.cos(10 * t * np.pi)
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
    
    @staticmethod
    def lattice_2d_hamil(n: int, J=1.0, freqs=None):
        """
        Nearest-neighbour 2D lattice with periodic (wraparound) boundary conditions.
        Required:
          - n : total number of qubits
        Optional:
          - J : scalar coupling or callable J(t)
          - freqs : None (zeros), scalar, or iterable of length n for onsite Z fields
        Chooses Lx x Ly with Lx*Ly=n that minimizes |Lx-Ly| (Lx ≥ Ly ≥ 2).
        Raises ValueError if no such rectangle exists (e.g., n is prime or n < 4).
        """
        import math

        if n < 4:
            msg = f"n={n} cannot form a rectangle with both sides ≥2."
            print(msg)
            raise ValueError(msg)

        # find factor closest to sqrt(n)
        root = int(math.isqrt(n))
        Lx = Ly = None
        for f in range(root, 1, -1):
            if n % f == 0:
                Ly = f
                Lx = n // f
                break
        if Lx is None or Ly is None or min(Lx, Ly) < 2:
            msg = f"n={n} has no factorization Lx x Ly with Lx, Ly ≥ 2 (likely prime)."
            print(msg)
            raise ValueError(msg)

        # coupling function
        Jfunc = J if callable(J) else (lambda t, J=J: J)

        def idx(x, y):
            return (x % Lx) + (y % Ly) * Lx

        terms = []
        # add each bond once: right and up neighbors
        for x in range(Lx):
            for y in range(Ly):
                i = idx(x, y)
                for j in (idx(x + 1, y), idx(x, y + 1)):
                    for gate in ["XX", "YY", "ZZ"]:
                        terms.append((gate, [i, j], Jfunc))

        # onsite Z fields
        if freqs is None:
            # default to zero field
            terms += [("Z", [k], (lambda w=0.0: (lambda t: w))()) for k in range(n)]
        elif hasattr(freqs, "__len__"):
            if len(freqs) != n:
                msg = f"freqs length {len(freqs)} != n={n}"
                print(msg)
                raise ValueError(msg)
            terms += [("Z", [k], (lambda w=freqs[k]: (lambda t: w))()) for k in range(n)]
        else:
            # scalar field
            terms += [("Z", [k], (lambda w=freqs: (lambda t: w))()) for k in range(n)]

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
