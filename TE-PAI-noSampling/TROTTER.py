import SIMULATOR
from dataclasses import dataclass
import numpy as np
from dataclasses import dataclass
import numpy as np
from scipy.stats import binom
import os
import pandas as pd

"""This module contains the trotterization from the TE-PAI paper."""

@dataclass
class Trotter:

    def __init__(self, hamil, N, n_snapshot, c, Δ_name, T, numQs):
        (self.N, self.n_snapshot, self.c, self.Δ_name, self.T, self.numQs) = (N, n_snapshot, c, Δ_name, T, numQs)
        self.L = len(hamil)
        steps = np.linspace(0, T, N)
        self.terms = [hamil.get_term(t) for t in steps]

    def get_lie_PDF(self, points=1000):
        sn = 1000
        x = np.linspace(0, sn, points, dtype=int)
        pdf = binom.pmf(x, sn, self.run()[-1])
        data = [[2 * x / sn - 1, sn / 2 * val] for x, val in zip(x, pdf)]
        return zip(*data)

    def run(self, err=None, getGates = False):
        noisy = "_noisy" if err is not None else ""
        filename = f"TE-PAI-noSampling/data/plotting/lie-N-{self.N}-n-{self.n_snapshot}-c-{self.c}-Δ-{self.Δ_name}-T-{self.T}-q-{self.numQs}{noisy}.csv"
        gates_arr = []
        if not os.path.exists(filename):
            n = int(self.N / self.n_snapshot)
            for i in range(self.N):
                if i % n == 0:
                    gates_arr.append([])
                gates_arr[-1] += [
                    (pauli, 2 * coef * self.T / self.N, ind)
                    for (pauli, ind, coef) in self.terms[i]
                ]
            if not getGates:
                res = SIMULATOR.get_probs(self.numQs, gates_arr, self.n_snapshot, err=err)
                pd.DataFrame(res).to_csv(filename, index=False)
                return res, gates_arr
            else:
                return gates_arr
        else:
            if not getGates:
                data = pd.read_csv(filename).values
                return data[:, 0], gates_arr
            else:
                return gates_arr
            
    def run2(self):
        print("ding")
        "Build and return the second-order (Strang) Trotter gate sequence grouped into n_snapshot blocks."
        gates_arr = []
        n = max(1, int(self.N / self.n_snapshot))
        dt_angle = 2 * self.T / self.N

        for s in range(self.n_snapshot):
            start = s * n
            end = self.N if s == self.n_snapshot - 1 else min((s + 1) * n, self.N)
            block = []

            # Forward half-steps
            for i in range(start, max(start, end - 1)):
                for (pauli, ind, coef) in self.terms[i]:
                    block.append((pauli, coef * dt_angle * 0.5, ind))

            # Middle full-step (if the block has at least one element)
            if end > start:
                for (pauli, ind, coef) in self.terms[end - 1]:
                    block.append((pauli, coef * dt_angle, ind))

            # Backward half-steps
            for i in range(end - 2, start - 1, -1):
                for (pauli, ind, coef) in self.terms[i]:
                    block.append((pauli, coef * dt_angle * 0.5, ind))

            gates_arr.append(block)

        return gates_arr
