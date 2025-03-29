# External packages
from functools import partial
from dataclasses import dataclass
import gc
import json
import uuid
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
import numpy as np
from scipy.stats import binom
import os
import pandas as pd

# Local functions and classes
from PAI import gamma, prob_list
from SAMPLING import batch_sampling
from SIMULATOR import get_probs

import os
import json
import multiprocessing as mp
from functools import partial

"""This module contains the TE-PAI algorithm from the TE-PAI paper, without resampling."""

@dataclass
class TE_PAI:

    def __init__(self, hamil, numQs, Δ, T, N, n_snap):
        (self.nq, self.n_snap, self.Δ, self.T, self.N) = (numQs, n_snap, Δ, T, N)
        self.L = len(hamil)
        steps = np.linspace(0, T, N)
        angles = [[2 * np.abs(coef) * T / N for coef in hamil.coefs(t)] for t in steps]
        n = int(N / n_snap)
        self.gam_list = [1] + [
            np.prod([gamma(angles[j], self.Δ) for j in range((i + 1) * n)])
            for i in range(n_snap)
        ]
        self.gamma = self.gam_list[-1] if N > 0 else 0
        self.probs = [prob_list(angles[i], Δ) for i in range(N)]
        self.terms = [hamil.get_term(t) for t in steps]
        self.overhead = np.exp(2 * hamil.l1_norm(T) * np.tan(Δ / 2))
        self.expected_num_gates = ((3 - np.cos(Δ)) / np.sin(Δ)) * hamil.l1_norm(T)
        self.rea_expect_num_gates = 0
        for prob in self.probs:
            for p in prob:
                self.rea_expect_num_gates += 1 - p[0]

    def sample_num_gates(self, n):
        res = batch_sampling(np.array(self.probs), n)
        return [sum(len(r) for r in re) for re in res]



    def run_te_pai(self, num_circuits, sign_file_path, gates_file_path, overhead, err=None):
        index = batch_sampling(np.array(self.probs), num_circuits)
        sign_list = []

        # Open file and write starting brace
        with open(gates_file_path, 'w') as f:
            f.write('{\n')
            first_entry = True

            with mp.Pool(4) as pool:
                for i, (sign_val, circuit) in enumerate(
                    pool.imap_unordered(partial(self.gen_rand_cir_with_details, err=err), index), start=1
                ):
                    sign_list.append(sign_val)

                    # Build and write the circuit
                    circuit_dict = {}
                    for snap_idx, snapshot in enumerate(circuit, start=1):
                        snapshot_dict = {}
                        for gate_idx, gate in enumerate(snapshot, start=1):
                            if not isinstance(gate, (list, tuple)) or len(gate) != 3:
                                print(f"Malformed gate at snapshot {snap_idx}, gate_idx {gate_idx}: {gate}")
                                continue
                            snapshot_dict[str(gate_idx)] = {
                                "gate_name": gate[0],
                                "angle": float(gate[1]),
                                "qubits": list(gate[2])
                            }
                        circuit_dict[str(snap_idx)] = snapshot_dict
                        del snapshot_dict

                    if not first_entry:
                        f.write(',\n')
                    first_entry = False
                    f.write(f'"{i}":')
                    json.dump(circuit_dict, f)
                    f.flush()
                    os.fsync(f.fileno())

                    # Kill memory now
                    del circuit_dict
                    del circuit
                    gc.collect()

                    if i % 50 == 0:
                        print(f"Wrote {i}/{num_circuits} circuits and cleaned memory")

            f.write('\n}')  # close JSON object

        # Save sign list
        sign_data = {"overhead": overhead}
        for i, sign_val in enumerate(sign_list):
            sign_data[str(i + 1)] = sign_val

        try:
            with open(sign_file_path, 'w') as file:
                json.dump(sign_data, file, indent=4)
            print(f"Sign list successfully saved to {sign_file_path}")
        except Exception as e:
            print(f"Error saving sign list file: {e}")


 

    def gen_rand_cir_with_details(self, index, err=None):
        (gates_arr, sign, n) = ([], 1, int(self.N / self.n_snap))
        for i, inde in enumerate(index):
            if i % n == 0:
                gates_arr.append([])
            for j, val in inde:
                (pauli, ind, coef) = self.terms[i][j]
                gate = None
                if val == 3:
                    sign *= -1
                    gate = (pauli, np.pi, ind)
                else:
                    gate = (pauli, np.sign(coef) * self.Δ, ind)
                gates_arr[-1].append(gate)
        return sign, gates_arr

    def gen_rand_cir(self, index, err=None):
            (gates_arr, sign, sign_list, n) = ([], 1, [], int(self.N / self.n_snap))
            for i, inde in enumerate(index):
                if i % n == 0:
                    gates_arr.append([])
                    sign_list.append(sign)
                for j, val in inde:
                    (pauli, ind, coef) = self.terms[i][j]
                    if val == 3:
                        sign *= -1
                        gates_arr[-1].append((pauli, np.pi, ind))
                    else:
                        gates_arr[-1].append((pauli, np.sign(coef) * self.Δ, ind))
            sign_list.append(sign)
            data = get_probs(self.nq, gates_arr, self.n_snap, err)
            return np.array([(sign_list[i] * self.gam_list[i], data[i]) for i in range(self.n_snap + 1)])  # type: ignore