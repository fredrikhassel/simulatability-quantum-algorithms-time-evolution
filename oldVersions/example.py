from pyquest import Register, Circuit, unitaries, gates
import numpy as np
import matplotlib.pyplot as plt

# Set up parameters
num_qubits = 2  # number of qubits
shots = 1024    # number of measurements

# Initialize a quantum circuit with specified number of qubits
reg = Register(num_qubits)

# Initializing a y-gate / pauli gate
h_gate = unitaries.H(0) # Targeting the first 

# Apply a CNOT gate with the first qubit as control and the second as target
cx_gate_2_1 = unitaries.X(1, controls=0)

m_gate = gates.M([0,1])

# an iterable of operations
op_iter = [h_gate, cx_gate_2_1, m_gate]

# Creating a quantum circuit from these operations
circ = Circuit(op_iter)

# Applying the circuit to our state
results = reg.apply_circuit(circ)

print(results)

'''

# Measure the qubits
results = qc.run(shots=shots)

# Process results
counts = circuit_results.get_counts(results)
print("Measurement Results:", counts)

# Visualize results (requires matplotlib for a bar chart of counts)

plt.bar(counts.keys(), counts.values(), color='blue')
plt.xlabel("Measurement Outcome")
plt.ylabel("Frequency")
plt.title("Measurement Results of the Circuit")
plt.show()
'''