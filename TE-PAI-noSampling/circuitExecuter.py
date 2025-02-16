import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from circuitSimilatorVarying import parse

# Define the output directory
output_dir = os.path.join('TE-PAI-noSampling', 'data', 'plotting')
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Parse the data
circuit_dict = parse('TE-PAI-noSampling/data/circuits/N-1000-n-1-c-10-Δ-pi_over_1024-q-4-dT-0.005-T-1')

# Iterate over the dictionary to extract data
magnetizations = []
stds = []
for key, value in circuit_dict.items():
    params = value["params"]
    magnetizations = value["magnetizations"]
    stds = value["stds"]

# Unpack the params tuple
N, n_snapshot, circuits_count, delta_name, dT, numQs = params

# Prepare the data
means = magnetizations / numQs
stds = stds / numQs

# Save plotting data to a CSV file
filename = f"N-{N}-n-{n_snapshot}-c-{circuits_count}-Δ-{delta_name}-T-{float(T):.1f}-q-{numQs}.csv"
output_path = os.path.join(output_dir, filename)

times = np.arange(dT,T+dT,dT)

# Create DataFrame for plotting data
data_dict = {
    'x': times,
    'y': means,
    'errorbars': stds
}
df_plot = pd.DataFrame(data_dict)

try:
    df_plot.to_csv(output_path, index=False)
    print(f"Plotting data saved to: {output_path}")
except Exception as e:
    print(f"Error saving plotting data: {e}")

# Plotting
plt.errorbar(times, means, yerr=stds, fmt='o', ecolor='red', capsize=4, label="Data with Error Bars")
plt.xlabel('Time (t)')
plt.ylabel('Magnetization')
plt.title('Plot of Magnetization vs Time')
plt.legend()
plt.show()