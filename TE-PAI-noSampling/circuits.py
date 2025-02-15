import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from quimbParserCopy import parse

# Define the output directory
output_dir = os.path.join('TE-PAI-noSampling', 'data', 'plotting')
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Parse the data
circuits, signs, mags, overhead, params = parse('TE-PAI-noSampling/data/circuits/')
N, n_snapshot, circuits_count, delta_name, T, numQs = params
numQs = int(numQs)

std_devs = np.std(mags, axis=0) / numQs

print(signs)

for i,circuitMag in enumerate(mags):
    sign = list(signs[i][0].values())[0]

    print("Circuit: "+str(i+1) + " Sign: "+str(signs[i]))
    print(circuitMag)
    mags[i] = [m * sign for m in mags[i]]

averages = np.average(mags, axis=0) / numQs

x_data = np.linspace(0, float(T), len(averages))

# Save plotting data to a CSV file
filename = f"N-{N}-n-{n_snapshot}-c-{circuits_count}-Δ-{delta_name}-T-{float(T):.1f}-q-{numQs}.csv"
output_path = os.path.join(output_dir, filename)

# Create DataFrame for plotting data
data_dict = {
    'x': x_data,
    'y': averages,
    'errorbars': std_devs
}
df_plot = pd.DataFrame(data_dict)

try:
    df_plot.to_csv(output_path, index=False)
    print(f"Plotting data saved to: {output_path}")
except Exception as e:
    print(f"Error saving plotting data: {e}")
 
# Plotting
plt.errorbar(x_data, averages, yerr=std_devs, fmt='o', ecolor='red', capsize=4, label="Data with Error Bars")
plt.xlabel('Time (t)')
plt.ylabel('Magnetization')
plt.title('Plot of Magnetization vs Time')
plt.legend()
plt.show()