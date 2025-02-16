import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from quimbParserCopy import parse
from collections import defaultdict

# Define the output directory
output_dir = os.path.join('TE-PAI-noSampling', 'data', 'plotting')
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Parse the data
circuit_dict = parse('TE-PAI-noSampling/data/circuits/N-1000-n-20-c-20-Δ-pi_over_1024-q-7/')

# Initialize dictionaries to store magnetizations and signs by time
magnetizations_by_time = defaultdict(list)
signs_by_time = defaultdict(list)
times = []
params = []
# Iterate over the dictionary to extract data
for key, value in circuit_dict.items():
    # Extract the time parameter (index 4 in the 'params' tuple)
    time_param = round(value["params"][4], 2)
    if params == []:
        params = value["params"]
    times.append(time_param)
    
    signs = value["sign_list"]
    mags = value["magnetizations"]
    print(mags)
    mags = [m * signs[str(i+1)][0] for i,m in enumerate(mags)]

    # Append magnetization data and signs data to the appropriate lists
    magnetizations_by_time[time_param].extend(mags)

# Unpack the params tuple
N, n_snapshot, circuits_count, delta_name, T, numQs = params

# Compute averages and standard deviations
averages = []
stds = []
for time in times:
    # Get the magnetizations and signs for the current time step
    magnetizations = magnetizations_by_time[time]    
    # Compute averages and stds
    averages.append((np.mean(magnetizations)) / numQs)
    stds.append(np.std(magnetizations) / numQs)

# Save plotting data to a CSV file
filename = f"N-{N}-n-{n_snapshot}-c-{circuits_count}-Δ-{delta_name}-T-{float(T):.1f}-q-{numQs}.csv"
output_path = os.path.join(output_dir, filename)

# Create DataFrame for plotting data
data_dict = {
    'x': times,
    'y': averages,
    'errorbars': stds
}
df_plot = pd.DataFrame(data_dict)

try:
    df_plot.to_csv(output_path, index=False)
    print(f"Plotting data saved to: {output_path}")
except Exception as e:
    print(f"Error saving plotting data: {e}")

# Plotting
plt.errorbar(times, averages, yerr=stds, fmt='o', ecolor='red', capsize=4, label="Data with Error Bars")
plt.xlabel('Time (t)')
plt.ylabel('Magnetization')
plt.title('Plot of Magnetization vs Time')
plt.legend()
plt.show()