from matplotlib import pyplot as plt
import numpy as np
from quimbParser import parse
from pyquest import Register, unitaries

# Getting the data
circuits, signs, mags, n, N, T = parse('TE-PAI-noSampling/data/circuits/')
for i,sign in enumerate(signs):
    mags[i] * sign

# Find the minimum non-zero length
min_length = min(len(entry) for entry in mags if len(entry) > 0)

# Filter for entries matching the minimum length
homogeneous_mags = [entry[:min_length] for entry in mags]
# Preparing it for plotting
std_mags = np.std(homogeneous_mags, axis=0)
std_mags = std_mags / n
averaged_mags = np.mean(homogeneous_mags, axis=0)
averaged_mags = averaged_mags / n
x_data = np.linspace(0, float(T), len(averaged_mags))

# Plotting
plt.errorbar(x_data, averaged_mags, yerr=std_mags, fmt='o', ecolor='red', capsize=4, label="Data with Error Bars")
plt.xlabel('Time (t)')
plt.ylabel('Magnetization')
plt.title('Plot of Magnitudes vs Time')
plt.show()