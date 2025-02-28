import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Base data directory
base_dir = "TE-PAI-noSampling/data/plotting"

# Patterns to match folder and file names
te_pai_pattern = re.compile(r'N-(\d+)-n-(\d+)-r-(\d+)-Δ-([\d\.]+)-T-((?:\d+\.\d+)|(?:\d+))-q-(\d+)')
quimb_pattern = re.compile(r'N-(\d+)-n-(\d+)-[cp]-(\d+)-Δ-(\w+)-T-([\d\.]+)-q-(\d+)-dT-([\d\.]+)\.csv')
lie_pattern = re.compile(r'lie-N-(\d+)-n-(\d+)-c-(\d+)-Δ-([\w\.]+)-T-((?:\d+\.\d+)|(?:\d+))-q-(\d+)(noisy)?\.csv')

# Lists to hold all the data for plotting
all_pai_means = []
all_pai_stds = []
all_lie_means = []
all_labels = []
quimb_data_points = []

# Read TE-PAI data
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path) and te_pai_pattern.match(folder):
        match = te_pai_pattern.match(folder)
        N, n, r, Delta, T, q = map(match.group, range(1, 7))
        Delta, T = float(Delta), float(T)

        # Paths to the data files
        pai_pattern = "pai_snap"
        lie_file = f"lie/lie{N}_snap_step{n}.csv"

        # Read TE-PAI snapshot data
        pai_means = []
        pai_stds = []
        pai_files = [os.path.join(folder_path, f"{pai_pattern}{i}.csv") for i in range(int(n)+1)]

        for file in pai_files:
            print(len(pai_means))
            try:
                data = np.genfromtxt(file, delimiter=',', dtype=float, skip_header=1)  # Skip potential headers
                if data.ndim == 1:
                    second_column = data[1]  # Single row case
                else:
                    second_column = data[:, 1]  # General case: extract second column
                
                pai_means.append(np.nanmean(second_column))  # Use nanmean to ignore NaNs
                pai_stds.append(np.nanstd(second_column))  # Use nanstd to ignore NaNs
            except Exception as e:
                print(f"Error reading {file}: {e}")

        # Read Lie-Trotter data
        lie_means = None
        try:
            lie_data = np.loadtxt(os.path.join(folder_path, lie_file), delimiter=',')
            if lie_data.ndim == 1:  # Single-column case
                lie_means = lie_data[2:]
            else:
                lie_means = lie_data[:, 1][2:]  # Exclude the first two Lie-Trotter datapoints
        except Exception as e:
            print(f"Error reading {lie_file}: {e}")

        # Append TE-PAI data only if valid
        if pai_means and lie_means is not None:
            all_pai_means.append(pai_means)
            all_pai_stds.append(pai_stds)
            all_lie_means.append(lie_means)
            all_labels.append(f'TE-PAI: N={N}, n={n}, r={r}, Δ={Delta}, T={T}, q={q}')

# Read QUIMB data
trotter_data = []
for file in os.listdir(base_dir):
    if quimb_pattern.match(file):
        file_path = os.path.join(base_dir, file)
        try:
            data = np.loadtxt(file_path, delimiter=',', skiprows=1)
            x_values, y_values, error_bars = data[:, 0], data[:, 1], data[:, 2]
            quimb_data_points.append((x_values, y_values, error_bars, f"QUIMB: {file}"))
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if lie_pattern.match(file):
        match = lie_pattern.match(file)
        N, n, r, Delta, T, q = map(match.group, range(1, 7))
        T = float(T)

        file_path = os.path.join(base_dir, file)
        try:
            data = np.loadtxt(file_path, delimiter=',')
            data = data[1:]

            if data.ndim != 1:
                raise ValueError(f"Unexpected data shape in {file}: {data.shape}")

            time_steps = np.linspace(0, T, len(data))
            trotter_data.append([data, time_steps, f' N={N}, T={T}, q={q}'])
        except Exception as e:
            print(f"Error reading {file}: {e}")

# Plotting
plt.figure(figsize=(12, 6))

# Plot TE-PAI data
for pai_means, pai_stds, lie_means, label in zip(all_pai_means, all_pai_stds, all_lie_means, all_labels):
    if len(pai_means) > 0 and lie_means is not None:
        xs = np.linspace(0, 0.1, len(pai_means))
        #xs = [0, .01, .02, .03, .04, .05, .06, .07, .08, .09, .1]
        plt.errorbar(xs, pai_means, yerr=pai_stds, fmt='-o', label=f"{label} (PAI)")
        #plt.plot(xs, lie_means, '--x', label=f"{label} (Lie-Trotter)")

# Plot QUIMB data
for x_values, y_values, error_bars, label in quimb_data_points:
    if len(x_values) > 0:
        plt.errorbar(x_values, y_values, yerr=error_bars, fmt='-s', label=label)

# Plot Lie data
for trotter in trotter_data:
    if trotter is not None and len(trotter[0]) > 0:
        plt.plot(trotter[1], trotter[0], '--x', label=f"Lie-Trotter: {trotter[2]}")

plt.title("X expectation value over time")
plt.xlabel("Time")
plt.ylabel("(X+1)/2")
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
