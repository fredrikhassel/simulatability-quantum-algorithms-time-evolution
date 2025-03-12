import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Paths to the data directory
data_dir = "data"

# Define the pattern for folder names
folder_pattern = re.compile(r'N-(\d+)-n-(\d+)-r-(\d+)-Δ-(\d+\.\d+)-T-(\d+\.\d+)-q-(\d+)')

# Find all matching folders
matching_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and folder_pattern.match(f)]

if len(matching_folders) == 0:
    folder_pattern = re.compile(r'N-(\d+)-n-(\d+)-r-(\d+)-Δ-(\d+\.\d+)-T-(\d+)-q-(\d+)')
    matching_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and folder_pattern.match(f)]

# Lists to hold all the data for plotting
all_pai_means = []
all_pai_stds = []
all_lie_means = []
all_labels = []

for folder in matching_folders:
    # Extract N and n from the folder name
    match = folder_pattern.match(folder)
    if match:
        N = int(match.group(1))
        n = int(match.group(2))
        r = int(match.group(3))
        Δ = float(match.group(4))
        T = float(match.group(5))
        q = int(match.group(6))

        # Paths to the data files
        pai_pattern = "pai_snap"
        lie_file = f"lie/lie{N}_snap_step{n}.csv"

        # Read TE-PAI snapshot data
        pai_means = []
        pai_stds = []

        pai_files = [os.path.join(data_dir, folder, f"{pai_pattern}{i}.csv") for i in range(n)]
        for file in pai_files:
            try:
                data = np.loadtxt(file, delimiter=',')
                if data.ndim == 1:  # Handle case where there's only one row of data
                    second_column = data[1]
                else:
                    second_column = data[:, 1]
                pai_means.append(np.mean(second_column))
                pai_stds.append(np.std(second_column))
            except Exception as e:
                print(f"Error reading {file}: {e}")

        # Read Lie-Trotter data
        try:
            lie_data = np.loadtxt(os.path.join(data_dir, folder, lie_file), delimiter=',')
            # Exclude the first two Lie-Trotter datapoints
            lie_data_filtered = lie_data[2:]
            lie_means = lie_data_filtered
        except Exception as e:
            print(f"Error reading {lie_file}: {e}")
            lie_means = None

        # Add the data to the lists for plotting
        all_pai_means.append(pai_means)
        all_pai_stds.append(pai_stds)
        all_lie_means.append(lie_means)
        all_labels.append(f'N={N}, n={n}, r={r}, Δ={Δ}, T={T}, q={q}')

# Plotting
plt.figure(figsize=(10, 6))

def generate_colors(labels):
    """
    Generate an array of colors for pyplot based on the labels in the format 'N={N}, n={n}, r={r}'.
    """
    unique_N_colors = {}
    colors = []
    r_values = [float(label.split(", ")[2].split("=")[1]) for label in labels]
    min_r = min(r_values) if r_values != [] else 1
    max_r = max(r_values) if r_values != [] else 1

    for label in labels:
        parts = label.split(", ")
        N = int(parts[0].split("=")[1])
        r = float(parts[2].split("=")[1])
        if N not in unique_N_colors:
            unique_N_colors[N] = plt.cm.tab10(len(unique_N_colors) % 10)
        base_color = unique_N_colors[N]
        luminance = 0.5 + (0.5 * (r - min_r) / (max_r - min_r)) if max_r != min_r else 1
        adjusted_color = np.array(base_color) * luminance
        colors.append(adjusted_color)
    return colors

colors = generate_colors(all_labels)
colors = ["r","g","b","c","m","y","k","w"]

# Plot all TE-PAI data with error bars
for pai_mean, pai_std, label, color in zip(all_pai_means, all_pai_stds, all_labels, colors):
    parts = label.split(", ")
    N = int(parts[0].split("=")[1])
    r = float(parts[2].split("=")[1])
    T = float(parts[4].split("=")[1])
    q = int(parts[5].split("=")[1])
    label ="N: "+str(N)+" r: "+str(r)+" T: "+str(T)+" q: "+str(q)
    
    x_values = range(len(pai_mean))
    plt.errorbar(np.array(x_values)*0.01, pai_mean, yerr=pai_std, fmt='o', capsize=5, color=color, label=f'{label}')

# Plot Lie-Trotter data without error bars
for lie_mean, label, color in zip(all_lie_means, all_labels, colors):
    if lie_mean is not None:
        plt.scatter(np.array(range(len(lie_mean)))*0.01, [lie_mean], marker='x', color=color)

# Labels and legend
plt.title("TE-PAI vs Lie-Trotter Data")
plt.xlabel("Time")
plt.ylabel("X expectation value")
plt.legend()
plt.grid(True)

plt.show()
