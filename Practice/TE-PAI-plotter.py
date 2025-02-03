from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_rgba
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Paths to the data directory
data_dir = "data"

# Define the pattern for folder names
folder_pattern = re.compile(r'N-(\d+)-n-(\d+)-r-(\d+)')

# Find all matching folders
matching_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and folder_pattern.match(f)]

# Lists to hold all the data for plotting
all_pai_data = []
all_lie_data = []
all_labels = []

for folder in matching_folders:
    # Extract N and n from the folder name
    match = folder_pattern.match(folder)
    if match:
        N = int(match.group(1))
        n = int(match.group(2))
        r = int(match.group(3))

        # Paths to the data files
        pai_pattern = "pai_snap"
        lie_file = f"lie/lie{N}_snap_step{n}.csv"

        # Read TE-PAI snapshot data
        pai_files = [os.path.join(data_dir, folder, f"{pai_pattern}{i}.csv") for i in range(n)]
        pai_data = []

        for file in pai_files:
            try:
                data = np.loadtxt(file, delimiter=',')
                if data.ndim == 1:  # Handle case where there's only one row of data
                    second_column = data[1]
                else:
                    second_column = data[:, 1]
                pai_data.append(np.mean(second_column))
            except Exception as e:
                print(f"Error reading {file}: {e}")

        # Read Lie-Trotter data
        try:
            lie_data = np.loadtxt(os.path.join(data_dir, folder, lie_file), delimiter=',')
            # Exclude the first two Lie-Trotter datapoints
            lie_data_filtered = lie_data[2:]
        except Exception as e:
            print(f"Error reading {lie_file}: {e}")
            lie_data_filtered = []

        # Add the data to the lists for plotting
        all_pai_data.append(pai_data)
        all_lie_data.append(lie_data_filtered)
        all_labels.append(f'N={N}, n={n}, r={r}')

# Plotting
plt.figure(figsize=(10, 6))

def generate_colors(labels):
    """
    Generate an array of colors for pyplot based on the labels in the format 'N={N}, n={n}, r={r}'.
    
    Parameters:
    labels (list): List of labels in the format 'N={N}, n={n}, r={r}'.
    
    Returns:
    list: List of colors for plotting.
    """
    # Dictionary to store a unique color for each N
    unique_N_colors = {}
    colors = []
    
    # Extract r values for later normalization
    r_values = [float(label.split(", ")[2].split("=")[1]) for label in labels]
    
    # Normalize r values
    min_r = min(r_values)
    max_r = max(r_values)
    
    # Iterate through each label
    for label in labels:
        # Parse N, n, r from the label
        parts = label.split(", ")
        N = int(parts[0].split("=")[1])
        r = float(parts[2].split("=")[1])
        
        # If this N has not been assigned a color, assign a unique color
        if N not in unique_N_colors:
            unique_N_colors[N] = plt.cm.tab10(len(unique_N_colors) % 10)  # Use tab10 colormap
        
        # Get the base color for this N
        base_color = unique_N_colors[N]
        
        # Adjust the brightness based on r (higher r means darker color)
        luminance = 0.5 + (0.5 * (r - min_r) / (max_r - min_r))  # Normalize r between min_r and max_r
        adjusted_color = np.array(base_color) * luminance
        colors.append(adjusted_color)
    
    return colors

colors = generate_colors(all_labels)

# Plot all TE-PAI data
for pai_data, label, color in zip(all_pai_data, all_labels, colors):
    plt.plot(range(len(pai_data)), pai_data, marker='o', label=f'TE-PAI {label}', c=color)

# Plot all Lie-Trotter data
for lie_data, label, color in zip(all_lie_data, all_labels, colors):
    if lie_data is not None:
        plt.scatter(range(len(lie_data)), lie_data, marker='x', c=color)

# Labels and legend
plt.title("TE-PAI vs Lie-Trotter Data")
plt.xlabel("Snapshot Index")
plt.ylabel("Data Points (Averaged Values)")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
plt.grid(True)

# Show plot
plt.show()
