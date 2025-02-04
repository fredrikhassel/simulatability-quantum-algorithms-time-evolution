import numpy as np
import matplotlib.pyplot as plt
import os

# Path to the data directory
data_dir = "data"

# Look for folders matching the desired pattern
matching_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

# Check if there are any matching folders
if not matching_folders:
    print("No matching data folders found.")
    exit()

# Look for sign_data.csv in the first matching folder
folder_path = os.path.join(data_dir, matching_folders[0])
sign_data_file_path = os.path.join(folder_path, "sign_data.csv")

if not os.path.exists(sign_data_file_path):
    print(f"sign_data.csv not found in {folder_path}.")
    exit()

# Load sign_data from CSV
try:
    sign_data = np.loadtxt(sign_data_file_path, delimiter=',', dtype=int)
except Exception as e:
    print(f"Error loading sign_data.csv: {e}")
    exit()

# Plot the grid
def plot_sign_data_grid(data):
    """
    Visualizes sign_data as a grid.

    Args:
        data (np.ndarray): 2D array of sign values (+1, -1, 0).
    """
    # Create color mapping
    color_map = np.full(data.shape, np.nan)  # Default transparent
    color_map[data == 1] = 1  # Green for +1
    color_map[data == -1] = -1  # Red for -1

    fig, ax = plt.subplots(figsize=(data.shape[1] / 2, data.shape[0] / 2))
    cmap = plt.cm.RdYlGn

    # Plot the grid with custom colors
    img = ax.imshow(color_map, cmap=cmap, interpolation='nearest', vmin=-1, vmax=1)

    # Formatting
    ax.set_xticks(np.arange(data.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    plt.title("Grid Visualization of sign_data")
    plt.show()

# Plot the sign_data grid
plot_sign_data_grid(sign_data)
