import numpy as np
import matplotlib.pyplot as plt
import os

# Paths to the data files
data_dir = "data"
pai_pattern = "pai_snap"
lie_file = "lie/lie20_snap_step10.csv"

# Read TE-PAI snapshot data
pai_files = [os.path.join(data_dir, f"{pai_pattern}{i}.csv") for i in range(10)]
pai_data = []

# Load data from each snapshot file and compute mean of the second column
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
    lie_data = np.loadtxt(os.path.join(data_dir, lie_file), delimiter=',')
    if lie_data.ndim == 1:  # Handle case where there's only one row of data
        second_column = lie_data[1]
    else:
        second_column = lie_data[:, 1]
    # Exclude the first two Lie-Trotter datapoints
    lie_snapshots = list(range(2, len(second_column)))
    lie_data_filtered = second_column[2:]
except Exception as e:
    print(f"Error reading {lie_file}: {e}")
    lie_snapshots = []
    lie_data_filtered = []

# Plotting
plt.figure(figsize=(10, 6))

# TE-PAI plot
plt.plot(range(len(pai_data)), pai_data, marker='o', label='TE-PAI (Averaged Snapshots)')

# Lie-Trotter plot
if lie_data_filtered is not None:
    plt.axhline(y=np.mean(second_column), color='r', linestyle='--', label='Lie-Trotter (Mean)')
    plt.scatter(lie_snapshots, lie_data_filtered, color='purple', marker='x', label='Lie-Trotter (Snapshots Excluding First Two)')

# Labels and legend
plt.title("TE-PAI vs Lie-Trotter Data")
plt.xlabel("Snapshot Index")
plt.ylabel("Average of Second Column")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
