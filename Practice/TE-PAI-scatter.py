import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Paths to the data directory
data_dir = "data"

# Define the pattern for folder names
folder_pattern = re.compile(r'N-(\d+)-n-(\d+)-r-(\d+)-Δ-(\d+\.\d+)-T-(\d+\.\d+)-q-(\d+)')

# Find all matching folders
matching_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and folder_pattern.match(f)]

if len(matching_folders) == 0:
    folder_pattern = re.compile(r'N-(\d+)-n-(\d+)-r-(\d+)-Δ-(\d+\.\d+)-T-(\d+)-q-(\d+)')
    matching_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and folder_pattern.match(f)]

# Load sign data from CSV
sign_data_file = None
for folder in matching_folders:
    potential_file = os.path.join(data_dir, folder, "sign_data.csv")
    if os.path.exists(potential_file):
        sign_data_file = potential_file
        break

if not sign_data_file:
    raise FileNotFoundError("No sign_data.csv file found in the expected directories.")

sign_data = np.loadtxt(sign_data_file, delimiter=',')

# Display as a heatmap using Matplotlib
plt.figure(figsize=(10, 8))
cmap = ListedColormap(["red", "white", "green"])
plt.imshow(sign_data, cmap=cmap, aspect='auto')
plt.colorbar(ticks=[-1, 0, 1], label="Sign Values")
plt.grid(False)
plt.title("Sign Data Heatmap Visualization")
plt.show()
