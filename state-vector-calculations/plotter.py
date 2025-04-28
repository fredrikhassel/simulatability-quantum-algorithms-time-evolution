import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the data folders
DATA_DIR = 'data'

# Regex pattern to match folder names and extract parameters
pattern = re.compile(
    r'^N-(?P<N>\d+)-n-(?P<n>\d+)-r-(?P<r>\d+)-Δ-(?P<Delta>[\d\.eE+\-]+)-T-(?P<T>[\d\.eE+\-]+)-q-(?P<q>\d+)$'
)

# Prepare the plot
plt.figure(figsize=(10, 6))

# Loop over each subfolder in the data directory
for folder in os.listdir(DATA_DIR):
    match = pattern.match(folder)
    if not match:
        continue

    # Extract parameters from folder name
    params = match.groupdict()
    N = int(params['N'])
    n = int(params['n'])
    r = int(params['r'])
    Delta = float(params['Delta'])
    T = float(params['T'])
    q = int(params['q'])

    # Time step
    dt = T / n
    times = np.arange(n) * dt

    # Collect PAI mean and std
    pai_means = []
    pai_stds = []
    for i in range(n):
        pai_file = os.path.join(DATA_DIR, folder, f'pai_snap{i}.csv')
        if not os.path.isfile(pai_file):
            raise FileNotFoundError(f"Missing file: {pai_file}")
        df = pd.read_csv(pai_file)
        # Second column (index 1), ignoring header row automatically
        col = df.iloc[:, 1]
        pai_means.append(col.mean())
        pai_stds.append(col.std())

    pai_means = np.array(pai_means)
    pai_stds = np.array(pai_stds)

    # Load LIE data
    lie_folder = os.path.join(DATA_DIR, folder, 'lie')
    lie_file = os.path.join(lie_folder, f'lie{N}_snap_step{n}.csv')
    if not os.path.isfile(lie_file):
        raise FileNotFoundError(f"Missing LIE file: {lie_file}")
    df_lie = pd.read_csv(lie_file)
    lie_series = df_lie.iloc[1:, 0].values

    # Construct a label for this dataset
    label = f"N={N}, n={n}, r={r}, Δ={Delta:.6g}, T={T}, q={q}"

    # Plot PAI with error bars
    plt.errorbar(times, pai_means, yerr=pai_stds, fmt='o-', capsize=4, label=f'{label} PAI')
    # Plot LIE data
    plt.plot(times, lie_series, '--', marker='x', label=f'{label} LIE')

# Customize the plot
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('PAI vs LIE over Time for Various Parameters')
plt.legend(fontsize='small', loc='best')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()