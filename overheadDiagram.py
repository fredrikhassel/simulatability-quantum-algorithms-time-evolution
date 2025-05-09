import numpy as np
import matplotlib.pyplot as plt

# Constants
B_values = [7, 8, 9, 10]
nu = np.logspace(0, 6, 1000)  # ν from 1 to 1e6
pi_squared = np.pi ** 2
max_y = 100  # upper y-limit of interest

# Set up the figure
plt.figure(figsize=(6, 4.5))

for B in B_values:
    exponent = pi_squared * nu * 2**(-2 * B)
    y = np.exp(exponent)

    # Mask out values greater than 1000 (to ignore them)
    y = np.where(y > max_y, np.nan, y)
    y = np.clip(y, 1, max_y)  # keep y within safe log range

    plt.plot(nu, y, label=f'B = {B}')

# Plot formatting
plt.xscale('log')
plt.yscale('log')
plt.title("Measurement cost as a function of number of gates")
plt.xlabel(r'$\nu$')
plt.ylabel(r'$\exp(\pi^2 \nu 2^{-2B_{\min}})$')
plt.legend()
plt.grid(False)
plt.ylim(1, max_y)

plt.show()
