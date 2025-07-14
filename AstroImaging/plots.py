import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the .tbl file (assuming tab-separated, adjust delimiter if needed)
# Replace 'input.tbl' with the path to your .tbl file
data = pd.read_table('./AstroImaging/Raw Data.tbl', delim_whitespace=True)

# Extract relevant columns (assuming column names or indices match the provided data)
obs = data.iloc[:, 0].values  # First column (OBS)
flux = data.iloc[:, 14].values  # FLUX (based on provided data column index)
flux_err = data.iloc[:, 34].values  # FLUX_ERR
mag = data.iloc[:, 18].values  # MAG
mag_err = data.iloc[:, 22].values  # MAG_ERR

# Create a figure with subplots
plt.figure(figsize=(12, 10))

# Plot FLUX
plt.subplot(2, 2, 1)
plt.errorbar(obs, flux, yerr=flux_err, fmt='o-', color='blue', ecolor='red', capsize=3)
plt.xlabel('Observation Number')
plt.ylabel('FLUX')
plt.title('FLUX vs Observation')
plt.grid(True)

# Plot FLUX_ERR
plt.subplot(2, 2, 2)
plt.plot(obs, flux_err, 'o-', color='green')
plt.xlabel('Observation Number')
plt.ylabel('FLUX_ERR')
plt.title('FLUX_ERR vs Observation')
plt.grid(True)

# Plot MAG
plt.subplot(2, 2, 3)
plt.errorbar(obs, mag, yerr=mag_err, fmt='o-', color='purple', ecolor='orange', capsize=3)
plt.xlabel('Observation Number')
plt.ylabel('MAG')
plt.title('MAG vs Observation')
plt.grid(True)

# Plot MAG_ERR
plt.subplot(2, 2, 4)
plt.plot(obs, mag_err, 'o-', color='cyan')
plt.xlabel('Observation Number')
plt.ylabel('MAG_ERR')
plt.title('MAG_ERR vs Observation')
plt.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as a PNG file (downloads to the current working directory)
plt.savefig('plot_output.png', dpi=300, bbox_inches='tight')

# Display the plot (optional, for environments that support it)
plt.show()