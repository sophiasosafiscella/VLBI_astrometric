import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata
import sys

sns.set_context('poster')
sns.set_style('ticks')

df = pd.read_pickle("./results/posteriors_2/J0030+0451_posteriors_results_71.pkl").dropna(how='any')

# Function to create contour plot
def plot_contour(df, x_col, y_col, w_col, ax):
    # Extract columns
    x = df[x_col]
    y = df[y_col]
    w = df[w_col]

    # Create grid values first
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate w values on grid
    zi = griddata((x, y), w, (xi, yi), method='linear')

    # Plot contour
    contour = ax.contourf(xi, yi, zi, levels=15, cmap="viridis")
    plt.colorbar(contour, ax=ax, label=w_col)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
#    ax.set_title(f'{x_col} vs {y_col} with {w_col} as color')

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
axs[0, 1].axis('off')

# Plot each pair
plot_contour(df, 'PMRA', 'PMDEC', 'posterior', axs[0, 0])
plot_contour(df, 'PMRA', 'PX', 'posterior', axs[1, 0])
plot_contour(df, 'PMDEC', 'PX', 'posterior', axs[1, 1])

plt.tight_layout()
plt.show()
