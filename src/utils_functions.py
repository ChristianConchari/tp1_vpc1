from typing import List
import numpy as np
import matplotlib.pyplot as plt

def plot_windows(windows: List[np.ndarray], title: str = "Sliding Windows") -> None:
    """
    Plots a grid of windows.

    Parameters:
    - windows (List[np.ndarray]): A list of windows to be plotted.
    - title (str): The title of the plot (default: "Sliding Windows").
    """
    # Calculate the number of windows
    num_windows = len(windows)

    # Calculate the grid size for subplots
    grid_size = int(np.ceil(np.sqrt(num_windows)))

    # Create a figure and axes for subplots
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axs = axs.flatten()

    # Plot each window
    for i, window in enumerate(windows):
        axs[i].imshow(window)
        axs[i].axis('off')

    # Turn off the remaining axes
    for i in range(len(windows), len(axs)):
        axs[i].axis('off')

    # Set the title and layout of the plot
    plt.suptitle(title)
    plt.tight_layout()

    # Display the plot
    plt.show()