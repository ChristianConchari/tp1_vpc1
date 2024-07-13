import numpy as np
from typing import Iterator, Tuple
import numpy as np

def sliding_window(
    image: np.ndarray,
    window_size: Tuple[int, int],
    step_size: int,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int]
    ) -> Iterator[Tuple[int, int, np.ndarray]]:
    """
    Generate sliding windows over an image.

    Args:
    image (numpy.ndarray): The input image.
    window_size (tuple): The size of the sliding window (width, height).
    step_size (int): The step size for sliding the window.
    x_range (tuple): The range of x-coordinates to slide the window over (start, end).
    y_range (tuple): The range of y-coordinates to slide the window over (start, end).

    Yields:
    tuple: A tuple containing the x-coordinate, y-coordinate, and the window image.

    """
    # Iterate over the y-coordinates within the specified range with the given step size
    for y in range(y_range[0], y_range[1], step_size):
        # Iterate over the x-coordinates within the specified range with the given step size
        for x in range(x_range[0], x_range[1], step_size):
            # Extract the window from the image based on the current x and y coordinates
            window = image[y:y + window_size[1], x:x + window_size[0]]
            # Check if the extracted window has the expected size
            if window.shape[0] == window_size[1] and window.shape[1] == window_size[0]:
                # Yield the x-coordinate, y-coordinate, and the window image
                yield (x, y, window)

def segment_region(image, x_start, x_end, y_start, y_end):
    """
    Segment a region of an image based on mean color values and standard deviations.

    Args:
        image (numpy.ndarray): The input image.
        x_start (int): The starting x-coordinate of the region.
        x_end (int): The ending x-coordinate of the region.
        y_start (int): The starting y-coordinate of the region.
        y_end (int): The ending y-coordinate of the region.

    Returns:
        tuple: A tuple containing the segmented image, the mask, and the windows.

    """
    # Define the size of the sliding window and the step size
    window_size = (50, 50)
    step_size = 25
    
    # Initialize lists to store mean colors and windows
    mean_colors = []
    windows = []
    
    # Iterate over sliding windows in the specified region
    for (x, y, window) in sliding_window(image, window_size, step_size, (x_start, x_end), (y_start, y_end)):
        # Calculate the mean color of the window
        mean_color = np.mean(window, axis=(0, 1))
        mean_colors.append(mean_color)
        windows.append(window)
        
    # Convert mean colors to a numpy array
    mean_colors = np.array(mean_colors)
    
    # Calculate the mean and standard deviation of each color channel
    mean_r = np.mean(mean_colors[:, 0])
    std_r = np.std(mean_colors[:, 0])
    mean_g = np.mean(mean_colors[:, 1])
    std_g = np.std(mean_colors[:, 1])
    mean_b = np.mean(mean_colors[:, 2])
    std_b = np.std(mean_colors[:, 2])
    
    # Calculate the threshold ranges for each color channel
    threshold_r = (mean_r - 2 * std_r, mean_r + 2 * std_r)
    threshold_g = (mean_g - 2 * std_g, mean_g + 2 * std_g)
    threshold_b = (mean_b - 2 * std_b, mean_b + 2 * std_b)

    # Create a mask based on the threshold ranges
    mask = ((image[:, :, 0] >= threshold_r[0]) & (image[:, :, 0] <= threshold_r[1]) &
            (image[:, :, 1] >= threshold_g[0]) & (image[:, :, 1] <= threshold_g[1]) &
            (image[:, :, 2] >= threshold_b[0]) & (image[:, :, 2] <= threshold_b[1]))
    
    # Create a segmented image by applying the mask
    segmented_img = np.zeros_like(image)
    segmented_img[mask] = image[mask]
    
    return segmented_img, mask, windows