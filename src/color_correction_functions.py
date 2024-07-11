import numpy as np

def convert_to_chromatic_coordinates(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to chromatic coordinates.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image.
        
    Returns
    -------
    np.ndarray
        Chromatic coordinates image.
    """
    # Ensure the image is in float32 format to avoid overflow
    image = image.astype(np.float32)
    # Initialize the output image with the same shape as the input image
    chromatic_coordinates = np.zeros_like(image)
    # Calculate the sum of the channels for each pixel
    channels_sum = np.sum(image, axis=2, keepdims=True)
    # Avoid division by zero by setting zeros to one
    channels_sum[channels_sum == 0] = 1.0
    # Calculate the chromatic coordinates by dividing each channel by the sum of the channels
    chromatic_coordinates = image / channels_sum
    # Convert the result to an 8-bit image (optional: scale to 0-255 for display purposes)
    chromatic_coordinates = (chromatic_coordinates * 255).astype(np.uint8)
    
    return chromatic_coordinates


def white_patch_algorithm(image: np.ndarray) -> np.ndarray:
    """
    Apply the white patch algorithm to an RGB image.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image.
        
    Returns
    -------
    np.ndarray
        Image after applying the white patch algorithm.
    """
    # Ensure the image is in float32 format to avoid overflow
    image = image.astype(np.float32)
    # Identify the brightest pixel in each channel of the image
    brightest_pixels = np.max(image, axis=(0, 1))
    # Calculate the scaling factors for each channel
    scaling_factors = 255.0 / brightest_pixels
    # Apply the scaling factors to the image
    white_balanced_image = image * scaling_factors
    # Clip the result to be in the range [0, 255]
    white_balanced_image = np.clip(white_balanced_image, 0, 255)
    # Convert the result to an 8-bit image
    white_balanced_image = white_balanced_image.astype(np.uint8)
    
    return white_balanced_image