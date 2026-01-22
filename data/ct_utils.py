"""
CT Utilities Module for Deliverable 2
BMEN 509/623 - Introduction to Biomedical Imaging

Helper functions for CT reconstruction and analysis.
Students may use these utilities or implement their own.
"""

import numpy as np
from typing import Tuple, Optional


def create_shepp_logan_phantom(size: int = 256) -> np.ndarray:
    """
    Create a Shepp-Logan phantom for CT reconstruction testing.
    
    The Shepp-Logan phantom is a standard test image consisting of
    overlapping ellipses with different intensities representing
    different tissue types in a head cross-section.
    
    Parameters
    ----------
    size : int
        Size of the output image (size x size pixels)
        
    Returns
    -------
    phantom : ndarray
        2D array containing the phantom image with values
        representing linear attenuation coefficients (relative)
    """
    phantom = np.zeros((size, size), dtype=np.float64)
    
    # Define ellipses: (center_x, center_y, axis_a, axis_b, angle, intensity)
    # Values are normalized to image size
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0, 2.0),           # Outer skull
        (0.0, -0.0184, 0.6624, 0.874, 0, -0.98),  # Brain
        (0.22, 0.0, 0.11, 0.31, -18, -0.02),      # Left ventricle
        (-0.22, 0.0, 0.16, 0.41, 18, -0.02),      # Right ventricle
        (0.0, 0.35, 0.21, 0.25, 0, 0.01),         # Top structure
        (0.0, 0.1, 0.046, 0.046, 0, 0.01),        # Small feature 1
        (0.0, -0.1, 0.046, 0.046, 0, 0.01),       # Small feature 2
        (-0.08, -0.605, 0.046, 0.023, 0, 0.01),   # Bottom left
        (0.0, -0.605, 0.023, 0.023, 0, 0.01),     # Bottom center
        (0.06, -0.605, 0.023, 0.046, 0, 0.01),    # Bottom right
    ]
    
    # Create coordinate grids
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    for cx, cy, a, b, angle, intensity in ellipses:
        # Rotate coordinates
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        X_rot = cos_t * (X - cx) + sin_t * (Y - cy)
        Y_rot = -sin_t * (X - cx) + cos_t * (Y - cy)
        
        # Ellipse equation
        mask = (X_rot / a) ** 2 + (Y_rot / b) ** 2 <= 1
        phantom[mask] += intensity
    
    return phantom


def forward_project(image: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Compute the Radon transform (sinogram) of an image.
    
    Parameters
    ----------
    image : ndarray
        2D input image to project
    angles : ndarray
        Array of projection angles in degrees
        
    Returns
    -------
    sinogram : ndarray
        2D sinogram where each row is a projection at the corresponding angle
    """
    from scipy.ndimage import rotate
    
    n_angles = len(angles)
    n_detectors = image.shape[0]
    sinogram = np.zeros((n_angles, n_detectors))
    
    for i, angle in enumerate(angles):
        # Rotate image and sum along one axis
        rotated = rotate(image, angle, reshape=False, order=1)
        sinogram[i, :] = np.sum(rotated, axis=0)
    
    return sinogram


def create_ramp_filter(size: int, filter_type: str = 'ramp') -> np.ndarray:
    """
    Create a frequency-domain filter for filtered back projection.
    
    Parameters
    ----------
    size : int
        Length of the filter (should match projection length)
    filter_type : str
        Type of filter: 'ramp', 'hamming', 'hann', or 'cosine'
        
    Returns
    -------
    filter : ndarray
        Frequency-domain filter
    """
    # Frequency axis
    freq = np.fft.fftfreq(size)
    
    # Ramp filter (absolute value of frequency)
    ramp = np.abs(freq)
    
    if filter_type == 'ramp':
        return ramp
    elif filter_type == 'hamming':
        # Hamming window
        n = np.arange(size)
        window = 0.54 - 0.46 * np.cos(2 * np.pi * n / size)
        return ramp * np.fft.fftshift(window)
    elif filter_type == 'hann':
        # Hann window
        n = np.arange(size)
        window = 0.5 * (1 - np.cos(2 * np.pi * n / size))
        return ramp * np.fft.fftshift(window)
    elif filter_type == 'cosine':
        # Cosine window
        window = np.cos(np.pi * freq / (2 * np.max(np.abs(freq)) + 1e-10))
        return ramp * window
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def add_beam_hardening(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """
    Add simulated beam hardening artifact to a CT image.
    
    Beam hardening causes "cupping" - lower values in the center
    of homogeneous regions.
    
    Parameters
    ----------
    image : ndarray
        Input CT image
    strength : float
        Strength of the beam hardening effect (0-1)
        
    Returns
    -------
    corrupted : ndarray
        Image with beam hardening artifact
    """
    size = image.shape[0]
    center = size // 2
    
    # Create radial distance map
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    r_normalized = r / (size / 2)
    
    # Cupping profile (higher at edges, lower in center)
    cupping = strength * (1 - r_normalized ** 2)
    cupping = np.clip(cupping, 0, strength)
    
    # Apply only where there is signal
    mask = image > 0.1 * np.max(image)
    corrupted = image.copy()
    corrupted[mask] -= cupping[mask] * np.max(image)
    
    return corrupted


def add_ring_artifact(image: np.ndarray, n_rings: int = 3, 
                      strength: float = 0.1) -> np.ndarray:
    """
    Add simulated ring artifacts to a CT image.
    
    Ring artifacts appear as concentric circles due to
    miscalibrated detector elements.
    
    Parameters
    ----------
    image : ndarray
        Input CT image
    n_rings : int
        Number of ring artifacts to add
    strength : float
        Intensity of the rings relative to image max
        
    Returns
    -------
    corrupted : ndarray
        Image with ring artifacts
    """
    size = image.shape[0]
    center = size // 2
    
    # Create radial distance map
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    
    corrupted = image.copy()
    
    # Add rings at random radii
    np.random.seed(42)  # Reproducible
    ring_radii = np.random.uniform(0.1, 0.8, n_rings) * (size / 2)
    ring_widths = np.random.uniform(1, 3, n_rings)
    
    for radius, width in zip(ring_radii, ring_widths):
        ring_mask = np.abs(r - radius) < width
        ring_intensity = strength * np.max(image) * np.random.choice([-1, 1])
        corrupted[ring_mask] += ring_intensity
    
    return corrupted


def calculate_rmse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate root mean square error between two images.
    
    Parameters
    ----------
    image1, image2 : ndarray
        Images to compare (must be same shape)
        
    Returns
    -------
    rmse : float
        Root mean square error
    """
    return np.sqrt(np.mean((image1 - image2) ** 2))


def calculate_ssim(image1: np.ndarray, image2: np.ndarray,
                   window_size: int = 11) -> float:
    """
    Calculate structural similarity index (simplified version).
    
    Parameters
    ----------
    image1, image2 : ndarray
        Images to compare
    window_size : int
        Size of the sliding window
        
    Returns
    -------
    ssim : float
        Structural similarity index (0-1)
    """
    from scipy.ndimage import uniform_filter
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Local means
    mu1 = uniform_filter(image1, window_size)
    mu2 = uniform_filter(image2, window_size)
    
    # Local variances and covariance
    sigma1_sq = uniform_filter(image1 ** 2, window_size) - mu1 ** 2
    sigma2_sq = uniform_filter(image2 ** 2, window_size) - mu2 ** 2
    sigma12 = uniform_filter(image1 * image2, window_size) - mu1 * mu2
    
    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    return np.mean(ssim_map)


def mu_to_hu(mu: np.ndarray, mu_water: float = 0.019) -> np.ndarray:
    """
    Convert linear attenuation coefficient to Hounsfield units.
    
    Parameters
    ----------
    mu : ndarray
        Linear attenuation coefficient(s) in mm^-1
    mu_water : float
        Linear attenuation coefficient of water (default: ~70 keV)
        
    Returns
    -------
    hu : ndarray
        Hounsfield unit values
    """
    return 1000 * (mu - mu_water) / mu_water


def hu_to_mu(hu: np.ndarray, mu_water: float = 0.019) -> np.ndarray:
    """
    Convert Hounsfield units to linear attenuation coefficient.
    
    Parameters
    ----------
    hu : ndarray
        Hounsfield unit value(s)
    mu_water : float
        Linear attenuation coefficient of water in mm^-1
        
    Returns
    -------
    mu : ndarray
        Linear attenuation coefficient(s) in mm^-1
    """
    return mu_water * (1 + hu / 1000)
