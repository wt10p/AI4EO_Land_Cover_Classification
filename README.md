# AI4EO_Land_Cover_Classification
# AI4EO Data Preprocessing Script

import rasterio
import numpy as np
import os

# Function to load Sentinel-2 image
def load_satellite_image(file_path):
    with rasterio.open(file_path) as src:
        image = src.read()  # Read multi-band image
        profile = src.profile
    return image, profile

# Function to normalize image data
def normalize_image(image):
    return image / 10000.0  # Normalize pixel values

# Function to apply cloud mask
def apply_clo...
}
