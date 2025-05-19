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

# Function to apply cloud mask based on band threshold
def apply_cloud_mask(image, band_index, threshold=0.2):
    cloud_mask = image[band_index] > threshold
    image[:, cloud_mask] = 0  # Mask cloudy pixels
    return image

# Function to calculate NDVI
def calculate_ndvi(image, nir_band=7, red_band=3):
    nir = image[nir_band].astype(float)
    red = image[red_band].astype(float)
    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi

# Function to save processed image
def save_processed_image(image, profile, output_path):
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(image)

# Main script
if __name__ == "__main__":
    image_path = "data/sample_image.tif"
    output_path = "data/processed_image.tif"

    # Load and preprocess image
    image, profile = load_satellite_image(image_path)
    image = normalize_image(image)

    # Apply cloud mask (using band 2 for cloud detection)
    image = apply_cloud_mask(image, band_index=2, threshold=0.2)

    # Calculate NDVI
    ndvi = calculate_ndvi(image)

    # Save processed image
    profile.update(dtype=rasterio.float32, count=1)
    save_processed_image(ndvi, profile, output_path)

    print("Preprocessing complete. Processed image saved at:", output_path)
