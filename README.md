# Flood Remote Sensing with AI4EO

## Problem Description
Flood events cause widespread damage and pose significant risks to communities. Traditional flood mapping depends on manual interpretation and scattered in-situ measurements, which can be slow and lack spatial coverage. Sentinel‑2 L2A multi‑spectral imagery (10 m resolution, 5‑day revisit) provides timely, high-resolution observations of inundation extents. This project automates five key tasks using AI techniques:

1. **Flood Extent Classification:** Semantic segmentation of inundated vs. non‑inundated areas.
2. **Water Depth Estimation:** Regression models to predict water depth from spectral indices and available ground observations.
3. **Image Alignment & Change Detection:** Feature‑based and optical‑flow methods to align multi‑temporal images and detect changes.
4. **Spatial Interpolation:** Gaussian Process Regression to interpolate sparse water depth observations into continuous water depth maps.
5. **Explainable AI:** SHAP and Grad‑CAM to interpret model decisions and identify key spectral drivers.

## Project Structure
```
AI4EO_Flood_Project/
├── data/                   # Raw Sentinel‑2 bands and preprocessed indices
├── src/                    # Source code scripts
│   ├── data_preprocessing.py   # Load bands, normalize, cloud mask, calculate NDVI, NDWI, etc.
│   ├── part1_classification.py # Flood extent segmentation
│   ├── part2_regression.py     # Water depth estimation
│   ├── part3_alignment.py      # Image alignment & change detection
│   └── part4_interpolation.py  # Gaussian Process interpolation
│   └── part5_explainability.py # Model interpretability
├── notebooks/              # Exploratory analysis and visualization notebooks
├── results/                # Output maps, models, evaluation plots
└── README.md               # Project documentation (this file)
```

## Data Preprocessing
The `data_preprocessing.py` script in `src/` performs:
- **Loading:** Read Sentinel‑2 bands (B02, B03, B04, B08) from `.tif`.
- **Normalization:** Scale reflectance values.
- **Cloud Masking:** Threshold‑based cloud removal.
- **Index Calculation:** Compute NDVI, NDWI, and other indices.
- **Output:** Save processed single‑band GeoTIFFs to `data/processed/`.

### Usage
```bash
# Preprocess raw Sentinel‑2 bands
python src/data_preprocessing.py --input-dir data/ --output-dir data/processed/
```

## Next Steps
1. **Run Flood Extent Classification:**
   ```bash
   python src/part1_classification.py --data-dir data/processed/ --output-dir results/
   ```
2. **Estimate Water Depth (Regression):**
   ```bash
   python src/part2_regression.py --data-dir data/processed/ --output-dir results/
   ```
3. **Align Images & Detect Changes:**
   ```bash
   python src/part3_alignment.py --data-dir data/processed/ --output-dir results/
   ```
4. **Interpolate Water Depth:**
   ```bash
   python src/part4_interpolation.py --data-dir results/observations.csv --output-dir results/
   ```
5. **Explain Models:**
   ```bash
   python src/part5_explainability.py --model-dir results/ --data-dir data/processed/
   ```

Refer to individual scripts’ docstrings for detailed options and parameters.
