# AI4EO Multi-Task Project

## Problem Description

Earth observation (EO) using satellite imagery has become a crucial tool for monitoring and understanding various environmental processes. However, the vast amount of data produced by satellite missions such as Sentinel-2 presents significant challenges in terms of data processing, analysis, and interpretation. This project aims to leverage Artificial Intelligence (AI) techniques to automate three key tasks in EO:

1. Land Cover Classification: Identifying different land cover types (water, forest, agricultural land, urban areas) using multi-spectral satellite imagery. This task helps in monitoring land use changes, detecting deforestation, and understanding urban expansion.

2. Vegetation Index Prediction: Accurately predicting vegetation indices (NDVI, EVI) using regression models. These indices are essential for assessing vegetation health, monitoring crop conditions, and detecting environmental stress.

3. Image Alignment and Change Detection: Aligning multi-temporal images and detecting changes over time using feature matching and change detection techniques. This is useful for monitoring natural disasters, assessing water body changes, and detecting land degradation.

## Project Overview
This project aims to demonstrate the application of AI in Earth Observation (AI4EO) using Sentinel-2 multi-spectral imagery. The project covers three main tasks:

1. Land Cover Classification (Image Classification)
2. Vegetation Index Prediction (Regression)
3. Image Alignment and Change Detection

## Project Structure
```
AI4EO_MultiTask_Project/
├── data/               # Raw and preprocessed data
├── src/                # Source code
│   ├── part1_classification.py   # Image Classification
│   ├── part2_regression.py       # Regression Analysis
│   ├── part3_image_alignment.py  # Image Alignment and Change Detection
│   └── part4_explainability.py   # Explainable AI
├── notebooks/          # Jupyter Notebook (Exploratory Analysis)
├── results/            # Model Results and Evaluation
└── README.md           # Project Documentation
```

## Data Preprocessing
The data preprocessing process is handled in the `data_preprocessing.py` script. It includes:

1. Loading Sentinel-2 multi-spectral imagery.
2. Normalizing image pixel values.
3. Applying cloud masking based on band threshold.
4. Calculating multiple vegetation indices (NDVI, EVI, NDWI).
5. Automatically saving processed images for further analysis.

### Additional Improvements:
- Added support for calculating EVI (Enhanced Vegetation Index) and NDWI (Normalized Difference Water Index).
- Added automatic saving of NDVI, EVI, and NDWI images.

### How to Use:
- Ensure the Sentinel-2 image is placed in the `data/` folder.
- Adjust the image path in the `data_preprocessing.py` script as needed.
- Run the script using:

```bash
python src/data_preprocessing.py
```
- The processed images (NDVI, EVI, NDWI) will be saved in the same directory with corresponding names.
