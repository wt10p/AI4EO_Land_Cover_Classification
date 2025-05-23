# Arctic Sea‐Ice Altimetry Interpolation with GPSat

## Background

To capture sea‐ice and ocean surface features, satellite radar altimetry employs a SAR‐mode radar that emits microwave pulses and measures return time and waveform properties (e.g., pulse peakiness) as the satellite moves along its orbit. Delay‐Doppler processing enhances along‐track resolution by combining multiple Doppler‐shifted echoes, enabling retrieval of sea surface height anomaly (SLA) and ice freeboard over leads and floes. However, these along‐track measurements remain spatially sparse and non‐uniform, motivating the use of data‐driven interpolation methods.

On the algorithmic side, we compare two interpolation paradigms: classical cubic‐spline interpolation, which fits smooth polynomials along the track coordinate, and Gaussian‐process regression (GPR), a Bayesian nonparametric approach that models SLA and freeboard as random functions with covariance kernels. The GPR implementation (via the GPSat toolkit) leverages localized “expert” subsets along the track to reduce computational cost while providing both predictions and uncertainty estimates. By combining these techniques, we can reconstruct continuous profiles of SLA and freeboard, filling measurement gaps and characterizing interpolation uncertainty.

Sea‐ice radar altimetry provides along‐track measurements of sea surface height anomaly (SLA) and ice freeboard at discrete points along satellite orbits. These observations are irregularly spaced and subject to gaps where the satellite does not traverse or where data are flagged as low quality (e.g., leads vs. floes). Reliable interpolation is therefore essential to reconstruct continuous SLA and freeboard profiles, quantify uncertainties, and enable detailed mapping of sea‐ice thickness and local ocean dynamics over polar regions.

Sea‐ice radar altimetry provides along‐track measurements of sea surface height anomaly (SLA) and ice freeboard at discrete points along satellite orbits. These observations are irregularly spaced and subject to gaps where the satellite does not traverse or where data are flagged as low quality (e.g., leads vs. floes). Reliable interpolation is therefore essential to reconstruct continuous SLA and freeboard profiles, quantify uncertainties, and enable detailed mapping of sea‐ice thickness and local ocean dynamics over polar regions.

Over a \~48‐hour observational window in the Arctic region (2019‑01‑10 00:22 UTC to 2019‑01‑11 19:17 UTC), we employ two distinct interpolation strategies:

## Interpolation Methods

### 1. Sparse Gaussian‐Process Regression (GPSat Toolkit)

**Notebook:** `Chapter_2_SLA_GPSat_GPOD.ipynb`

* **Input:** GPOD‐processed Sentinel‑3 SAR Level‑1A `.proc` files parsed into a consolidated CSV (`df_GPOD.csv`) containing longitude, latitude, SLA, error estimates, freeboard, and pulse peakiness.
* **Approach:** We build a Sparse Gaussian‐Process Regression (SGPR) model using the GPSat package. The workflow defines a set of local “expert” subsets along the satellite tracks, fits a Gaussian process to each subset, and then stitches the predictions onto a regular grid.
* **Outputs:** Gridded SLA fields, uncertainty maps, and scatterplots comparing observed vs. predicted SLA.

### 2. Classical Cubic‐Spline Interpolation & Along‐Track Comparison

**Notebook:** `Chapter_2_GPSat_along_track.ipynb`

* **Input:** The same GPOD `.proc` files and their parsed DataFrame of along‐track measurements.
* **Approaches:**

  1. **Cubic‐Spline Interpolation:** A one‐dimensional cubic spline is fit to each satellite track independently, interpolating SLA (and freeboard) along the track coordinate.
  2. **Gaussian‐Process Interpolation:** Using GPSat, a one‐dimensional Gaussian‐process regressor is trained on the same track data to produce continuous predictions with uncertainty estimates.
* **Comparison:** Side‐by‐side profile plots and error metrics (RMSE, bias) quantify performance differences between spline and GP methods along each track.

## Repository Structure

```text
.
├── data/
│   ├── *.proc                        # GPOD-provided Sentinel-3 .proc files
│   └── df_GPOD.csv                   # Parsed CSV for SLA notebook
├── notebooks/
│   ├── Chapter_2_SLA_GPSat_GPOD.ipynb
│   └── Chapter_2_GPSat_along_track.ipynb
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment file
└── README.md                         # This document
```

## Installation

1. **Clone repository**:

   ```bash
   git clone https://github.com/yourusername/Arctic-Altimetry-Interpolation.git
   cd Arctic-Altimetry-Interpolation
   ```

2. **Set up environment**:

   ```bash
   conda env create -f environment.yml   # or create venv + pip install
   pip install -r requirements.txt
   pip install -e .
   ```

## Data Acquisition

* **GPOD .proc files**: Place the provided Sentinel‑3 SAR `.proc` files into `data/`. These cover 2019‑01‑10 00:22 UTC to 2019‑01‑11 19:17 UTC.
* **Parsing:** Run the first cells of either notebook to convert `.proc` files into `data/df_GPOD.csv`.

## How to Run

1. **Notebook 1** (`Chapter_2_SLA_GPSat_GPOD.ipynb`):

   * Load `df_GPOD.csv`, configure GPSat parameters, fit SGPR models, and generate gridded SLA and uncertainty outputs.

2. **Notebook 2** (`Chapter_2_GPSat_along_track.ipynb`):

   * Load `df_GPOD.csv`, parse individual tracks, apply cubic‐spline and GP interpolations, and compare along‐track profiles with error metrics.

Each notebook contains detailed comments, inline plots, and example parameter settings to reproduce the full analysis in Colab or Jupyter.

## License

This project is licensed under the **MIT License**.

## Contact

For questions or feedback, please open an issue or contact **Your Name** at [your.email@example.com](mailto:your.email@example.com).
