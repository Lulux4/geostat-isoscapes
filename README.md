# SPIKED : SPeleothem Isoscapes from Kriging with an External Drift 

This repository contains a master's thesis project for reconstructing $\delta^{18}\text{O}_p$ isoscapes using geostatistics applied to speleothem (cave deposit) data. This project integrates paleoclimate simulation data (iTrace simulations) with stable isotope records from the SISAL database to create maps of oxygen isotope composition.

The kriged isoscapes generated are available in file  
```
output/kriging/spiked_11_to_20kaBP.nc
```

## Overview of the data and methods used in this project

- **Speleothem data**: Oxygen isotope records from stalagmite samples in caves (SISAL)
- **Temperature datasets**: iTrace climate model simulations (12-20 ka BP)

- **Geostatistical methods**: Variography analysis and kriging interpolation for spatial prediction, more specifically kriging with an external drift. 

- **Trend modeling**: Multiple linear regression to account for geographical trends in the isoscapes (latitude, elevation, distance to the coast)

## Key Features

- **Variography & Kriging**: Geostatistical analysis of spatial structure and kriging-based interpolation
- **Data Integration**: Combines speleothem isotope records with paleoclimate simulations
- **Trend Analysis**: Multiple linear regression for detrending data based on geographic variables
- **Visualization**: Interactive maps and static plots using matplotlib, plotly, and cartopy
- **Quality Control**: Cross-validation and statistical testing (Kleijnen test) for model assessment

## Project Structure

```
geostat-isoscapes/
├── scripts/                     # Executables scripts
│   ├── kriging.py               # Main kriging interpolation script
│   └── variogram_parameters_scan.py  # Iterative variogram computations
├── data/                         # Data files
│   ├── iTrace_simulations_dict.json  # iTrace simulation metadata
│   ├── elevation/               # Elevation/topography data (ETOPO) (file to download)
│   ├── modern/                  # Modern climate data 
│   ├── temperature/             # Temperature dataset to download (Krapp)
│   ├── shapefiles/              # Geographic boundaries and coastlines (files to download)
│   └── sisalv3_csv/             # SISAL database CSV files
├── notebooks/                   # Jupyter notebooks for exploration & analysis
│   ├── sisal_data_exploration.ipynb
│   ├── variography.ipynb
│   ├── kriging.ipynb
│   ├── lgmr_validation.ipynb
│   ├── fit_trend_model.ipynb
│   ├── correlation_analysis.ipynb
│   └── calcite_to_dripwater_conversion.ipynb
├── src/geostat_isoscapes_tools/ # Main package
│   ├── geostat_utils.py        # Geostatistical functions (kriging, variography, detrending)
│   ├── sisal_utils.py          # SISAL database processing & filtering
│   ├── plot_utils.py           # Visualization functions
│   ├── variogram_models.py     # Custom variogram models
│   └── utils.py                # General utilities
├── output/                      # Generated outputs
│   ├── kriging/                # Kriging interpolation results
│   ├── results_figs/           # Figures and visualizations
|   ├── interdistances/         # Graph of SISAL sites separated by between 4000 and 6000km.
└── pyproject.toml              # Project configuration
```

## Installation

### Prerequisites
- Python >= 3.9
- Conda environment

### Setup Steps

1. **Clone the repository**:
   ```bash
   cd /path/to/geostat-isoscapes
   ```

2. **Create and activate a conda environment**:
   ```bash
   conda create -n geostat-env python=3.9
   conda activate geostat-env
   ```

3. **Install the package and dependencies**:
   ```bash
   pip install -e .
   pip install -r requirements.txt
   plotly_get_chrome
   ```

### Dependencies

Key packages:
- **Geostatistics**: `gstools`, `skgstat`, `pykrige` (kriging interpolation)
- **Data Processing**: `pandas`, `numpy`, `xarray`, `netCDF4`
- **Geospatial**: `cartopy`, `geopandas`, `shapely`, `pyproj`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Scientific**: `scipy`, `scikit-learn`, `statsmodels`, `properscoring`

## Usage

### 1. Data Exploration

Start with the notebooks to understand the datasets:

```bash
jupyter notebook notebooks/sisal_data_exploration.ipynb
jupyter notebook notebooks/secondary_variable_data_exploration.ipynb
```

### 2. Variography Analysis

The analysis of the spatial structure of oxygen isoscapes can be explored for all time slices pver 20-11kaBP using the parameter scan :

```bash
python bin/variogram_parameters_scan.py
```

Review results in:
```bash
jupyter notebook notebooks/variography.ipynb
```

### 3. Kriging Interpolation

Run the main kriging workflow to generate isoscapes:

```bash
python scripts/kriging.py
```

Configure the script by editing parameters in `kriging.py`:
- `exp_name`: Name of the experiment run
- `kyr`: Time slice(s) to process (e.g., 12 ka BP)
- `res_months_list`: Temporal resolution
- `caves_to_exclude`: Caves to exclude for sensitivity analysis
- `temperature_ds_name`: Temperature dataset ('itrace' or 'krapp'),
- etc...

Results are saved in `output/kriging/{exp_name}/`.

### 4. Additional Analysis

```bash
jupyter notebook notebooks/correlation_analysis.ipynb
jupyter notebook notebooks/fit_trend_model.ipynb
jupyter notebook notebooks/calcite_to_dripwater_conversion.ipynb
```

## Common Workflow

### Full Pipeline (Variography → Kriging)

```bash
# 1. Scan variogram parameters
python scripts/variogram_parameters_scan.py

# 2. Run kriging with best-fit variogram
python scripts/kriging.py

# 3. Evaluate results
jupyter notebook notebooks/kriging.ipynb
```

### Sensitivity Analysis

Edit `caves_to_exclude` in `kriging.py` to test robustness:
```python
caves_to_exclude = ['Devils Hole', 'Cave of Interest']
```

### Custom Temperature Dataset

Modify `temperature_ds_name` and `temperature_ds_path` to use different climate data:
```python
temperature_ds_name = 'krapp'
temperature_ds_path = '/path/to/krapp/data/'
```

## Requirements

See `requirements.txt` for full dependency list.

## Author

**Léa Gainon** (leagainon@proton.me, lea.gainon@unil.ch)

Master's Thesis Project, 2025-2026

## Notes

- Data loading can be memory-intensive for full iTrace simulations
- Kriging computation time depends on sample size and grid resolution
- Use `verbose=True` in functions for detailed processing logs


