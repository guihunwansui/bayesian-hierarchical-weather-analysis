# Bayesian Hierarchical Modeling for Weather Station Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyMC](https://img.shields.io/badge/PyMC-5.0+-orange.svg)](https://www.pymc.io/)

**DATASCI 451 Final Project** | University of Michigan, Winter 2026

## Key Findings

### Hierarchical Advantage by Data Availability

| Observations | Hierarchical | No Pooling | Improvement |
|--------------|--------------|------------|-------------|
| **1 month** | 1.78°F | 3.76°F | **+52%** |
| **2 months** | 2.06°F | 3.39°F | **+39%** |
| 4 months | 1.83°F | 3.25°F | +44% |

**Core insight**: Hierarchical models allow data-poor stations to "borrow strength" from data-rich stations.

![Analysis](plots/25_full_dataset_hierarchical_analysis.png)

## Model

```
y_ij ~ Normal(α_i + β_j, σ²)
α_i  ~ Normal(μ_α, τ²)      # Station effects from population

Parameters:
  μ_α = 27.65°F (population mean)
  τ   = 4.42°F  (between-station SD)
  σ   = 2.84°F  (observation noise)
```

## Data

- **Source**: NOAA Global Historical Climatology Network
- **Region**: Michigan, USA (167 stations)
- **Period**: January - April 2024
- **Observations**: 643 monthly records

## Project Structure

```
├── data/                    # Weather data and MCMC traces
├── plots/                   # 16 visualizations
├── *.ipynb                  # Jupyter notebooks (EDA, modeling, applications)
├── *.py                     # Analysis scripts
└── ANALYSIS_REPORT.md       # Full report with all figures
```

## Quick Start

```bash
pip install pymc arviz pandas matplotlib seaborn

# Run models
python run_models_v2.py

# Analyze results  
python analyze_results_v2.py
```

## Report

See [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) for the complete analysis with all figures.

## Technologies

- **Bayesian Inference**: PyMC 5, ArviZ
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
