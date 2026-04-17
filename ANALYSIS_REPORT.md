# Bayesian Hierarchical Modeling for Weather Station Analysis

**DATASCI 451 Final Project**  
University of Michigan, Winter 2026

---

## 1. Introduction

### Research Question
How do different Bayesian modeling strategies compare for temperature prediction across weather stations with varying data availability?

### Modeling Approaches
| Model | Description | Key Feature |
|-------|-------------|-------------|
| **Complete Pooling** | All stations share one mean | Ignores station differences |
| **No Pooling** | Each station modeled independently | No information sharing |
| **Hierarchical** | Station effects from population distribution | Partial pooling / shrinkage |

### Data Overview
- **Source**: NOAA Global Historical Climatology Network
- **Region**: Michigan, USA
- **Period**: January - April 2024
- **Stations**: 167 with valid data (465 total)
- **Observations**: 643 monthly records

---

## 2. Exploratory Data Analysis

### 2.1 Data Coverage

![Station Coverage](plots/02_station_coverage_distribution.png)

**Key Finding**: Stations have varying data availability - this natural sparsity is crucial for demonstrating hierarchical model advantages.

| Coverage | Stations | Percentage |
|----------|----------|------------|
| 4 months (complete) | 150 | 90% |
| 3 months | 6 | 4% |
| 1-2 months (sparse) | 10 | 6% |

### 2.2 Temperature Distributions

![Temperature Distribution](plots/03_temperature_distributions.png)

![Monthly Boxplot](plots/04_monthly_temperature_boxplot.png)

### 2.3 Geographic Distribution

![Michigan Stations](plots/05_michigan_stations_overall.png)

![Monthly Temperature Maps](plots/06_monthly_temperature_maps.png)

### 2.4 Station Temperature Patterns

![Station Trends](plots/07_monthly_trends_by_station.png)

![Temperature Heatmap](plots/08_temperature_heatmap.png)

---

## 3. Model Specification

### 3.1 Hierarchical Model

$$y_{ij} \sim N(\alpha_i + \beta_j, \sigma^2)$$

$$\alpha_i \sim N(\mu_\alpha, \tau^2)$$

| Parameter | Meaning | Prior |
|-----------|---------|-------|
| $\alpha_i$ | Station baseline temperature | $N(\mu_\alpha, \tau^2)$ |
| $\beta_j$ | Month effect (seasonality) | $N(0, 15^2)$ |
| $\mu_\alpha$ | Population mean | $N(25, 20^2)$ |
| $\tau$ | Between-station SD | HalfCauchy(10) |
| $\sigma$ | Observation noise | HalfCauchy(10) |

### 3.2 MCMC Configuration
- Sampler: NUTS (No-U-Turn Sampler)
- Chains: 2, Samples: 2000, Tune: 1000
- Parameterization: Non-centered (avoids funnel geometry)

---

## 4. Results

### 4.1 Population Parameters

```
μ_α = 27.65°F  (population mean baseline)
τ   = 4.42°F   (between-station SD)
σ   = 2.84°F   (observation noise)
```

### 4.2 Shrinkage Effect

![Shrinkage Effect](plots/12_shrinkage_effect.png)

All station estimates shrink toward the population mean. Extreme stations (coldest/warmest) show the strongest shrinkage.

### 4.3 Station Effects Comparison

![Forest Plot](plots/13_forest_plot.png)

### 4.4 Seasonal Effects

![Month Effects](plots/14_month_effects.png)

| Month | Effect | Interpretation |
|-------|--------|----------------|
| January | -11.2°F | Coldest |
| February | -4.2°F | Cold |
| March | +6.2°F | Warming |
| April | +9.7°F | Warmest |

**Seasonal swing**: 20.9°F

### 4.5 Geographic Posterior

![Posterior Map](plots/16_michigan_posterior_map.png)

---

## 5. Hierarchical Model Advantage

### 5.1 Key Insight: Data Sparsity Matters

**The advantage of hierarchical models depends on varying data availability across groups.**

With our full dataset (167 stations, 1-4 observations each), we can properly evaluate when hierarchical models excel.

### 5.2 Prediction Error by Data Availability

![Full Dataset Analysis](plots/25_full_dataset_hierarchical_analysis.png)

| Observations | Stations | Hierarchical | No Pooling | Improvement |
|--------------|----------|--------------|------------|-------------|
| **1 month** | 3 | **1.78°F** | 3.76°F | **+52%** |
| **2 months** | 7 | **2.06°F** | 3.39°F | **+39%** |
| 3 months | 6 | 3.63°F | 5.12°F | +29% |
| 4 months | 151 | 1.83°F | 3.25°F | +44% |

**Key Finding**: Hierarchical models show the largest advantage for sparse-data stations.

### 5.3 Real Sparse Station Predictions

Using actual stations with limited data:

![Sparse Station Prediction](plots/24_real_sparse_station_prediction.png)

| Station | Data | Hierarchical α | No Pooling α | Shrinkage |
|---------|------|----------------|--------------|-----------|
| GRAYLING | Jan only | 24.7°F | 24.2°F | +0.5°F |
| BENTON HARBOR | Jan only | 34.2°F | 36.2°F | -2.0°F |
| IRON MTN KINGSFORD | Jan-Feb | 25.4°F | 25.2°F | +0.2°F |

Sparse stations are pulled toward the reliable population mean (27.7°F).

### 5.4 New Station Prediction (LOSO Test)

**Scenario**: Predict temperature for a completely new station with no historical data.

| Model | Capability | Error |
|-------|------------|-------|
| **Hierarchical** | ✅ Can predict using $N(\mu_\alpha, \tau^2)$ | 6.4°F |
| No Pooling | ❌ Impossible (no data) | - |

This is the fundamental advantage of hierarchical models.

---

## 6. Practical Applications

### 6.1 Frost Probability Estimation

![Freezing Probability](plots/18_freezing_probability.png)

Using posterior distributions to compute $P(T < 32°F)$:

| Station | January | February | March | April |
|---------|---------|----------|-------|-------|
| Bergland Dam (UP) | 100% | 99% | 78% | 45% |
| Traverse City | 98% | 90% | 55% | 28% |
| Ann Arbor | 95% | 82% | 35% | 12% |

### 6.2 Agricultural Decision Support

![Planting Decision Map](plots/19_planting_decision_map.png)

**Decision Rule**:
- P(frost) < 20%: ✅ Safe to plant
- 20% ≤ P(frost) < 50%: ⚠️ Caution
- P(frost) ≥ 50%: ❌ Do not plant

### 6.3 Road Maintenance Budget

![Icy Days Budget](plots/20_icy_days_budget.png)

---

## 7. Conclusions

### 7.1 Main Findings

1. **Hierarchical models excel with sparse data**
   - 52% error reduction for 1-observation stations
   - 39% error reduction for 2-observation stations

2. **Shrinkage is a double-edged sword**
   - Helps moderate groups (near population mean)
   - May hurt extreme groups (pulled toward wrong mean)

3. **Unique capability: New group prediction**
   - Hierarchical: Uses population distribution
   - No Pooling: Cannot predict without data

### 7.2 When to Use Hierarchical Models

| Scenario | Recommendation |
|----------|----------------|
| Varying data per group | ✅ Hierarchical |
| Predicting new groups | ✅ Hierarchical |
| Need uncertainty quantification | ✅ Hierarchical |
| All groups have sufficient data | ≈ Similar to No Pooling |
| Groups are extreme outliers | ⚠️ Shrinkage may hurt |

### 7.3 Key Takeaway

> **Partial pooling allows data-poor groups to "borrow strength" from data-rich groups through the population distribution. This is the core value of Bayesian hierarchical modeling.**

---

## Appendix: Figures

| # | Figure | Description |
|---|--------|-------------|
| 1 | `02_station_coverage_distribution.png` | Data coverage distribution |
| 2 | `03_temperature_distributions.png` | Temperature histograms |
| 3 | `04_monthly_temperature_boxplot.png` | Monthly boxplots |
| 4 | `05_michigan_stations_overall.png` | Station map |
| 5 | `06_monthly_temperature_maps.png` | Monthly geo-maps |
| 6 | `07_monthly_trends_by_station.png` | Station trends |
| 7 | `08_temperature_heatmap.png` | Temperature heatmap |
| 8 | `12_shrinkage_effect.png` | Shrinkage visualization |
| 9 | `13_forest_plot.png` | Forest plot |
| 10 | `14_month_effects.png` | Seasonal effects |
| 11 | `16_michigan_posterior_map.png` | Posterior map |
| 12 | `18_freezing_probability.png` | Frost probability |
| 13 | `19_planting_decision_map.png` | Planting decisions |
| 14 | `20_icy_days_budget.png` | Icy days prediction |
| 15 | `24_real_sparse_station_prediction.png` | Sparse station test |
| 16 | `25_full_dataset_hierarchical_analysis.png` | Full dataset analysis |

---

**Repository**: [github.com/guihunwansui/bayesian-hierarchical-weather-analysis](https://github.com/guihunwansui/bayesian-hierarchical-weather-analysis)
