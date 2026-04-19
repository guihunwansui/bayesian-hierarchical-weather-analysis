# Bayesian Hierarchical Models: Statistical Analysis

**DATASCI 451 Final Project**  
University of Michigan, Winter 2026

---

## 1. Data Overview

### 1.1 Data Source
- **Source**: NOAA Global Historical Climatology Network
- **Region**: Michigan, USA
- **Period**: January - April 2024

### 1.2 Dataset Summary

| Metric | Monthly Data | Daily Data |
|--------|--------------|------------|
| Total Observations | 643 | 14,569 |
| Stations | 168 | 167 |
| Obs per Station | 1-4 months | 29-190 days |

### 1.3 Data Availability Distribution

| Months | Stations | Percentage |
|--------|----------|------------|
| 4 (complete) | 152 | 90% |
| 3 | 6 | 4% |
| 1-2 (sparse) | 10 | 6% |

---

## 2. Model Specification

### 2.1 Statistical Model

$$y_{ij} \sim N(\alpha_i + \beta_j, \sigma^2)$$

where:
- $\alpha_i$ = station baseline temperature
- $\beta_j$ = month effect (seasonality)
- $\sigma$ = observation noise

### 2.2 Three Modeling Approaches

| Model | Station Effect | Information Sharing |
|-------|----------------|---------------------|
| **Complete Pooling** | $\alpha_i = \mu$ | All stations identical |
| **No Pooling** | $\alpha_i$ independent | No sharing |
| **Hierarchical** | $\alpha_i \sim N(\mu_\alpha, \tau^2)$ | Partial pooling |

### 2.3 Hierarchical Priors

| Parameter | Prior | Interpretation |
|-----------|-------|----------------|
| $\mu_\alpha$ | $N(25, 20^2)$ | Population mean |
| $\tau$ | HalfCauchy(10) | Between-station SD |
| $\beta_j$ | $N(0, 15^2)$ | Month effects |
| $\sigma$ | HalfCauchy(10) | Observation noise |

---

## 3. Results

### 3.1 Population Parameters

| Parameter | Estimate | Interpretation |
|-----------|----------|----------------|
| $\mu_\alpha$ | 28.53°F | Population mean baseline |
| $\tau$ | 4.39°F | Between-station SD |
| $\sigma$ | 2.84°F | Observation noise |
| $\tau/\sigma$ | **1.55** | Signal-to-noise ratio |

### 3.2 Month Effects

![Month Effects](plots/S03_month_effects.png)

| Month | Effect | Temperature |
|-------|--------|-------------|
| January | -11.6°F | ~17°F |
| February | -5.4°F | ~23°F |
| March | +5.4°F | ~34°F |
| April | +9.3°F | ~38°F |

**Seasonal swing**: ~21°F from January to April

---

## 4. Shrinkage Analysis

### 4.1 What is Shrinkage?

Hierarchical models "shrink" extreme estimates toward the population mean. This regularization:
- Reduces overfitting for data-poor groups
- Trades small bias for large variance reduction

### 4.2 Shrinkage Verification

![Shrinkage Effect](plots/S01_shrinkage_effect.png)

**Key findings:**
- Stations **above** mean: 90/91 (99%) correctly shrink DOWN
- Stations **below** mean: 77/77 (100%) correctly shrink UP
- **Overall: 167/168 (99%) shrink toward population mean**

### 4.3 Sparse Station Analysis

![Sparse Stations](plots/S02_sparse_stations.png)

Stations with only 1-2 observations show clear shrinkage toward the population mean (28°F). The arrows indicate the direction of shrinkage from No Pooling (blue) to Hierarchical (orange) estimates.

---

## 5. Daily vs Monthly Comparison

### 5.1 The Key Question

Why use monthly aggregates instead of daily observations? Daily data provides 23x more observations.

### 5.2 tau/sigma Ratio Comparison

| Data | tau | sigma | tau/sigma |
|------|-----|-------|-----------|
| Monthly | 4.39°F | 2.84°F | **1.55** |
| Daily | 3.87°F | 10.75°F | **0.36** |

![Monthly vs Daily](plots/S04_monthly_vs_daily.png)

### 5.3 Why Daily Noise is High

Even within the same station and same month, temperature varies ~10°F day-to-day due to weather systems. This is irreducible weather variability.

| Source | SD |
|--------|-----|
| Day-to-day weather | 10.0°F |
| Seasonal variation | 9.3°F |
| Station differences | 4.5°F |

### 5.4 The Aggregation Insight

**Monthly averaging reduces noise ($\sigma$) without reducing station signal ($\tau$).**

- Daily: Weather noise dominates → low tau/sigma → weak hierarchical advantage
- Monthly: Noise averages out → high tau/sigma → strong hierarchical advantage

> **Key Insight**: More data ≠ more hierarchical advantage. Better signal-to-noise ratio = more hierarchical advantage.

---

## 6. Summary

### 6.1 Main Findings

| Finding | Evidence |
|---------|----------|
| Shrinkage works correctly | 99% of stations shrink toward mean |
| tau/sigma determines advantage | Monthly (1.55) >> Daily (0.36) |
| Aggregation improves SNR | sigma drops from 10.75 to 2.84°F |

### 6.2 When to Use Hierarchical Models

| Scenario | Recommendation |
|----------|----------------|
| Varying data per group | Hierarchical |
| Need new group prediction | Hierarchical |
| High within-group noise | Consider aggregation first |
| Groups are extreme outliers | Shrinkage may hurt |

### 6.3 Key Takeaways

1. **Partial pooling** allows data-poor groups to "borrow strength" from the population distribution.

2. **The tau/sigma ratio** determines hierarchical model advantage.

3. **Data aggregation** can improve hierarchical performance by reducing noise while preserving the group-level signal.

---

## Appendix: Figures

| Figure | Description |
|--------|-------------|
| `plots/S01_shrinkage_effect.png` | Shrinkage visualization |
| `plots/S02_sparse_stations.png` | Sparse station forest plot |
| `plots/S03_month_effects.png` | Seasonal effects |
| `plots/S04_monthly_vs_daily.png` | Data granularity comparison |

---

**Repository**: [github.com/guihunwansui/bayesian-hierarchical-weather-analysis](https://github.com/guihunwansui/bayesian-hierarchical-weather-analysis)
