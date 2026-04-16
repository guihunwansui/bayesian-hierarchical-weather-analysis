# DATASCI 451 Final Project - Implementation Plan

## Bayesian Hierarchical Modeling of NOAA Weather Data

---

## 1. Research Framework

### 1.1 Core Research Question

> **How do monthly temperature patterns vary across Michigan weather stations, and can hierarchical Bayesian models improve station-level estimation compared to analyzing each station independently?**

### 1.2 Sub-questions

1. **Seasonal Pattern**: What is the magnitude of monthly temperature variation across stations?
2. **Station Heterogeneity**: How much do baseline temperatures differ across locations?
3. **Shrinkage Effect**: For stations with limited data, does partial pooling improve estimates?
4. **Model Comparison**: Which pooling strategy (complete/no/partial) best balances bias and variance?

---

## 2. Statistical Models

### 2.1 Data Structure

Let:
- $y_{ij}$ = average temperature for station $i$ in month $j$
- $i = 1, \ldots, 8$ (stations)
- $j = 1, \ldots, 4$ (months: Jan, Feb, Mar, Apr)
- $m(j) \in \{1, 2, 3, 4\}$ = month indicator

### 2.2 Model 1: Complete Pooling (Baseline)

**Assumption**: All stations share the same baseline temperature.

$$y_{ij} \sim N(\mu + \beta_{m(j)}, \sigma^2)$$

**Priors**:
- $\mu \sim N(25, 20^2)$ — overall mean (centered on ~25°F)
- $\beta_m \sim N(0, 15^2)$ — month effects (sum-to-zero constraint: $\beta_1 = -\sum_{m=2}^4 \beta_m$)
- $\sigma \sim \text{Half-Cauchy}(0, 10)$ — observation noise

**Interpretation**: Ignores station differences; all variation attributed to month + noise.

### 2.3 Model 2: No Pooling (Separate)

**Assumption**: Each station has completely independent parameters.

$$y_{ij} \sim N(\alpha_i + \beta_{m(j)}, \sigma^2)$$

**Priors**:
- $\alpha_i \sim N(25, 20^2)$ — independent station intercepts
- $\beta_m \sim N(0, 15^2)$ — shared month effects
- $\sigma \sim \text{Half-Cauchy}(0, 10)$

**Interpretation**: No information sharing; high variance for stations with few observations.

### 2.4 Model 3: Partial Pooling (Hierarchical) — **Main Model**

**Assumption**: Station effects come from a common distribution.

$$y_{ij} \sim N(\alpha_i + \beta_{m(j)}, \sigma^2)$$

**Hierarchical Structure**:
$$\alpha_i \sim N(\mu_\alpha, \tau^2)$$

**Hyperpriors**:
- $\mu_\alpha \sim N(25, 20^2)$ — population mean temperature
- $\tau \sim \text{Half-Cauchy}(0, 10)$ — between-station standard deviation
- $\beta_m \sim N(0, 15^2)$ — month effects
- $\sigma \sim \text{Half-Cauchy}(0, 10)$ — within-station noise

**Interpretation**: Stations "borrow strength" from each other; estimates shrink toward the group mean.

---

## 3. Expected Results

### 3.1 Parameter Estimates

| Parameter | Expected Value | Interpretation |
|-----------|----------------|----------------|
| $\mu_\alpha$ | ~24-26°F | Overall mean temperature across all stations |
| $\tau$ | ~4-6°F | Between-station variability (north-south gradient) |
| $\sigma$ | ~3-5°F | Day-to-day variability within station-month |
| $\beta_1$ (Jan) | ~ -10 to -15°F | January is coldest (reference: overall mean) |
| $\beta_2$ (Feb) | ~ -5 to -8°F | February slightly warmer than January |
| $\beta_3$ (Mar) | ~ +5 to +8°F | March warming trend |
| $\beta_4$ (Apr) | ~ +10 to +15°F | April is warmest |

### 3.2 Station-Specific Effects ($\alpha_i$)

Based on EDA, expected ranking (coldest to warmest):

| Station | Expected $\alpha_i$ | Why |
|---------|---------------------|-----|
| Bergland Dam | ~17°F | Northernmost, Upper Peninsula west |
| Gwinn Sawyer | ~21°F | Central Upper Peninsula |
| Iron Mountain | ~22°F | Southwest Upper Peninsula |
| Atlanta MI | ~22°F | Northern Lower Peninsula |
| Traverse City | ~24°F | Northwest Lower Peninsula (lake effect) |
| Bad Axe | ~26°F | Thumb region |
| Pontiac | ~29°F | Southeast Michigan |
| Ann Arbor | ~31°F | Southernmost |

### 3.3 Shrinkage Effect Visualization

```
                    No Pooling          Partial Pooling
                    Estimate            Estimate
Station A (n=31):   ●─────────────────────●──────── (little shrinkage)
Station B (n=7):    ●───────────●──────────────────  (more shrinkage toward mean)
                              ↑
                         Group Mean
```

**Key Insight**: Stations with fewer observations (e.g., April data) will show more shrinkage toward the population mean.

---

## 4. Model Comparison Metrics

### 4.1 Quantitative Criteria

| Metric | Formula | Best Model Has |
|--------|---------|----------------|
| **WAIC** | Widely Applicable Information Criterion | Lower value |
| **LOO-CV** | Leave-One-Out Cross-Validation | Higher elpd |
| **Posterior Predictive Check** | $p(y_{rep} \| y)$ | Data falls within 95% CI |

### 4.2 Expected Comparison Results

| Model | WAIC | Pros | Cons |
|-------|------|------|------|
| Complete Pooling | Highest (worst) | Simple | Ignores station differences |
| No Pooling | Medium | Captures station effects | High variance, overfits |
| **Partial Pooling** | **Lowest (best)** | Balances bias-variance | More complex |

---

## 5. Key Visualizations for Report

### 5.1 Model Results Plots

1. **Shrinkage Plot**: No-pooling vs partial-pooling estimates
   - X-axis: No-pooling estimate
   - Y-axis: Partial-pooling estimate
   - Diagonal line = no shrinkage
   - Points closer to horizontal line = more shrinkage

2. **Forest Plot**: Station effects with 95% credible intervals
   - Compare CI width: partial pooling should have narrower CIs

3. **Posterior Predictive Check**: Observed vs predicted temperatures
   - Overlay actual data on posterior predictive distribution

4. **Month Effects Plot**: $\beta_m$ estimates with uncertainty

### 5.2 Interpretation Guide

| Observation | Interpretation |
|-------------|----------------|
| $\tau$ is large relative to $\sigma$ | Stations are heterogeneous; hierarchical model justified |
| $\tau \approx 0$ | Stations are similar; complete pooling may suffice |
| CIs overlap across stations | No significant station differences |
| Partial pooling WAIC << No pooling WAIC | Hierarchical model provides regularization benefit |

---

## 6. Implementation Code Structure

### 6.1 Notebook Organization

```
03_model_fitting.ipynb
├── Load cleaned data
├── Define Stan/PyMC models
│   ├── Complete Pooling
│   ├── No Pooling  
│   └── Partial Pooling (Hierarchical)
├── Run MCMC sampling
├── Convergence diagnostics (R-hat, ESS, trace plots)
└── Save posterior samples

04_model_comparison.ipynb
├── Load posterior samples
├── Compute WAIC / LOO-CV
├── Compare models
├── Posterior predictive checks
└── Generate comparison table

05_results_visualization.ipynb
├── Shrinkage plot
├── Forest plot (station effects)
├── Month effects plot
├── Geographic map with posterior means
└── Final summary figures for report
```

### 6.2 Stan Model Code (Hierarchical)

```stan
data {
  int<lower=0> N;           // total observations
  int<lower=0> J;           // number of stations
  int<lower=0> M;           // number of months (4)
  array[N] int<lower=1,upper=J> station;  // station index
  array[N] int<lower=1,upper=M> month;    // month index
  vector[N] y;              // observed temperature
}

parameters {
  real mu_alpha;            // population mean
  real<lower=0> tau;        // between-station SD
  vector[J] alpha_raw;      // station effects (non-centered)
  vector[M-1] beta_raw;     // month effects (M-1 for identifiability)
  real<lower=0> sigma;      // observation noise
}

transformed parameters {
  vector[J] alpha = mu_alpha + tau * alpha_raw;  // non-centered parameterization
  vector[M] beta;
  beta[2:M] = beta_raw;
  beta[1] = -sum(beta_raw);  // sum-to-zero constraint
}

model {
  // Hyperpriors
  mu_alpha ~ normal(25, 20);
  tau ~ cauchy(0, 10);
  
  // Priors
  alpha_raw ~ normal(0, 1);  // implies alpha ~ normal(mu_alpha, tau)
  beta_raw ~ normal(0, 15);
  sigma ~ cauchy(0, 10);
  
  // Likelihood
  for (n in 1:N) {
    y[n] ~ normal(alpha[station[n]] + beta[month[n]], sigma);
  }
}

generated quantities {
  vector[N] y_rep;  // posterior predictive
  vector[N] log_lik;  // for WAIC/LOO
  
  for (n in 1:N) {
    real mu_n = alpha[station[n]] + beta[month[n]];
    y_rep[n] = normal_rng(mu_n, sigma);
    log_lik[n] = normal_lpdf(y[n] | mu_n, sigma);
  }
}
```

---

## 7. Report Structure

### 7.1 Suggested Outline

1. **Introduction** (0.5 page)
   - Research question
   - Why hierarchical modeling?

2. **Data Description** (1 page)
   - Source, stations, date range
   - EDA summary (1-2 key figures)
   - Data quality issues

3. **Statistical Model** (1.5 pages)
   - Model specification (equations)
   - Prior choices and justification
   - Three pooling strategies comparison

4. **Results** (2 pages)
   - Convergence diagnostics
   - Parameter estimates table
   - Shrinkage effect visualization
   - Model comparison (WAIC table)

5. **Discussion** (1 page)
   - Key findings (seasonal pattern, north-south gradient)
   - Advantages of hierarchical model
   - Limitations

6. **Conclusion** (0.5 page)

### 7.2 Key Figures for Report

| Figure | Purpose |
|--------|---------|
| Michigan station map | Show geographic coverage |
| Monthly temperature heatmap | EDA: seasonal + station patterns |
| Shrinkage plot | Core result: partial pooling effect |
| Model comparison table | WAIC/LOO comparison |
| Posterior predictive check | Model validation |

---

## 8. Timeline (Remaining)

| Date | Task | Owner |
|------|------|-------|
| Apr 15 | EDA complete, implementation plan | Xintong |
| Apr 16-17 | Fit all three models in Stan/PyMC | Jingyang |
| Apr 18 | Model diagnostics & comparison | Jingyang |
| Apr 19 | Results visualization | Xintong |
| Apr 20 | Presentation | All |
| Apr 21-23 | Write final report | All |
| Apr 24 | Submit report + code | All |

---

## 9. Potential Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| MCMC convergence issues | Use non-centered parameterization; increase warmup |
| Small sample size (32 obs) | That's the point — hierarchical models help! |
| Choosing priors | Use weakly informative priors; sensitivity analysis |
| Interpreting $\tau$ | Compare to $\sigma$; if $\tau/\sigma > 1$, stations differ substantially |

---

## 10. Success Criteria

The project succeeds if we demonstrate:

1. **Partial pooling provides better estimates** than complete or no pooling (lower WAIC)
2. **Shrinkage is observable**: estimates for months with fewer data points move toward the mean
3. **Clear seasonal pattern**: $\beta$ coefficients show Jan < Feb < Mar < Apr
4. **Geographic gradient**: northern stations have lower $\alpha_i$ than southern stations
5. **Uncertainty quantification**: credible intervals appropriately capture estimation uncertainty
