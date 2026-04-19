"""
DATASCI 451 - Fit Bayesian Models on Daily Data
================================================
Fit three models:
1. Complete Pooling: All stations share one baseline
2. No Pooling: Each station has independent baseline
3. Hierarchical: Station baselines from population distribution

Model:
  y_it ~ N(alpha_i + beta_month(t), sigma^2)

For Hierarchical:
  alpha_i ~ N(mu_alpha, tau^2)
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FITTING BAYESIAN MODELS ON DAILY DATA")
print("="*70)

# Load prepared data
df = pd.read_csv('data/daily_data_prepared.csv')
station_meta = pd.read_csv('data/station_metadata.csv')

# Extract arrays for modeling
y = df['TEMP'].values
station = df['station_id'].values
month = df['month_id'].values

n_stations = station_meta['station_id'].nunique()
n_months = 4
n_obs = len(y)

print(f"\nData summary:")
print(f"  Observations: {n_obs}")
print(f"  Stations: {n_stations}")
print(f"  Months: {n_months}")

# ============================================================
# Model 1: Complete Pooling
# ============================================================
print("\n" + "-"*60)
print("Model 1: Complete Pooling")
print("-"*60)

with pm.Model() as model_cp:
    # Single baseline for all stations
    mu = pm.Normal('mu', mu=30, sigma=20)

    # Month effects
    beta = pm.Normal('beta', mu=0, sigma=15, shape=n_months)

    # Noise (daily temperature variability)
    sigma = pm.HalfCauchy('sigma', beta=10)

    # Likelihood
    mu_y = mu + beta[month]
    y_obs = pm.Normal('y_obs', mu=mu_y, sigma=sigma, observed=y)

    # Sample
    trace_cp = pm.sample(2000, tune=1000, cores=2, random_seed=451,
                        return_inferencedata=True, progressbar=True)

trace_cp.to_netcdf('data/trace_daily_complete_pooling.nc')
print("  Saved: data/trace_daily_complete_pooling.nc")

# ============================================================
# Model 2: No Pooling
# ============================================================
print("\n" + "-"*60)
print("Model 2: No Pooling")
print("-"*60)

with pm.Model() as model_np:
    # Independent baseline for each station
    alpha = pm.Normal('alpha', mu=30, sigma=20, shape=n_stations)

    # Month effects (shared)
    beta = pm.Normal('beta', mu=0, sigma=15, shape=n_months)

    # Noise
    sigma = pm.HalfCauchy('sigma', beta=10)

    # Likelihood
    mu_y = alpha[station] + beta[month]
    y_obs = pm.Normal('y_obs', mu=mu_y, sigma=sigma, observed=y)

    # Sample
    trace_np = pm.sample(2000, tune=1000, cores=2, random_seed=451,
                        return_inferencedata=True, progressbar=True)

trace_np.to_netcdf('data/trace_daily_no_pooling.nc')
print("  Saved: data/trace_daily_no_pooling.nc")

# ============================================================
# Model 3: Hierarchical (Partial Pooling)
# ============================================================
print("\n" + "-"*60)
print("Model 3: Hierarchical (Partial Pooling)")
print("-"*60)

with pm.Model() as model_hier:
    # Population parameters
    mu_alpha = pm.Normal('mu_alpha', mu=30, sigma=20)
    tau = pm.HalfCauchy('tau', beta=10)

    # Station effects (non-centered parameterization)
    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_stations)
    alpha = pm.Deterministic('alpha', mu_alpha + tau * alpha_offset)

    # Month effects (shared)
    beta = pm.Normal('beta', mu=0, sigma=15, shape=n_months)

    # Noise
    sigma = pm.HalfCauchy('sigma', beta=10)

    # Likelihood
    mu_y = alpha[station] + beta[month]
    y_obs = pm.Normal('y_obs', mu=mu_y, sigma=sigma, observed=y)

    # Sample
    trace_hier = pm.sample(2000, tune=1000, cores=2, random_seed=451,
                          return_inferencedata=True, progressbar=True)

trace_hier.to_netcdf('data/trace_daily_hierarchical.nc')
print("  Saved: data/trace_daily_hierarchical.nc")

# ============================================================
# Quick Results Summary
# ============================================================
print("\n" + "="*70)
print("QUICK RESULTS SUMMARY")
print("="*70)

# Hierarchical parameters
mu_alpha_est = float(trace_hier.posterior['mu_alpha'].mean())
tau_est = float(trace_hier.posterior['tau'].mean())
sigma_hier = float(trace_hier.posterior['sigma'].mean())
sigma_np = float(trace_np.posterior['sigma'].mean())
sigma_cp = float(trace_cp.posterior['sigma'].mean())

print(f"\nHierarchical Model Population Parameters:")
print(f"  mu_alpha (population mean): {mu_alpha_est:.2f}F")
print(f"  tau (between-station SD):   {tau_est:.2f}F")
print(f"  sigma (daily noise):        {sigma_hier:.2f}F")

print(f"\nObservation Noise Comparison:")
print(f"  Complete Pooling: {sigma_cp:.2f}F")
print(f"  No Pooling:       {sigma_np:.2f}F")
print(f"  Hierarchical:     {sigma_hier:.2f}F")

# Month effects
beta_hier = trace_hier.posterior['beta'].mean(dim=['chain', 'draw']).values
print(f"\nMonth Effects (Hierarchical):")
for i, month_name in enumerate(['January', 'February', 'March', 'April']):
    print(f"  {month_name}: {beta_hier[i]:+.1f}F")

print("\n" + "="*70)
print("Model fitting complete!")
print("="*70)
