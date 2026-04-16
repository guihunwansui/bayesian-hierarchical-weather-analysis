"""
DATASCI 451 - Run Bayesian Models (Fixed Version)
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DATASCI 451 - Bayesian Hierarchical Model Fitting")
print("="*70)

# Load data
df = pd.read_csv('data/selected_stations_monthly.csv')
print(f"\nData loaded: {len(df)} observations")

# Prepare indices
station_names = df['short_name'].unique()
station_idx = {name: i for i, name in enumerate(station_names)}
df['station_id'] = df['short_name'].map(station_idx)
df['month_id'] = df['Month'] - 1

y = df['TAVG_mean'].values
station = df['station_id'].values
month = df['month_id'].values

n_stations = len(station_names)
n_months = 4

print(f"Stations: {n_stations}, Months: {n_months}, Observations: {len(y)}")

# ============================================================
# Model 1: Complete Pooling
# ============================================================
print("\n" + "-"*60)
print("Model 1: Complete Pooling")

with pm.Model() as model_cp:
    mu = pm.Normal('mu', mu=25, sigma=20)
    beta = pm.Normal('beta', mu=0, sigma=15, shape=n_months)
    sigma = pm.HalfCauchy('sigma', beta=10)
    mu_y = mu + beta[month]
    y_obs = pm.Normal('y_obs', mu=mu_y, sigma=sigma, observed=y)
    trace_cp = pm.sample(2000, tune=1000, cores=2, random_seed=451, return_inferencedata=True)

trace_cp.to_netcdf('data/trace_complete_pooling.nc')
print("  Saved!")

# ============================================================
# Model 2: No Pooling
# ============================================================
print("\n" + "-"*60)
print("Model 2: No Pooling")

with pm.Model() as model_np:
    alpha = pm.Normal('alpha', mu=25, sigma=20, shape=n_stations)
    beta = pm.Normal('beta', mu=0, sigma=15, shape=n_months)
    sigma = pm.HalfCauchy('sigma', beta=10)
    mu_y = alpha[station] + beta[month]
    y_obs = pm.Normal('y_obs', mu=mu_y, sigma=sigma, observed=y)
    trace_np = pm.sample(2000, tune=1000, cores=2, random_seed=451, return_inferencedata=True)

trace_np.to_netcdf('data/trace_no_pooling.nc')
print("  Saved!")

# ============================================================
# Model 3: Hierarchical
# ============================================================
print("\n" + "-"*60)
print("Model 3: Hierarchical (Partial Pooling)")

with pm.Model() as model_hier:
    mu_alpha = pm.Normal('mu_alpha', mu=25, sigma=20)
    tau = pm.HalfCauchy('tau', beta=10)
    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_stations)
    alpha = pm.Deterministic('alpha', mu_alpha + tau * alpha_offset)
    beta = pm.Normal('beta', mu=0, sigma=15, shape=n_months)
    sigma = pm.HalfCauchy('sigma', beta=10)
    mu_y = alpha[station] + beta[month]
    y_obs = pm.Normal('y_obs', mu=mu_y, sigma=sigma, observed=y)
    trace_hier = pm.sample(2000, tune=1000, cores=2, random_seed=451, return_inferencedata=True)

trace_hier.to_netcdf('data/trace_hierarchical.nc')
print("  Saved!")

# Save station mapping
pd.DataFrame({'station_name': station_names, 'station_id': range(n_stations)}).to_csv(
    'data/station_mapping.csv', index=False)

# ============================================================
# Results Summary
# ============================================================
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

# Hierarchical parameters
mu_alpha = float(trace_hier.posterior['mu_alpha'].mean())
tau = float(trace_hier.posterior['tau'].mean())
sigma_h = float(trace_hier.posterior['sigma'].mean())
alpha_hier = trace_hier.posterior['alpha'].mean(dim=['chain', 'draw']).values
beta_hier = trace_hier.posterior['beta'].mean(dim=['chain', 'draw']).values
alpha_np = trace_np.posterior['alpha'].mean(dim=['chain', 'draw']).values

print(f"\nHierarchical Model Parameters:")
print(f"  μ_α = {mu_alpha:.2f}°F (population mean)")
print(f"  τ   = {tau:.2f}°F (between-station SD)")
print(f"  σ   = {sigma_h:.2f}°F (observation noise)")
print(f"  τ/σ = {tau/sigma_h:.2f}")

print(f"\nStation Effects (α):")
for i, name in enumerate(station_names):
    shrink = alpha_np[i] - alpha_hier[i]
    print(f"  {name:<20}: No Pool={alpha_np[i]:.1f}, Hier={alpha_hier[i]:.1f}, Shrink={shrink:+.1f}°F")

print(f"\nMonth Effects (β):")
for i, m in enumerate(['Jan', 'Feb', 'Mar', 'Apr']):
    print(f"  {m}: {beta_hier[i]:+.1f}°F")

# Model comparison
print("\n" + "-"*60)
print("Model Comparison (WAIC)")
with model_cp:
    pm.compute_log_likelihood(trace_cp)
with model_np:
    pm.compute_log_likelihood(trace_np)
with model_hier:
    pm.compute_log_likelihood(trace_hier)

comparison = az.compare({
    'Complete Pooling': trace_cp,
    'No Pooling': trace_np,
    'Hierarchical': trace_hier
}, ic='waic')
comparison.to_csv('data/model_comparison.csv')
print(comparison[['elpd_waic', 'p_waic', 'weight']].to_string())

print("\n" + "="*70)
print("DONE! Traces saved to data/ directory")
