"""
DATASCI 451 - Run Bayesian Models
Quick script to fit all three models and save results
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("DATASCI 451 - Bayesian Hierarchical Model Fitting")
print("="*60)

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
n_obs = len(y)

print(f"Stations: {n_stations}, Months: {n_months}, Observations: {n_obs}")
print(f"Station names: {list(station_names)}")

# ============================================================
# Model 1: Complete Pooling
# ============================================================
print("\n" + "-"*60)
print("Fitting Model 1: Complete Pooling...")
print("-"*60)

with pm.Model() as model_cp:
    mu = pm.Normal('mu', mu=25, sigma=20)
    beta = pm.Normal('beta', mu=0, sigma=15, shape=n_months)
    sigma = pm.HalfCauchy('sigma', beta=10)

    mu_y = mu + beta[month]
    y_obs = pm.Normal('y_obs', mu=mu_y, sigma=sigma, observed=y)

    trace_cp = pm.sample(2000, tune=1000, cores=2, random_seed=451,
                         return_inferencedata=True, progressbar=True)

print("\nComplete Pooling Results:")
print(az.summary(trace_cp, var_names=['mu', 'beta', 'sigma']))

# ============================================================
# Model 2: No Pooling
# ============================================================
print("\n" + "-"*60)
print("Fitting Model 2: No Pooling...")
print("-"*60)

with pm.Model() as model_np:
    alpha = pm.Normal('alpha', mu=25, sigma=20, shape=n_stations)
    beta = pm.Normal('beta', mu=0, sigma=15, shape=n_months)
    sigma = pm.HalfCauchy('sigma', beta=10)

    mu_y = alpha[station] + beta[month]
    y_obs = pm.Normal('y_obs', mu=mu_y, sigma=sigma, observed=y)

    trace_np = pm.sample(2000, tune=1000, cores=2, random_seed=451,
                         return_inferencedata=True, progressbar=True)

print("\nNo Pooling Results:")
print(az.summary(trace_np, var_names=['alpha', 'beta', 'sigma']))

# ============================================================
# Model 3: Partial Pooling (Hierarchical)
# ============================================================
print("\n" + "-"*60)
print("Fitting Model 3: Hierarchical (Partial Pooling)...")
print("-"*60)

with pm.Model() as model_hier:
    # Hyperpriors
    mu_alpha = pm.Normal('mu_alpha', mu=25, sigma=20)
    tau = pm.HalfCauchy('tau', beta=10)

    # Station effects (non-centered)
    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_stations)
    alpha = pm.Deterministic('alpha', mu_alpha + tau * alpha_offset)

    # Month effects
    beta = pm.Normal('beta', mu=0, sigma=15, shape=n_months)

    # Noise
    sigma = pm.HalfCauchy('sigma', beta=10)

    # Likelihood
    mu_y = alpha[station] + beta[month]
    y_obs = pm.Normal('y_obs', mu=mu_y, sigma=sigma, observed=y)

    trace_hier = pm.sample(2000, tune=1000, cores=2, random_seed=451,
                           return_inferencedata=True, progressbar=True)

print("\nHierarchical Model Results:")
print(az.summary(trace_hier, var_names=['mu_alpha', 'tau', 'alpha', 'beta', 'sigma']))

# ============================================================
# Model Comparison
# ============================================================
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

# Compute log likelihood
with model_cp:
    pm.compute_log_likelihood(trace_cp)
with model_np:
    pm.compute_log_likelihood(trace_np)
with model_hier:
    pm.compute_log_likelihood(trace_hier)

# WAIC comparison
comparison = az.compare({
    'Complete Pooling': trace_cp,
    'No Pooling': trace_np,
    'Hierarchical': trace_hier
}, ic='waic')

print("\nWAIC Comparison (lower is better):")
print(comparison[['rank', 'elpd_waic', 'p_waic', 'waic', 'weight']])

# ============================================================
# Key Results Summary
# ============================================================
print("\n" + "="*60)
print("KEY RESULTS SUMMARY")
print("="*60)

# Hierarchical model parameters
mu_alpha_val = float(trace_hier.posterior['mu_alpha'].mean())
tau_val = float(trace_hier.posterior['tau'].mean())
sigma_val = float(trace_hier.posterior['sigma'].mean())
alpha_vals = trace_hier.posterior['alpha'].mean(dim=['chain', 'draw']).values
beta_vals = trace_hier.posterior['beta'].mean(dim=['chain', 'draw']).values

print(f"\nHyperparameters:")
print(f"  μ_α (population mean): {mu_alpha_val:.2f}°F")
print(f"  τ (between-station SD): {tau_val:.2f}°F")
print(f"  σ (observation noise): {sigma_val:.2f}°F")
print(f"  τ/σ ratio: {tau_val/sigma_val:.2f}")

print(f"\nStation Effects (α):")
for i, name in enumerate(station_names):
    print(f"  {name:<20}: {alpha_vals[i]:.2f}°F")

print(f"\nMonth Effects (β):")
month_names = ['January', 'February', 'March', 'April']
for i, name in enumerate(month_names):
    print(f"  {name:<10}: {beta_vals[i]:+.2f}°F")

# Shrinkage analysis
alpha_np = trace_np.posterior['alpha'].mean(dim=['chain', 'draw']).values
print(f"\nShrinkage Analysis:")
print(f"  {'Station':<20} {'No Pool':>10} {'Hier':>10} {'Shrinkage':>12}")
for i, name in enumerate(station_names):
    shrink = alpha_np[i] - alpha_vals[i]
    print(f"  {name:<20} {alpha_np[i]:>10.2f} {alpha_vals[i]:>10.2f} {shrink:>+12.2f}°F")

# ============================================================
# Save Results
# ============================================================
print("\n" + "-"*60)
print("Saving results...")

trace_cp.to_netcdf('data/trace_complete_pooling.nc')
trace_np.to_netcdf('data/trace_no_pooling.nc')
trace_hier.to_netcdf('data/trace_hierarchical.nc')

# Save station mapping
pd.DataFrame({
    'station_name': station_names,
    'station_id': range(n_stations)
}).to_csv('data/station_mapping.csv', index=False)

# Save comparison
comparison.to_csv('data/model_comparison.csv')

print("Results saved to data/ directory")
print("\n" + "="*60)
print("DONE!")
print("="*60)
