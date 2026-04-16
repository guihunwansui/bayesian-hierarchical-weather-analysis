"""
DATASCI 451 - Demonstrate Hierarchical Model Advantage

Two scenarios where hierarchical models clearly outperform:
1. Leave-One-Station-Out: Predicting a NEW station
2. Sparse Data Simulation: When some stations have very few observations
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DEMONSTRATING HIERARCHICAL MODEL ADVANTAGE")
print("="*70)

# Load data
df = pd.read_csv('data/selected_stations_monthly.csv')
station_names = list(df['short_name'].unique())
n_stations = len(station_names)

# Station coordinates for spatial prediction
coords = {
    'Ann Arbor UMich': (42.28, -83.74),
    'Atlanta MI': (45.00, -84.14),
    'Bad Axe': (43.80, -83.00),
    'Bergland Dam': (46.59, -89.57),
    'Traverse City': (44.74, -85.58),
    'Pontiac Airport': (42.67, -83.42),
    'Gwinn Sawyer AFB': (46.35, -87.40),
    'Iron Mountain': (45.82, -88.11),
}

# ============================================================
# TEST 1: Leave-One-Station-Out (LOSO)
# Predict ALL observations for a held-out station
# ============================================================
print("\n" + "="*70)
print("TEST 1: LEAVE-ONE-STATION-OUT")
print("="*70)
print("\nScenario: A new weather station is installed. Can we predict its")
print("temperature using data from nearby stations?")
print("\nNo Pooling: Cannot predict (no data for new station)")
print("Hierarchical: Uses population distribution to predict\n")

def fit_hierarchical_loso(df_train, station_names_train, n_samples=1500):
    """Fit hierarchical model for LOSO test."""
    station_idx = {name: i for i, name in enumerate(station_names_train)}
    df_train = df_train.copy()
    df_train['station_id'] = df_train['short_name'].map(station_idx)
    df_train['month_id'] = df_train['Month'] - 1

    y = df_train['TAVG_mean'].values
    station = df_train['station_id'].values
    month = df_train['month_id'].values
    n_stations = len(station_names_train)

    with pm.Model() as model:
        mu_alpha = pm.Normal('mu_alpha', mu=25, sigma=20)
        tau = pm.HalfCauchy('tau', beta=10)
        alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_stations)
        alpha = pm.Deterministic('alpha', mu_alpha + tau * alpha_offset)
        beta = pm.Normal('beta', mu=0, sigma=15, shape=4)
        sigma = pm.HalfCauchy('sigma', beta=10)

        mu_y = alpha[station] + beta[month]
        y_obs = pm.Normal('y_obs', mu=mu_y, sigma=sigma, observed=y)

        trace = pm.sample(n_samples, tune=500, cores=2, random_seed=42,
                         return_inferencedata=True, progressbar=False)

    return trace

loso_results = []

# Test with 3 stations (to save time)
test_stations = ['Bergland Dam', 'Ann Arbor UMich', 'Traverse City']

for held_out_station in test_stations:
    print(f"\nHeld out: {held_out_station}")

    # Get true values
    true_vals = df[df['short_name'] == held_out_station][['Month', 'TAVG_mean']].values

    # Training data (all other stations)
    df_train = df[df['short_name'] != held_out_station].copy()
    train_stations = [s for s in station_names if s != held_out_station]

    # Fit hierarchical model
    trace = fit_hierarchical_loso(df_train, train_stations)

    # Predict for new station using population distribution
    mu_alpha = trace.posterior['mu_alpha'].values.flatten()
    tau = trace.posterior['tau'].values.flatten()
    beta = trace.posterior['beta'].values.reshape(-1, 4)
    sigma = trace.posterior['sigma'].values.flatten()

    # New station's alpha comes from population distribution
    alpha_new = np.random.normal(mu_alpha, tau)

    # Predict each month
    hier_errors = []
    for month, true_temp in true_vals:
        month_idx = int(month) - 1
        pred_mu = alpha_new + beta[:, month_idx]
        pred_mean = pred_mu.mean()
        hier_error = abs(pred_mean - true_temp)
        hier_errors.append(hier_error)

        loso_results.append({
            'held_out': held_out_station,
            'month': int(month),
            'true': true_temp,
            'hier_pred': pred_mean,
            'hier_error': hier_error
        })

    print(f"  Mean prediction error: {np.mean(hier_errors):.2f}°F")

loso_df = pd.DataFrame(loso_results)

print("\n" + "-"*70)
print("LOSO RESULTS SUMMARY")
print("-"*70)
print(f"\n{'Held-out Station':<20} {'Month':>6} {'True':>8} {'Pred':>8} {'Error':>8}")
print("-"*50)
for _, row in loso_df.iterrows():
    print(f"{row['held_out']:<20} {row['month']:>6} {row['true']:>8.1f} "
          f"{row['hier_pred']:>8.1f} {row['hier_error']:>8.1f}")

print(f"\nOverall Mean Error: {loso_df['hier_error'].mean():.2f}°F")
print("\n→ Hierarchical model CAN predict new stations!")
print("→ No Pooling CANNOT (would need data from the new station)")

# ============================================================
# TEST 2: Sparse Data Simulation
# Some stations have only 1-2 observations
# ============================================================
print("\n\n" + "="*70)
print("TEST 2: SPARSE DATA SIMULATION")
print("="*70)
print("\nScenario: Some stations have very few observations.")
print("Which model handles sparse data better?")

def create_sparse_data(df, sparse_stations, n_obs_sparse=1):
    """Create dataset where some stations have sparse data."""
    df_sparse = []
    for station in df['short_name'].unique():
        station_data = df[df['short_name'] == station].copy()
        if station in sparse_stations:
            # Keep only first n observations
            station_data = station_data.head(n_obs_sparse)
        df_sparse.append(station_data)
    return pd.concat(df_sparse, ignore_index=True)

def fit_and_predict_sparse(df_train, df_full, station_names, model_type='hierarchical'):
    """Fit model on sparse data, predict for all station-months."""
    station_idx = {name: i for i, name in enumerate(station_names)}
    df_train = df_train.copy()
    df_train['station_id'] = df_train['short_name'].map(station_idx)
    df_train['month_id'] = df_train['Month'] - 1

    y = df_train['TAVG_mean'].values
    station = df_train['station_id'].values
    month = df_train['month_id'].values
    n_stations = len(station_names)

    with pm.Model() as model:
        if model_type == 'hierarchical':
            mu_alpha = pm.Normal('mu_alpha', mu=25, sigma=20)
            tau = pm.HalfCauchy('tau', beta=10)
            alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_stations)
            alpha = pm.Deterministic('alpha', mu_alpha + tau * alpha_offset)
        else:  # no pooling
            alpha = pm.Normal('alpha', mu=25, sigma=20, shape=n_stations)

        beta = pm.Normal('beta', mu=0, sigma=15, shape=4)
        sigma = pm.HalfCauchy('sigma', beta=10)

        mu_y = alpha[station] + beta[month]
        y_obs = pm.Normal('y_obs', mu=mu_y, sigma=sigma, observed=y)

        trace = pm.sample(1500, tune=500, cores=2, random_seed=42,
                         return_inferencedata=True, progressbar=False)

    # Predict for all station-months
    alpha_post = trace.posterior['alpha'].mean(dim=['chain', 'draw']).values
    beta_post = trace.posterior['beta'].mean(dim=['chain', 'draw']).values

    predictions = []
    for _, row in df_full.iterrows():
        station = row['short_name']
        month = int(row['Month'])
        true_val = row['TAVG_mean']

        pred = alpha_post[station_idx[station]] + beta_post[month - 1]
        error = abs(pred - true_val)

        predictions.append({
            'station': station,
            'month': month,
            'true': true_val,
            'pred': pred,
            'error': error,
            'is_sparse': station in sparse_stations
        })

    return pd.DataFrame(predictions)

# Designate some stations as sparse (only 1 observation each)
sparse_stations = ['Bergland Dam', 'Gwinn Sawyer AFB', 'Iron Mountain']
n_obs_sparse = 1

print(f"\nSparse stations (only {n_obs_sparse} obs each): {sparse_stations}")
print(f"Full stations (4 obs each): {[s for s in station_names if s not in sparse_stations]}")

# Create sparse dataset
df_sparse = create_sparse_data(df, sparse_stations, n_obs_sparse)
print(f"\nTraining data: {len(df_sparse)} observations")

# Fit both models
print("\nFitting Hierarchical model...")
hier_pred = fit_and_predict_sparse(df_sparse, df, station_names, 'hierarchical')

print("Fitting No Pooling model...")
np_pred = fit_and_predict_sparse(df_sparse, df, station_names, 'no_pooling')

# Compare errors
print("\n" + "-"*70)
print("SPARSE DATA RESULTS")
print("-"*70)

# Results for sparse stations only
sparse_hier = hier_pred[hier_pred['is_sparse']]
sparse_np = np_pred[np_pred['is_sparse']]

print("\nPrediction Errors for SPARSE STATIONS:")
print(f"  Hierarchical: {sparse_hier['error'].mean():.2f}°F (mean), {sparse_hier['error'].std():.2f}°F (std)")
print(f"  No Pooling:   {sparse_np['error'].mean():.2f}°F (mean), {sparse_np['error'].std():.2f}°F (std)")

# Results for full stations
full_hier = hier_pred[~hier_pred['is_sparse']]
full_np = np_pred[~np_pred['is_sparse']]

print("\nPrediction Errors for FULL DATA STATIONS:")
print(f"  Hierarchical: {full_hier['error'].mean():.2f}°F (mean)")
print(f"  No Pooling:   {full_np['error'].mean():.2f}°F (mean)")

improvement = (sparse_np['error'].mean() - sparse_hier['error'].mean()) / sparse_np['error'].mean() * 100

print(f"\n→ Hierarchical improvement for sparse stations: {improvement:.1f}%")

# ============================================================
# Visualization
# ============================================================
print("\n\nGENERATING VISUALIZATIONS...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: LOSO results
ax1 = axes[0]
stations_unique = loso_df['held_out'].unique()
x = np.arange(len(stations_unique))
width = 0.8

for i, station in enumerate(stations_unique):
    station_data = loso_df[loso_df['held_out'] == station]
    errors = station_data['hier_error'].values
    ax1.bar(i, errors.mean(), width, color='coral', edgecolor='black',
            yerr=errors.std(), capsize=5)

ax1.set_xticks(x)
ax1.set_xticklabels([s[:12] for s in stations_unique], rotation=15)
ax1.set_ylabel('Mean Absolute Error (°F)')
ax1.set_title('Leave-One-Station-Out Prediction\n(Hierarchical model predicting NEW stations)')
ax1.axhline(loso_df['hier_error'].mean(), color='red', linestyle='--',
            label=f'Mean: {loso_df["hier_error"].mean():.1f}°F')
ax1.legend()
ax1.set_ylim(0, 15)

# Plot 2: Sparse data comparison
ax2 = axes[1]
labels = ['Sparse Stations\n(1 obs each)', 'Full Stations\n(4 obs each)']
hier_errors = [sparse_hier['error'].mean(), full_hier['error'].mean()]
np_errors = [sparse_np['error'].mean(), full_np['error'].mean()]

x = np.arange(len(labels))
width = 0.35

bars1 = ax2.bar(x - width/2, hier_errors, width, label='Hierarchical', color='coral', edgecolor='black')
bars2 = ax2.bar(x + width/2, np_errors, width, label='No Pooling', color='steelblue', edgecolor='black')

ax2.set_ylabel('Mean Absolute Error (°F)')
ax2.set_title('Sparse Data Performance\n(Hierarchical excels with limited data)')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()
ax2.set_ylim(0, max(np_errors) * 1.3)

# Add value labels
for bar, val in zip(bars1, hier_errors):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.1f}',
             ha='center', fontsize=11, fontweight='bold')
for bar, val in zip(bars2, np_errors):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.1f}',
             ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/22_hierarchical_advantage.png', dpi=150, bbox_inches='tight')
print("  Saved: plots/22_hierarchical_advantage.png")

# ============================================================
# Summary
# ============================================================
print("\n\n" + "="*70)
print("SUMMARY: WHEN HIERARCHICAL MODELS EXCEL")
print("="*70)

print(f"""
SCENARIO 1: PREDICTING NEW STATIONS (LOSO)
  - No Pooling: IMPOSSIBLE (no data for new station)
  - Hierarchical: Mean error = {loso_df['hier_error'].mean():.1f}°F
  - Uses population mean (μ_α) and variance (τ) to predict

SCENARIO 2: SPARSE DATA
  - For stations with only 1 observation:
    - Hierarchical error: {sparse_hier['error'].mean():.1f}°F
    - No Pooling error:   {sparse_np['error'].mean():.1f}°F
    - Improvement: {improvement:.1f}%

  - For stations with full data:
    - Both models perform similarly (as expected)

KEY INSIGHT:
  Hierarchical models shine when:
  1. Predicting for new/unseen groups
  2. Groups have very few observations (<3)
  3. You need uncertainty estimates that account for group structure

  In our original analysis with 4 obs/station, both models had enough
  data, so the advantage was minimal. The value of hierarchical modeling
  is in these edge cases and in principled uncertainty quantification.
""")

print("="*70)
print("Analysis complete!")
