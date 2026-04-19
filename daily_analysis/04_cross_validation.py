"""
DATASCI 451 - Cross-Validation Analysis
========================================
Proper comparison: Hold out data and compare prediction accuracy.

Key insight: For stations with few observations, hierarchical should
predict held-out data better because it "borrows strength" from other stations.
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CROSS-VALIDATION: COMPARING HIERARCHICAL vs NO-POOLING")
print("="*70)

# Load data
df = pd.read_csv('data/daily_data_prepared.csv')
station_meta = pd.read_csv('data/station_metadata.csv')

# We'll do a simple hold-out CV:
# For each station, hold out 20% of observations and predict them
np.random.seed(451)

print("\nSplitting data: 80% train, 20% test for each station...")

train_indices = []
test_indices = []

for station_id in station_meta['station_id']:
    station_data = df[df['station_id'] == station_id]
    n = len(station_data)
    n_test = max(1, int(n * 0.2))  # At least 1 test observation

    indices = station_data.index.tolist()
    np.random.shuffle(indices)

    test_indices.extend(indices[:n_test])
    train_indices.extend(indices[n_test:])

df_train = df.loc[train_indices].copy()
df_test = df.loc[test_indices].copy()

print(f"Training observations: {len(df_train)}")
print(f"Test observations: {len(df_test)}")

# ============================================================
# Fit models on training data
# ============================================================
print("\n" + "-"*60)
print("Fitting models on training data...")
print("-"*60)

y_train = df_train['TEMP'].values
station_train = df_train['station_id'].values
month_train = df_train['month_id'].values

n_stations = station_meta['station_id'].nunique()
n_months = 4

# No Pooling
print("\nFitting No Pooling model...")
with pm.Model() as model_np:
    alpha = pm.Normal('alpha', mu=30, sigma=20, shape=n_stations)
    beta = pm.Normal('beta', mu=0, sigma=15, shape=n_months)
    sigma = pm.HalfCauchy('sigma', beta=10)
    mu_y = alpha[station_train] + beta[month_train]
    y_obs = pm.Normal('y_obs', mu=mu_y, sigma=sigma, observed=y_train)
    trace_np = pm.sample(1500, tune=500, cores=2, random_seed=451,
                        return_inferencedata=True, progressbar=True)

# Hierarchical
print("\nFitting Hierarchical model...")
with pm.Model() as model_hier:
    mu_alpha = pm.Normal('mu_alpha', mu=30, sigma=20)
    tau = pm.HalfCauchy('tau', beta=10)
    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_stations)
    alpha = pm.Deterministic('alpha', mu_alpha + tau * alpha_offset)
    beta = pm.Normal('beta', mu=0, sigma=15, shape=n_months)
    sigma = pm.HalfCauchy('sigma', beta=10)
    mu_y = alpha[station_train] + beta[month_train]
    y_obs = pm.Normal('y_obs', mu=mu_y, sigma=sigma, observed=y_train)
    trace_hier = pm.sample(1500, tune=500, cores=2, random_seed=451,
                          return_inferencedata=True, progressbar=True)

# Extract posterior means
alpha_np = trace_np.posterior['alpha'].mean(dim=['chain', 'draw']).values
alpha_hier = trace_hier.posterior['alpha'].mean(dim=['chain', 'draw']).values
beta_np = trace_np.posterior['beta'].mean(dim=['chain', 'draw']).values
beta_hier = trace_hier.posterior['beta'].mean(dim=['chain', 'draw']).values
mu_alpha = float(trace_hier.posterior['mu_alpha'].mean())
tau = float(trace_hier.posterior['tau'].mean())

# ============================================================
# Predict on test data
# ============================================================
print("\n" + "-"*60)
print("Predicting on held-out test data...")
print("-"*60)

# Predictions
pred_np = alpha_np[df_test['station_id'].values] + beta_np[df_test['month_id'].values]
pred_hier = alpha_hier[df_test['station_id'].values] + beta_hier[df_test['month_id'].values]
actual = df_test['TEMP'].values

# Calculate errors per station
test_results = []
for station_id in station_meta['station_id']:
    mask = df_test['station_id'] == station_id
    if mask.sum() == 0:
        continue

    station_name = station_meta.loc[station_meta['station_id'] == station_id, 'station_name'].values[0]
    n_train = (df_train['station_id'] == station_id).sum()
    n_test = mask.sum()

    mae_np = np.mean(np.abs(pred_np[mask] - actual[mask]))
    mae_hier = np.mean(np.abs(pred_hier[mask] - actual[mask]))

    test_results.append({
        'station_id': station_id,
        'station_name': station_name,
        'n_train': n_train,
        'n_test': n_test,
        'mae_np': mae_np,
        'mae_hier': mae_hier,
        'improvement': mae_np - mae_hier
    })

results_df = pd.DataFrame(test_results)

# Group by training data availability
bins = [0, 30, 50, 70, 90, 200]
labels = ['<30', '30-50', '50-70', '70-90', '90+']
results_df['train_group'] = pd.cut(results_df['n_train'], bins=bins, labels=labels)

print(f"\nCross-Validation Results by Training Data Availability:")
print(f"{'Train Group':<12} {'N Stations':>12} {'NP MAE':>10} {'Hier MAE':>10} {'Improvement':>12}")
print("-"*60)

for group in labels:
    group_data = results_df[results_df['train_group'] == group]
    if len(group_data) > 0:
        np_mae = group_data['mae_np'].mean()
        hier_mae = group_data['mae_hier'].mean()
        improvement = (np_mae - hier_mae) / np_mae * 100
        print(f"{group:<12} {len(group_data):>12} {np_mae:>10.2f}F {hier_mae:>10.2f}F {improvement:>+11.1f}%")

# Overall
print("-"*60)
overall_np = results_df['mae_np'].mean()
overall_hier = results_df['mae_hier'].mean()
overall_improvement = (overall_np - overall_hier) / overall_np * 100
print(f"{'Overall':<12} {len(results_df):>12} {overall_np:>10.2f}F {overall_hier:>10.2f}F {overall_improvement:>+11.1f}%")

# ============================================================
# Visualization
# ============================================================
print("\n" + "-"*60)
print("Generating visualization...")
print("-"*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: MAE by training data availability
ax1 = axes[0]
group_stats = results_df.groupby('train_group').agg({
    'mae_np': 'mean',
    'mae_hier': 'mean'
}).reindex(labels)

x = np.arange(len(labels))
width = 0.35
bars1 = ax1.bar(x - width/2, group_stats['mae_hier'], width, label='Hierarchical',
                color='coral', edgecolor='black')
bars2 = ax1.bar(x + width/2, group_stats['mae_np'], width, label='No Pooling',
                color='steelblue', edgecolor='black')

ax1.set_xlabel('Training Observations per Station', fontsize=12)
ax1.set_ylabel('Test MAE (F)', fontsize=12)
ax1.set_title('Cross-Validation: Held-Out Prediction Error\n(Lower is better)', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Scatter of improvement vs training size
ax2 = axes[1]
scatter = ax2.scatter(results_df['n_train'], results_df['improvement'],
                      c=results_df['n_train'], cmap='RdYlBu_r',
                      s=60, alpha=0.7, edgecolors='black')
ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2.set_xlabel('Training Observations', fontsize=12)
ax2.set_ylabel('Improvement (NP MAE - Hier MAE) in F', fontsize=12)
ax2.set_title('Hierarchical Improvement by Station\n(Positive = Hierarchical better)', fontsize=13)
ax2.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(results_df['n_train'], results_df['improvement'], 1)
p = np.poly1d(z)
x_line = np.linspace(results_df['n_train'].min(), results_df['n_train'].max(), 100)
ax2.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend')
ax2.legend()

plt.colorbar(scatter, ax=ax2, label='Training observations')

plt.tight_layout()
plt.savefig('plots/D04_cross_validation.png', dpi=150, bbox_inches='tight')
print("  Saved: plots/D04_cross_validation.png")

# Save results
results_df.to_csv('data/cv_results.csv', index=False)
print("  Saved: data/cv_results.csv")

# ============================================================
# Key Finding
# ============================================================
print("\n" + "="*70)
print("KEY FINDING")
print("="*70)

sparse_results = results_df[results_df['train_group'] == '<30']
if len(sparse_results) > 0:
    sparse_improvement = (sparse_results['mae_np'].mean() - sparse_results['mae_hier'].mean()) / sparse_results['mae_np'].mean() * 100
    print(f"""
For stations with < 30 training observations:
  - No Pooling Test MAE:   {sparse_results['mae_np'].mean():.2f}F
  - Hierarchical Test MAE: {sparse_results['mae_hier'].mean():.2f}F
  - Improvement:           {sparse_improvement:+.1f}%

This demonstrates that hierarchical models better predict HELD-OUT
data for sparse-data stations by "borrowing strength" from the
population distribution.
""")

print("="*70)
print("Cross-validation analysis complete!")
print("="*70)
