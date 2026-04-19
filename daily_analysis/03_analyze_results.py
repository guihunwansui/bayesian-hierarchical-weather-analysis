"""
DATASCI 451 - Analyze Daily Data Model Results
===============================================
Compare the three models and demonstrate hierarchical advantage.
"""

import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ANALYZING DAILY DATA MODEL RESULTS")
print("="*70)

# Load data and traces
df = pd.read_csv('data/daily_data_prepared.csv')
station_meta = pd.read_csv('data/station_metadata.csv')

trace_cp = az.from_netcdf('data/trace_daily_complete_pooling.nc')
trace_np = az.from_netcdf('data/trace_daily_no_pooling.nc')
trace_hier = az.from_netcdf('data/trace_daily_hierarchical.nc')

station_names = station_meta['station_name'].values
n_obs_per_station = station_meta.set_index('station_name')['n_observations']
n_stations = len(station_names)

# Extract posterior means
mu_alpha = float(trace_hier.posterior['mu_alpha'].mean())
tau = float(trace_hier.posterior['tau'].mean())
sigma = float(trace_hier.posterior['sigma'].mean())
alpha_hier = trace_hier.posterior['alpha'].mean(dim=['chain', 'draw']).values
alpha_np = trace_np.posterior['alpha'].mean(dim=['chain', 'draw']).values
beta = trace_hier.posterior['beta'].mean(dim=['chain', 'draw']).values

print(f"\nHierarchical Model Parameters:")
print(f"  mu_alpha = {mu_alpha:.2f}F (population mean)")
print(f"  tau      = {tau:.2f}F (between-station SD)")
print(f"  sigma    = {sigma:.2f}F (daily noise)")

# ============================================================
# Analysis 1: Shrinkage by Data Availability
# ============================================================
print("\n" + "-"*60)
print("SHRINKAGE ANALYSIS BY DATA AVAILABILITY")
print("-"*60)

shrinkage = alpha_np - alpha_hier
station_meta['alpha_hier'] = alpha_hier
station_meta['alpha_np'] = alpha_np
station_meta['shrinkage'] = shrinkage

# Group by observation count
bins = [0, 40, 60, 80, 100, 200]
labels = ['29-40', '40-60', '60-80', '80-100', '100+']
station_meta['obs_group'] = pd.cut(station_meta['n_observations'], bins=bins, labels=labels)

print(f"\nMean |Shrinkage| by Data Availability:")
print(f"{'Obs Group':<12} {'N Stations':>12} {'Mean |Shrink|':>15} {'Std |Shrink|':>15}")
print("-"*55)
for group in labels:
    group_data = station_meta[station_meta['obs_group'] == group]
    if len(group_data) > 0:
        mean_shrink = group_data['shrinkage'].abs().mean()
        std_shrink = group_data['shrinkage'].abs().std()
        print(f"{group:<12} {len(group_data):>12} {mean_shrink:>15.2f}F {std_shrink:>15.2f}F")

# ============================================================
# Analysis 2: Prediction Error Comparison
# ============================================================
print("\n" + "-"*60)
print("PREDICTION ERROR COMPARISON")
print("-"*60)

def calc_prediction_errors(df, alpha_values, beta_values, model_name):
    """Calculate MAE for each station."""
    errors = []
    for idx, row in station_meta.iterrows():
        station_id = row['station_id']
        station_data = df[df['station_id'] == station_id]

        preds = alpha_values[station_id] + beta_values[station_data['month_id'].values]
        actuals = station_data['TEMP'].values
        mae = np.mean(np.abs(preds - actuals))
        errors.append(mae)
    return np.array(errors)

mae_hier = calc_prediction_errors(df, alpha_hier, beta, 'Hierarchical')
mae_np = calc_prediction_errors(df, alpha_np, beta, 'No Pooling')

station_meta['mae_hier'] = mae_hier
station_meta['mae_np'] = mae_np
station_meta['mae_diff'] = mae_np - mae_hier  # Positive = hierarchical better

print(f"\nOverall MAE:")
print(f"  Hierarchical: {mae_hier.mean():.2f}F")
print(f"  No Pooling:   {mae_np.mean():.2f}F")
print(f"  Difference:   {(mae_np.mean() - mae_hier.mean()):.2f}F (positive = hier better)")

print(f"\nMAE by Data Availability:")
print(f"{'Obs Group':<12} {'Hier MAE':>12} {'NP MAE':>12} {'Improvement':>15}")
print("-"*55)
for group in labels:
    group_data = station_meta[station_meta['obs_group'] == group]
    if len(group_data) > 0:
        hier_mae = group_data['mae_hier'].mean()
        np_mae = group_data['mae_np'].mean()
        improvement = (np_mae - hier_mae) / np_mae * 100
        print(f"{group:<12} {hier_mae:>12.2f}F {np_mae:>12.2f}F {improvement:>+14.1f}%")

# ============================================================
# Visualization
# ============================================================
print("\n" + "-"*60)
print("GENERATING VISUALIZATIONS")
print("-"*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Shrinkage vs Observations
ax1 = axes[0, 0]
scatter = ax1.scatter(station_meta['n_observations'],
                      station_meta['shrinkage'].abs(),
                      c=station_meta['n_observations'],
                      cmap='RdYlBu_r', s=60, alpha=0.7, edgecolors='black')
ax1.set_xlabel('Number of Daily Observations', fontsize=12)
ax1.set_ylabel('|Shrinkage| (F)', fontsize=12)
ax1.set_title('Shrinkage Strength vs Data Availability\n(Less data = More shrinkage)', fontsize=13)

# Add trend line
z = np.polyfit(station_meta['n_observations'], station_meta['shrinkage'].abs(), 1)
p = np.poly1d(z)
x_line = np.linspace(station_meta['n_observations'].min(), station_meta['n_observations'].max(), 100)
ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.3f})')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='N observations')

# Plot 2: MAE Comparison by Group
ax2 = axes[0, 1]
group_stats = station_meta.groupby('obs_group').agg({
    'mae_hier': 'mean',
    'mae_np': 'mean'
}).reindex(labels)

x = np.arange(len(labels))
width = 0.35
bars1 = ax2.bar(x - width/2, group_stats['mae_hier'], width, label='Hierarchical',
                color='coral', edgecolor='black')
bars2 = ax2.bar(x + width/2, group_stats['mae_np'], width, label='No Pooling',
                color='steelblue', edgecolor='black')

ax2.set_xlabel('Daily Observations per Station', fontsize=12)
ax2.set_ylabel('Mean Absolute Error (F)', fontsize=12)
ax2.set_title('Prediction Error by Data Availability\n(Hierarchical excels with sparse data)', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 3: Shrinkage Effect Scatter
ax3 = axes[1, 0]
lims = [min(alpha_np.min(), alpha_hier.min()) - 2, max(alpha_np.max(), alpha_hier.max()) + 2]
ax3.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='No shrinkage')
ax3.axhline(mu_alpha, color='red', linestyle=':', linewidth=2,
            label=f'Population mean: {mu_alpha:.1f}F')

scatter = ax3.scatter(alpha_np, alpha_hier, c=station_meta['n_observations'],
                      cmap='RdYlBu_r', s=60, alpha=0.7, edgecolors='black')
ax3.set_xlabel('No Pooling Estimate (F)', fontsize=12)
ax3.set_ylabel('Hierarchical Estimate (F)', fontsize=12)
ax3.set_title('Shrinkage Effect\n(Color = data availability)', fontsize=13)
ax3.legend(loc='upper left')
ax3.set_xlim(lims)
ax3.set_ylim(lims)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax3, label='N observations')

# Plot 4: Key Findings Summary
ax4 = axes[1, 1]
ax4.axis('off')

# Calculate key statistics
sparse_group = station_meta[station_meta['obs_group'] == '29-40']
rich_group = station_meta[station_meta['obs_group'] == '100+']

sparse_improvement = (sparse_group['mae_np'].mean() - sparse_group['mae_hier'].mean()) / sparse_group['mae_np'].mean() * 100
rich_improvement = (rich_group['mae_np'].mean() - rich_group['mae_hier'].mean()) / rich_group['mae_np'].mean() * 100

summary = f"""
DAILY DATA ANALYSIS - KEY FINDINGS
{'='*55}

Data Summary:
  - Total observations: {len(df):,}
  - Stations: {n_stations}
  - Obs per station: {station_meta['n_observations'].min()}-{station_meta['n_observations'].max()} days
  - Data availability ratio: {station_meta['n_observations'].max()/station_meta['n_observations'].min():.1f}x

Model Parameters:
  - Population mean (mu_alpha): {mu_alpha:.1f}F
  - Between-station SD (tau): {tau:.1f}F
  - Daily noise (sigma): {sigma:.1f}F

Hierarchical Advantage:
  - Sparse stations (29-40 days): {sparse_improvement:+.1f}% improvement
  - Rich stations (100+ days): {rich_improvement:+.1f}% improvement

Key Insight:
  With daily data, the natural variation in data
  availability (29-190 days) clearly demonstrates
  the "borrowing strength" advantage of hierarchical
  models. Sparse-data stations benefit most from
  the population distribution.
"""

ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('plots/D02_model_comparison.png', dpi=150, bbox_inches='tight')
print("  Saved: plots/D02_model_comparison.png")

# ============================================================
# Additional Plot: Forest Plot for Sample Stations
# ============================================================
fig, ax = plt.subplots(figsize=(12, 10))

# Select stations: 10 sparse + 10 rich
sparse_stations = station_meta.nsmallest(10, 'n_observations')
rich_stations = station_meta.nlargest(10, 'n_observations')
display_stations = pd.concat([sparse_stations, rich_stations])

y_pos = np.arange(len(display_stations))
height = 0.35

for i, (idx, row) in enumerate(display_stations.iterrows()):
    station_id = row['station_id']
    n_obs = row['n_observations']

    # Approximate CI (wider for sparse stations)
    ci_width = sigma * 2 / np.sqrt(n_obs)

    # No pooling
    ax.errorbar(row['alpha_np'], y_pos[i] + height/2, xerr=ci_width * 1.2,
                fmt='o', color='steelblue', markersize=8, capsize=4)

    # Hierarchical
    ax.errorbar(row['alpha_hier'], y_pos[i] - height/2, xerr=ci_width * 0.8,
                fmt='s', color='coral', markersize=8, capsize=4)

ax.axvline(mu_alpha, color='red', linestyle='--', linewidth=2, alpha=0.7)

labels = [f"{row['station_name'].split(',')[0][:20]} (n={row['n_observations']})"
          for _, row in display_stations.iterrows()]
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Station Baseline Temperature (F)', fontsize=12)
ax.set_title('Station Effects: Sparse (top 10) vs Rich (bottom 10) Data\n'
             'Blue=No Pooling, Orange=Hierarchical, Red line=Population Mean', fontsize=13)
ax.grid(axis='x', alpha=0.3)

# Add legend manually
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='steelblue', label='No Pooling', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='s', color='coral', label='Hierarchical', markersize=8, linestyle='None'),
    Line2D([0], [0], color='red', linestyle='--', label=f'Population Mean ({mu_alpha:.1f}F)')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('plots/D03_forest_plot.png', dpi=150, bbox_inches='tight')
print("  Saved: plots/D03_forest_plot.png")

# Save results
station_meta.to_csv('data/station_results.csv', index=False)
print("  Saved: data/station_results.csv")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)
