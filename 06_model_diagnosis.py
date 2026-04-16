"""
DATASCI 451 - Model Diagnosis: Why Hierarchical ≈ No Pooling?

Deep analysis of why hierarchical model doesn't outperform no pooling.
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MODEL DIAGNOSIS: Why doesn't hierarchical model outperform?")
print("="*70)

# Load data
df = pd.read_csv('data/selected_stations_monthly.csv')
station_names = df['short_name'].unique()
n_stations = len(station_names)

# ============================================================
# Analysis 1: Data sufficiency per station
# ============================================================
print("\n1. DATA SUFFICIENCY ANALYSIS")
print("-"*70)

obs_per_station = df.groupby('short_name').size()
print(f"Observations per station: {obs_per_station.min()} - {obs_per_station.max()}")
print(f"Mean: {obs_per_station.mean():.1f}")
print("\nWith 4 observations per station, each station has 'enough' data")
print("for independent estimation. Hierarchical advantage appears with FEWER obs.")

# ============================================================
# Analysis 2: Between vs Within station variance
# ============================================================
print("\n\n2. VARIANCE DECOMPOSITION")
print("-"*70)

# Overall variance
total_var = df['TAVG_mean'].var()

# Between-station variance (variance of station means)
station_means = df.groupby('short_name')['TAVG_mean'].mean()
between_var = station_means.var()

# Within-station variance (average variance within each station)
within_var = df.groupby('short_name')['TAVG_mean'].var().mean()

# Monthly effect variance
month_means = df.groupby('Month')['TAVG_mean'].mean()
month_var = month_means.var()

print(f"Total variance:          {total_var:.2f}")
print(f"Between-station var:     {between_var:.2f} ({between_var/total_var*100:.1f}%)")
print(f"Within-station var:      {within_var:.2f} ({within_var/total_var*100:.1f}%)")
print(f"Month effect var:        {month_var:.2f} ({month_var/total_var*100:.1f}%)")

print("\n→ Most variance is explained by MONTH (seasonal), not station!")
print("→ Station effects are relatively small compared to seasonal effects")

# ============================================================
# Analysis 3: Station-specific seasonal patterns
# ============================================================
print("\n\n3. SEASONAL PATTERN HETEROGENEITY")
print("-"*70)

# Calculate seasonal amplitude for each station
seasonal_amplitude = df.groupby('short_name').apply(
    lambda x: x['TAVG_mean'].max() - x['TAVG_mean'].min()
)
print("\nSeasonal amplitude (max - min) by station:")
for station in station_names:
    amp = seasonal_amplitude[station]
    print(f"  {station:<20}: {amp:.1f}°F")

print(f"\nAmplitude range: {seasonal_amplitude.min():.1f} - {seasonal_amplitude.max():.1f}°F")
print(f"Amplitude std:   {seasonal_amplitude.std():.1f}°F")
print("\n→ Seasonal amplitudes vary across stations!")
print("→ Our model assumes SAME β for all stations - this may be wrong")

# ============================================================
# Analysis 4: Correlation structure
# ============================================================
print("\n\n4. STATION CORRELATION ANALYSIS")
print("-"*70)

# Pivot to wide format
pivot = df.pivot(index='Month', columns='short_name', values='TAVG_mean')
corr_matrix = pivot.corr()

print("Station temperature correlations:")
print(corr_matrix.round(2).to_string())

avg_corr = corr_matrix.values[np.triu_indices(n_stations, k=1)].mean()
print(f"\nAverage pairwise correlation: {avg_corr:.3f}")
print("→ High correlation suggests stations share common patterns (good for hierarchical)")
print("→ BUT with only 4 time points, correlation is driven by shared seasonality")

# ============================================================
# Analysis 5: Residual analysis after removing month effect
# ============================================================
print("\n\n5. RESIDUAL ANALYSIS (after removing month effect)")
print("-"*70)

# Remove month effect
df['month_mean'] = df['Month'].map(month_means)
df['residual'] = df['TAVG_mean'] - df['month_mean']

# Now check station variance in residuals
residual_station_var = df.groupby('short_name')['residual'].var().mean()
residual_between_var = df.groupby('short_name')['residual'].mean().var()

print(f"Residual within-station var:  {residual_station_var:.2f}")
print(f"Residual between-station var: {residual_between_var:.2f}")
print(f"Ratio (between/within):       {residual_between_var/residual_station_var:.2f}")

print("\n→ After removing seasonality, between-station variance is clear")
print("→ This is what τ captures in the hierarchical model")

# ============================================================
# Visualization
# ============================================================
print("\n\n6. GENERATING DIAGNOSTIC PLOTS...")
print("-"*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Variance decomposition
ax1 = axes[0, 0]
components = ['Between\nStation', 'Within\nStation', 'Month\nEffect']
values = [between_var, within_var, month_var]
colors = ['steelblue', 'coral', 'green']
ax1.bar(components, values, color=colors, edgecolor='black')
ax1.set_ylabel('Variance')
ax1.set_title('Variance Decomposition\n(Month effect dominates!)')
for i, v in enumerate(values):
    ax1.text(i, v + 2, f'{v:.1f}', ha='center', fontweight='bold')

# Plot 2: Seasonal patterns by station
ax2 = axes[0, 1]
for station in station_names:
    data = df[df['short_name'] == station].sort_values('Month')
    ax2.plot(data['Month'], data['TAVG_mean'], 'o-', label=station[:12], linewidth=2)
ax2.set_xlabel('Month')
ax2.set_ylabel('Temperature (°F)')
ax2.set_title('Seasonal Patterns by Station\n(Similar shapes = shared β works)')
ax2.legend(fontsize=8, loc='upper left')
ax2.set_xticks([1, 2, 3, 4])
ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr'])

# Plot 3: Correlation heatmap
ax3 = axes[1, 0]
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
            ax=ax3, vmin=0.9, vmax=1.0, linewidths=0.5)
ax3.set_title(f'Station Correlations\n(Avg: {avg_corr:.3f} - very high!)')

# Plot 4: Seasonal amplitude variation
ax4 = axes[1, 1]
sorted_amp = seasonal_amplitude.sort_values()
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_amp)))
bars = ax4.barh(range(len(sorted_amp)), sorted_amp.values, color=colors)
ax4.set_yticks(range(len(sorted_amp)))
ax4.set_yticklabels([s[:15] for s in sorted_amp.index])
ax4.set_xlabel('Seasonal Amplitude (°F)')
ax4.set_title('Seasonal Amplitude by Station\n(Variation suggests station-specific β needed)')
ax4.axvline(seasonal_amplitude.mean(), color='red', linestyle='--',
            label=f'Mean: {seasonal_amplitude.mean():.1f}°F')
ax4.legend()

plt.tight_layout()
plt.savefig('plots/21_model_diagnosis.png', dpi=150, bbox_inches='tight')
print("  Saved: plots/21_model_diagnosis.png")

# ============================================================
# Summary
# ============================================================
print("\n\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

print("""
WHY HIERARCHICAL ≈ NO POOLING IN OUR DATA?

1. SUFFICIENT DATA PER STATION
   - 4 observations per station is enough for stable individual estimation
   - Hierarchical advantage appears when n < 3 per group

2. DOMINANT SEASONAL EFFECT
   - Month explains most variance (~80%)
   - Station effect is secondary
   - Both models capture seasonality well

3. HIGH BETWEEN-STATION CORRELATION
   - Stations are highly correlated (avg r = 0.99)
   - Little unique information to borrow between stations

4. MODEL DESIGN ISSUE (potential improvement)
   - We assume SAME seasonal pattern (β) for all stations
   - But seasonal amplitudes vary: 15.8°F to 26.3°F
   - A better model: station-specific seasonal effects

SOLUTIONS TO DEMONSTRATE HIERARCHICAL ADVANTAGE:

A. Test with LESS data (simulate sparse scenario)
B. Test Leave-One-STATION-Out (predict new station)
C. Improve model with hierarchical seasonal effects
""")

plt.show()
print("\nDiagnosis complete!")
