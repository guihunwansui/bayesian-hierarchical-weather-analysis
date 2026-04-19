"""
DATASCI 451 - Daily Data Preparation
=====================================
Prepare daily temperature data for hierarchical modeling.

Key difference from monthly analysis:
- Uses individual daily observations instead of monthly aggregates
- Preserves natural variation in data availability (29-190 days per station)
- 14,569 observations vs 643 in monthly data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("="*70)
print("DAILY DATA PREPARATION")
print("="*70)

# Load the cleaned daily data
df = pd.read_csv('../data/cleaned_daily_data.csv')
print(f"\nRaw data: {len(df)} rows, {df['NAME'].nunique()} stations")

# Use TAVG_computed (from TMAX/TMIN average) if TAVG is missing
df['TEMP'] = df['TAVG_computed'].fillna(df['TAVG'])

# Keep only rows with valid temperature
df_valid = df[df['TEMP'].notna()].copy()
print(f"Valid temperature observations: {len(df_valid)}")

# Create station index
station_names = df_valid['NAME'].unique()
station_idx_map = {name: i for i, name in enumerate(station_names)}
df_valid['station_id'] = df_valid['NAME'].map(station_idx_map)

# Create month index (0-based)
df_valid['month_id'] = df_valid['Month'] - 1

# Summary statistics
n_stations = len(station_names)
n_obs = len(df_valid)
obs_per_station = df_valid.groupby('NAME').size()

print(f"\n" + "-"*50)
print("DATA SUMMARY")
print("-"*50)
print(f"Total observations: {n_obs}")
print(f"Total stations: {n_stations}")
print(f"Observations per station: {obs_per_station.min()} - {obs_per_station.max()}")
print(f"  Mean: {obs_per_station.mean():.1f}")
print(f"  Std:  {obs_per_station.std():.1f}")

# Distribution of observations per station
print(f"\nObservation count distribution:")
bins = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 200)]
for low, high in bins:
    count = ((obs_per_station >= low) & (obs_per_station < high)).sum()
    print(f"  {low:3d}-{high:3d} days: {count:3d} stations")

# Save prepared data
df_valid.to_csv('data/daily_data_prepared.csv', index=False)
print(f"\nSaved: data/daily_data_prepared.csv")

# Save station metadata
station_meta = pd.DataFrame({
    'station_id': range(n_stations),
    'station_name': station_names,
    'n_observations': [obs_per_station[name] for name in station_names]
})
station_meta.to_csv('data/station_metadata.csv', index=False)
print(f"Saved: data/station_metadata.csv")

# ============================================================
# Visualization: Data Availability
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Distribution of observations per station
ax1 = axes[0]
ax1.hist(obs_per_station.values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(obs_per_station.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {obs_per_station.mean():.0f} days')
ax1.axvline(obs_per_station.median(), color='orange', linestyle='--', linewidth=2,
            label=f'Median: {obs_per_station.median():.0f} days')
ax1.set_xlabel('Number of Daily Observations', fontsize=12)
ax1.set_ylabel('Number of Stations', fontsize=12)
ax1.set_title('Distribution of Data Availability\n(Key for Hierarchical Advantage)', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Top and bottom stations by data availability
ax2 = axes[1]
sorted_obs = obs_per_station.sort_values()
bottom_10 = sorted_obs.head(10)
top_10 = sorted_obs.tail(10)
display_stations = pd.concat([bottom_10, top_10])

colors = ['coral'] * 10 + ['steelblue'] * 10
y_pos = np.arange(len(display_stations))
ax2.barh(y_pos, display_stations.values, color=colors, edgecolor='black')
ax2.set_yticks(y_pos)
ax2.set_yticklabels([name.split(',')[0][:20] for name in display_stations.index], fontsize=8)
ax2.set_xlabel('Number of Daily Observations', fontsize=12)
ax2.set_title('Stations with Least (coral) vs Most (blue) Data\n(6.5x difference in data availability)', fontsize=13)
ax2.axvline(obs_per_station.mean(), color='red', linestyle='--', alpha=0.7)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/D01_data_availability.png', dpi=150, bbox_inches='tight')
print(f"Saved: plots/D01_data_availability.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("COMPARISON: MONTHLY vs DAILY DATA")
print("="*70)
print(f"""
                        Monthly Data    Daily Data
                        ------------    ----------
Total observations:     643             {n_obs}
Stations:               167             {n_stations}
Obs per station range:  1-4             {obs_per_station.min()}-{obs_per_station.max()}
Obs per station std:    0.8             {obs_per_station.std():.1f}

Key advantage of daily data:
  - {obs_per_station.max()/obs_per_station.min():.1f}x difference in data availability
  - This natural sparsity is ideal for demonstrating hierarchical advantages
  - Stations with few observations will "borrow strength" from data-rich stations
""")

print("="*70)
print("Data preparation complete!")
print("="*70)
