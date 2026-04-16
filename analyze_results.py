"""
DATASCI 451 - Analyze Model Results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import warnings
warnings.filterwarnings('ignore')

# Load traces
trace_cp = az.from_netcdf('data/trace_complete_pooling.nc')
trace_np = az.from_netcdf('data/trace_no_pooling.nc')
trace_hier = az.from_netcdf('data/trace_hierarchical.nc')

# Load data
df = pd.read_csv('data/selected_stations_monthly.csv')
station_names = df['short_name'].unique()

print("="*70)
print("MODEL RESULTS ANALYSIS")
print("="*70)

# ============================================================
# Model Comparison (WAIC)
# ============================================================
comparison = az.compare({
    'Complete Pooling': trace_cp,
    'No Pooling': trace_np,
    'Hierarchical': trace_hier
}, ic='waic')

print("\n1. MODEL COMPARISON (WAIC - lower is better)")
print("-"*70)
print(comparison.to_string())

# ============================================================
# Hierarchical Model Key Parameters
# ============================================================
print("\n\n2. HIERARCHICAL MODEL PARAMETERS")
print("-"*70)

mu_alpha = float(trace_hier.posterior['mu_alpha'].mean())
tau = float(trace_hier.posterior['tau'].mean())
sigma = float(trace_hier.posterior['sigma'].mean())
alpha_hier = trace_hier.posterior['alpha'].mean(dim=['chain', 'draw']).values
beta_hier = trace_hier.posterior['beta'].mean(dim=['chain', 'draw']).values

print(f"\nPopulation Parameters:")
print(f"  μ_α (population mean):     {mu_alpha:.2f}°F")
print(f"  τ  (between-station SD):   {tau:.2f}°F")
print(f"  σ  (observation noise):    {sigma:.2f}°F")
print(f"  τ/σ ratio:                 {tau/sigma:.2f}")
print(f"  → Interpretation: Station variation is {tau/sigma:.1f}x the observation noise")

print(f"\nStation Effects (sorted by temperature):")
sorted_idx = np.argsort(alpha_hier)
for idx in sorted_idx:
    print(f"  {station_names[idx]:<20}: {alpha_hier[idx]:.2f}°F")

month_names = ['January', 'February', 'March', 'April']
print(f"\nMonth Effects (relative to mean):")
for i, name in enumerate(month_names):
    print(f"  {name:<10}: {beta_hier[i]:+.2f}°F")

# ============================================================
# Shrinkage Analysis
# ============================================================
print("\n\n3. SHRINKAGE ANALYSIS")
print("-"*70)

alpha_np = trace_np.posterior['alpha'].mean(dim=['chain', 'draw']).values

print(f"\n{'Station':<20} {'No Pool':>10} {'Hierarchical':>12} {'Shrinkage':>12} {'Toward Mean':>12}")
print("-"*70)
total_shrinkage = 0
for i, name in enumerate(station_names):
    shrink = alpha_np[i] - alpha_hier[i]
    toward_mean = "Yes" if (alpha_np[i] > mu_alpha and shrink > 0) or (alpha_np[i] < mu_alpha and shrink < 0) else "No"
    print(f"{name:<20} {alpha_np[i]:>10.2f} {alpha_hier[i]:>12.2f} {shrink:>+12.2f} {toward_mean:>12}")
    total_shrinkage += abs(shrink)

print(f"\nTotal shrinkage: {total_shrinkage:.2f}°F")
print(f"Average shrinkage per station: {total_shrinkage/len(station_names):.2f}°F")

# ============================================================
# Credible Intervals Comparison
# ============================================================
print("\n\n4. CREDIBLE INTERVAL WIDTH COMPARISON")
print("-"*70)

print(f"\n{'Station':<20} {'No Pool 95% CI':>15} {'Hier 95% CI':>15} {'CI Reduction':>12}")
print("-"*70)

ci_reductions = []
for i, name in enumerate(station_names):
    # No pooling CI
    np_samples = trace_np.posterior['alpha'].sel(alpha_dim_0=i).values.flatten()
    np_lo, np_hi = np.percentile(np_samples, [2.5, 97.5])
    np_width = np_hi - np_lo

    # Hierarchical CI
    hier_samples = trace_hier.posterior['alpha'].sel(alpha_dim_0=i).values.flatten()
    hier_lo, hier_hi = np.percentile(hier_samples, [2.5, 97.5])
    hier_width = hier_hi - hier_lo

    reduction = (np_width - hier_width) / np_width * 100
    ci_reductions.append(reduction)

    print(f"{name:<20} {np_width:>15.2f} {hier_width:>15.2f} {reduction:>+11.1f}%")

print(f"\nAverage CI change: {np.mean(ci_reductions):+.1f}%")

# ============================================================
# Generate Visualizations
# ============================================================
print("\n\n5. GENERATING VISUALIZATIONS...")
print("-"*70)

# Figure 1: Shrinkage Plot
fig, ax = plt.subplots(figsize=(10, 8))

lims = [min(alpha_np.min(), alpha_hier.min()) - 3, max(alpha_np.max(), alpha_hier.max()) + 3]
ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='No shrinkage line')
ax.axhline(mu_alpha, color='red', linestyle=':', alpha=0.7, linewidth=2,
           label=f'Population mean: {mu_alpha:.1f}°F')
ax.axvline(mu_alpha, color='red', linestyle=':', alpha=0.3, linewidth=1)

colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(station_names)))
for i, name in enumerate(station_names):
    ax.scatter(alpha_np[i], alpha_hier[i], s=200, c=[colors[i]],
               edgecolors='black', linewidths=2, zorder=5)
    offset = (8, 5) if i % 2 == 0 else (8, -12)
    ax.annotate(name, (alpha_np[i], alpha_hier[i]),
                xytext=offset, textcoords='offset points', fontsize=10, fontweight='bold')

ax.set_xlabel('No Pooling Estimate (°F)', fontsize=13)
ax.set_ylabel('Hierarchical (Partial Pooling) Estimate (°F)', fontsize=13)
ax.set_title('Shrinkage Effect in Hierarchical Model\n'
             'Points pulled toward population mean (red dashed line)', fontsize=14)
ax.legend(loc='upper left', fontsize=11)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/12_shrinkage_effect.png', dpi=150, bbox_inches='tight')
print("  Saved: plots/12_shrinkage_effect.png")

# Figure 2: Forest Plot
fig, ax = plt.subplots(figsize=(12, 8))

y_pos = np.arange(len(station_names))
height = 0.35

for i, name in enumerate(station_names):
    # No pooling
    np_samples = trace_np.posterior['alpha'].sel(alpha_dim_0=i).values.flatten()
    np_mean = np_samples.mean()
    np_lo, np_hi = np.percentile(np_samples, [2.5, 97.5])
    ax.errorbar(np_mean, y_pos[i] + height/2,
                xerr=[[np_mean - np_lo], [np_hi - np_mean]],
                fmt='o', color='steelblue', markersize=10, capsize=5, capthick=2,
                label='No Pooling' if i == 0 else '')

    # Hierarchical
    hier_samples = trace_hier.posterior['alpha'].sel(alpha_dim_0=i).values.flatten()
    hier_mean = hier_samples.mean()
    hier_lo, hier_hi = np.percentile(hier_samples, [2.5, 97.5])
    ax.errorbar(hier_mean, y_pos[i] - height/2,
                xerr=[[hier_mean - hier_lo], [hier_hi - hier_mean]],
                fmt='s', color='coral', markersize=10, capsize=5, capthick=2,
                label='Hierarchical' if i == 0 else '')

ax.axvline(mu_alpha, color='red', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Population mean: {mu_alpha:.1f}°F')
ax.set_yticks(y_pos)
ax.set_yticklabels(station_names, fontsize=11)
ax.set_xlabel('Station Effect α (°F)', fontsize=13)
ax.set_title('Station Effects: No Pooling vs Hierarchical\n(95% Credible Intervals)', fontsize=14)
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/13_forest_plot.png', dpi=150, bbox_inches='tight')
print("  Saved: plots/13_forest_plot.png")

# Figure 3: Month Effects
fig, ax = plt.subplots(figsize=(10, 6))

beta_samples = trace_hier.posterior['beta'].values.reshape(-1, 4)
x = np.arange(4)
colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']

for i in range(4):
    mean = beta_samples[:, i].mean()
    lo, hi = np.percentile(beta_samples[:, i], [2.5, 97.5])
    ax.bar(x[i], mean, color=colors[i], alpha=0.7, edgecolor='black', linewidth=2)
    ax.errorbar(x[i], mean, yerr=[[mean - lo], [hi - mean]],
                fmt='none', color='black', capsize=8, capthick=2)
    ax.annotate(f'{mean:+.1f}°F', (x[i], mean),
                xytext=(0, 15 if mean > 0 else -20), textcoords='offset points',
                ha='center', fontsize=12, fontweight='bold')

ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(month_names, fontsize=12)
ax.set_ylabel('Month Effect β (°F)', fontsize=13)
ax.set_title('Seasonal Temperature Effects\n(Relative to Population Mean, 95% CI)', fontsize=14)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/14_month_effects.png', dpi=150, bbox_inches='tight')
print("  Saved: plots/14_month_effects.png")

# Figure 4: Model Comparison Bar
fig, ax = plt.subplots(figsize=(10, 5))

model_names = comparison.index.tolist()
waic_values = comparison['elpd_waic'].values

colors = ['coral' if 'Hierarchical' in name else 'steelblue' for name in model_names]
bars = ax.barh(model_names, waic_values, color=colors, edgecolor='black', linewidth=1.5)

ax.set_xlabel('ELPD (higher is better)', fontsize=13)
ax.set_title('Model Comparison: Expected Log Pointwise Predictive Density', fontsize=14)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

for bar, val in zip(bars, waic_values):
    ax.text(val - 2, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
            va='center', ha='right', fontsize=11, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('plots/15_model_comparison.png', dpi=150, bbox_inches='tight')
print("  Saved: plots/15_model_comparison.png")

# ============================================================
# Summary
# ============================================================
print("\n\n" + "="*70)
print("SUMMARY OF KEY FINDINGS")
print("="*70)

best_model = comparison.index[0]
print(f"""
1. MODEL COMPARISON
   Best model: {best_model}
   The hierarchical model balances fit and complexity.

2. STATION HETEROGENEITY
   - Between-station SD (τ): {tau:.2f}°F
   - Within-station noise (σ): {sigma:.2f}°F
   - τ/σ = {tau/sigma:.2f} → Stations differ by ~{tau/sigma:.1f}x the noise level
   - This justifies using a hierarchical model!

3. SHRINKAGE EFFECT
   - Average shrinkage: {total_shrinkage/len(station_names):.2f}°F toward population mean
   - Extreme stations (Bergland Dam, Ann Arbor) show most shrinkage
   - CI width change: {np.mean(ci_reductions):+.1f}% (negative = wider, but more regularized)

4. SEASONAL PATTERN
   - Temperature range: {beta_hier.min():.1f}°F (Jan) to {beta_hier.max():.1f}°F (Apr)
   - Total seasonal swing: {beta_hier.max() - beta_hier.min():.1f}°F

5. GEOGRAPHIC GRADIENT
   - Coldest: {station_names[np.argmin(alpha_hier)]} ({alpha_hier.min():.1f}°F)
   - Warmest: {station_names[np.argmax(alpha_hier)]} ({alpha_hier.max():.1f}°F)
   - North-south range: {alpha_hier.max() - alpha_hier.min():.1f}°F
""")

print("="*70)
print("Analysis complete! Check plots/ directory for visualizations.")
print("="*70)
