"""
DATASCI 451 - Final Summary: Monthly vs Daily Data Analysis
============================================================
Key finding: Why hierarchical advantage differs by data granularity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

print("="*70)
print("FINAL COMPARISON: MONTHLY vs DAILY DATA ANALYSIS")
print("="*70)

# Summary data
comparison = {
    'Metric': [
        'Total Observations',
        'Stations',
        'Obs per Station (range)',
        'Obs per Station (std)',
        'Data Availability Ratio',
        '',
        'tau (between-station SD)',
        'sigma (observation noise)',
        'tau/sigma ratio',
        '',
        'Hierarchical Improvement',
        '  - Sparse stations',
        '  - Rich stations'
    ],
    'Monthly Data': [
        '643',
        '167',
        '1-4',
        '0.8',
        '4x',
        '',
        '4.42 F',
        '2.84 F',
        '1.56',
        '',
        '',
        '+52% (1 obs)',
        '+44% (4 obs)'
    ],
    'Daily Data': [
        '14,569',
        '167',
        '29-190',
        '18.7',
        '6.6x',
        '',
        '3.87 F',
        '10.75 F',
        '0.36',
        '',
        '',
        '+0.2% (<30 obs)',
        '+0.1% (90+ obs)'
    ]
}

df_comp = pd.DataFrame(comparison)
print("\n" + df_comp.to_string(index=False))

print("\n" + "="*70)
print("KEY INSIGHT")
print("="*70)

print("""
WHY HIERARCHICAL ADVANTAGE DIFFERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The tau/sigma ratio determines hierarchical model advantage:

  tau/sigma = (between-group variance) / (within-group noise)

Monthly Data: tau/sigma = 1.56
  → Station differences are 1.5x the noise level
  → Clear signal, strong shrinkage benefit

Daily Data: tau/sigma = 0.36
  → Station differences are only 0.36x the noise level
  → Signal masked by day-to-day weather variability
  → Shrinkage benefit diminished

PARADOX RESOLVED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: Why doesn't more data (daily) show more hierarchical advantage?

A: Aggregation (monthly means) REDUCES noise (sigma) by averaging out
   day-to-day variability, making the station-level signal (tau)
   more apparent relative to noise.

   More data ≠ More hierarchical advantage
   Better signal-to-noise ratio = More hierarchical advantage

IMPLICATIONS FOR PRACTICE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Consider the tau/sigma ratio when choosing hierarchical models
2. Data aggregation can IMPROVE hierarchical advantage by reducing noise
3. The key is group-level signal, not just data quantity
4. Hierarchical models always provide:
   - New group prediction capability
   - Principled uncertainty quantification
   - Interpretable population parameters
""")

# ============================================================
# Create comparison visualization
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: tau/sigma comparison
ax1 = axes[0]
x = [0, 1]
tau_values = [4.42, 3.87]
sigma_values = [2.84, 10.75]
width = 0.35

bars1 = ax1.bar([i - width/2 for i in x], tau_values, width, label='tau (between-station SD)',
                color='coral', edgecolor='black')
bars2 = ax1.bar([i + width/2 for i in x], sigma_values, width, label='sigma (observation noise)',
                color='steelblue', edgecolor='black')

ax1.set_xticks(x)
ax1.set_xticklabels(['Monthly Data', 'Daily Data'])
ax1.set_ylabel('Standard Deviation (F)', fontsize=12)
ax1.set_title('Signal vs Noise\n(tau/sigma ratio determines hierarchical advantage)', fontsize=12)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add ratio annotations
ax1.annotate(f'tau/sigma = 1.56', (0, max(tau_values[0], sigma_values[0]) + 1),
            ha='center', fontsize=11, fontweight='bold', color='green')
ax1.annotate(f'tau/sigma = 0.36', (1, max(tau_values[1], sigma_values[1]) + 1),
            ha='center', fontsize=11, fontweight='bold', color='red')

# Plot 2: Hierarchical improvement comparison
ax2 = axes[1]
categories = ['Sparse\nStations', 'Rich\nStations']
monthly_improvement = [52, 44]
daily_improvement = [0.2, 0.1]

x = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x - width/2, monthly_improvement, width, label='Monthly Data',
                color='coral', edgecolor='black')
bars2 = ax2.bar(x + width/2, daily_improvement, width, label='Daily Data',
                color='steelblue', edgecolor='black')

ax2.set_ylabel('Hierarchical Improvement (%)', fontsize=12)
ax2.set_title('Hierarchical Model Advantage\n(Higher = Better)', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars1, monthly_improvement):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'+{val}%',
            ha='center', fontsize=11, fontweight='bold')
for bar, val in zip(bars2, daily_improvement):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'+{val}%',
            ha='center', fontsize=11, fontweight='bold')

# Plot 3: Key insight summary
ax3 = axes[2]
ax3.axis('off')

insight_text = """
KEY TAKEAWAY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hierarchical advantage depends on:

  tau/sigma ratio

  ↑ Higher ratio = ↑ More advantage

Monthly data: tau/sigma = 1.56 ✓
  - Aggregation reduces noise
  - Station signal is clear
  - Strong shrinkage benefit

Daily data: tau/sigma = 0.36 ✗
  - High day-to-day variability
  - Station signal is masked
  - Minimal shrinkage benefit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Lesson: Data granularity matters!
More data ≠ More hierarchical advantage
Better signal-to-noise = More advantage
"""

ax3.text(0.05, 0.95, insight_text, transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('plots/D05_monthly_vs_daily_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: plots/D05_monthly_vs_daily_comparison.png")

print("\n" + "="*70)
print("Summary comparison complete!")
print("="*70)
