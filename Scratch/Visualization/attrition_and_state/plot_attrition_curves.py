import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.style.use('dark_background')

print("Loading dataset for trajectory visualization...")
df = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv")

# Filter for ambitious cases in ANY dimension (Height, FAR, Coverage, or SqFt)
df_ambitious = df[
    (df['Requested_max_height_ft'] > df['Initial_max_height_ft']) |
    (df['Requested_max_far'] > df['Initial_max_far']) |
    (df['Requested_max_bldg_cov_pct'] > df['Initial_max_bldg_cov_pct']) |
    (df['Phase_Requested_SqFt'] > (df['Initial_max_far'] * df['shape_area']))
].copy()

# Ensure we have Days in Pipeline
df_plot = df_ambitious.dropna(subset=['label_real_days_in_pipeline']).copy()
df_plot = df_plot[(df_plot['label_real_days_in_pipeline'] > 30) & (df_plot['label_real_days_in_pipeline'] < 1500)]

print(f"Plotting temporal trajectories for {len(df_plot)} ambitious cases using opacity blending...")

fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
ax1, ax2 = axs[0]
ax3, ax4 = axs[1]

p90_x = df_plot['label_real_days_in_pipeline'].quantile(0.90)

# Panel 1: Height
df_h = df_plot.dropna(subset=['Requested_max_height_ft', 'Effective_Approved_Height']).copy()
p90_h = df_h['Requested_max_height_ft'].quantile(0.90)
for idx, row in df_h.iterrows():
    comp_cap = row['GIS_Compatibility_Height_Cap']
    eff_req = min(row['Requested_max_height_ft'], comp_cap) if pd.notna(comp_cap) else row['Requested_max_height_ft']
    y = [eff_req, row['Effective_Approved_Height']]
    x = [0, row['label_real_days_in_pipeline']]
    ax1.plot(x, y, color='cyan', alpha=0.1, linewidth=1.5)

ax1.set_xlim(0, p90_x + 30)
ax1.set_ylim(0, p90_h + 10)
ax1.set_xlabel('Days in Pipeline', fontsize=12, fontweight='bold', color='white')
ax1.set_ylabel('Effective Height (ft)', fontsize=12, fontweight='bold', color='white')
ax1.set_title('Height Attrition', fontsize=16, fontweight='bold', color='white')
ax1.grid(axis='both', color='#333333', linestyle='--', linewidth=0.5)
ax1.legend(handles=[Line2D([0],[0], color='cyan', lw=3, label=f'Height (n={len(df_h)})')], loc='upper right', facecolor='black')

# Panel 2: FAR
df_f = df_plot.dropna(subset=['Requested_max_far', 'Approved_max_far']).copy()
p90_f = df_f['Requested_max_far'].quantile(0.90)
for idx, row in df_f.iterrows():
    y = [row['Requested_max_far'], row['Approved_max_far']]
    x = [0, row['label_real_days_in_pipeline']]
    ax2.plot(x, y, color='cyan', alpha=0.1, linewidth=1.5)

ax2.set_xlim(0, p90_x + 30)
ax2.set_ylim(0, p90_f + 0.5)
ax2.set_xlabel('Days in Pipeline', fontsize=12, fontweight='bold', color='white')
ax2.set_ylabel('Max FAR (Ratio)', fontsize=12, fontweight='bold', color='white')
ax2.set_title('FAR Attrition', fontsize=16, fontweight='bold', color='white')
ax2.grid(axis='both', color='#333333', linestyle='--', linewidth=0.5)
ax2.legend(handles=[Line2D([0],[0], color='cyan', lw=3, label=f'FAR (n={len(df_f)})')], loc='upper right', facecolor='black')

# Panel 3: Building Coverage
df_c = df_plot.dropna(subset=['Requested_max_bldg_cov_pct', 'Approved_max_bldg_cov_pct']).copy()
for idx, row in df_c.iterrows():
    y = [row['Requested_max_bldg_cov_pct'], row['Approved_max_bldg_cov_pct']]
    x = [0, row['label_real_days_in_pipeline']]
    ax3.plot(x, y, color='cyan', alpha=0.1, linewidth=1.5)

ax3.set_xlim(0, p90_x + 30)
ax3.set_ylim(0, 100) # Coverage is max 100%
ax3.set_xlabel('Days in Pipeline', fontsize=12, fontweight='bold', color='white')
ax3.set_ylabel('Building Coverage (%)', fontsize=12, fontweight='bold', color='white')
ax3.set_title('Building Coverage Attrition', fontsize=16, fontweight='bold', color='white')
ax3.grid(axis='both', color='#333333', linestyle='--', linewidth=0.5)
ax3.legend(handles=[Line2D([0],[0], color='cyan', lw=3, label=f'Coverage (n={len(df_c)})')], loc='upper right', facecolor='black')

# Panel 4: Square Footage (Actual Parcel Volume)
df_s = df_plot.dropna(subset=['Phase_Requested_SqFt', 'Phase_Approved_SqFt']).copy()
for idx, row in df_s.iterrows():
    y = [row['Phase_Requested_SqFt'], row['Phase_Approved_SqFt']]
    x = [0, row['label_real_days_in_pipeline']]
    ax4.plot(x, y, color='cyan', alpha=0.1, linewidth=1.5)

p90_s = df_s['Phase_Requested_SqFt'].quantile(0.90)
ax4.set_xlim(0, p90_x + 30)
ax4.set_ylim(0, p90_s * 1.05)
ax4.set_xlabel('Days in Pipeline', fontsize=12, fontweight='bold', color='white')
ax4.set_ylabel('Buildable Square Footage', fontsize=12, fontweight='bold', color='white')
ax4.set_title('Square Footage Attrition (Real Parcel Volume)', fontsize=16, fontweight='bold', color='white')
ax4.grid(axis='both', color='#333333', linestyle='--', linewidth=0.5)
ax4.legend(handles=[Line2D([0],[0], color='cyan', lw=3, label=f'Actual SqFt (n={len(df_s)})')], loc='upper right', facecolor='black')

import matplotlib.ticker as ticker
ax4.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

fig.suptitle('Multi-Dimensional Temporal Attrition in the Zoning Pipeline', fontsize=22, fontweight='bold', color='white', y=0.95)

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
output_path = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\multi_temporal_attrition_2x2_fixed.png"
plt.savefig(output_path, facecolor='black', edgecolor='none', bbox_inches='tight')
print(f"Plot saved successfully to {output_path}")
