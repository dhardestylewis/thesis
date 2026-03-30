import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import os

print("[*] Generating Stage A Exhibits (Figures 3 & 4)...")

# 1. Load probabilities
probs = pd.read_csv('Analysis/Output/Track0_Predictive/stage_a_hazard_results.csv')

# 2. Extract Latitude/Longitude and events efficiently
print("    Loading spatial coordinates and calculating events...")
panel_geo = pd.read_csv('Data/Panel/Output/Property_Year_Panel_v3.csv', usecols=['standardized_tcad_id', 'year', 'latitude', 'longitude'])
case_tcad = pd.read_csv('Data/Warehouse_As_Of/H0_Filing_Master_Enriched.csv', usecols=['standardized_tcad_id', 'year'], low_memory=False)

# Clean IDs
case_tcad['standardized_tcad_id'] = case_tcad['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
panel_geo['standardized_tcad_id'] = panel_geo['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
probs['standardized_tcad_id'] = probs['standardized_tcad_id'].astype(str).str.zfill(10)

case_years = case_tcad[['standardized_tcad_id', 'year']].drop_duplicates()
case_years['event'] = 1

merged = panel_geo.merge(case_years, on=['standardized_tcad_id', 'year'], how='left')
merged['event'] = merged['event'].fillna(0).astype(int)
merged = merged.sort_values(['standardized_tcad_id', 'year'])

merged['y_1yr'] = merged.groupby('standardized_tcad_id')['event'].shift(-1).fillna(0)
merged['y_2yr'] = merged.groupby('standardized_tcad_id')['event'].shift(-2).fillna(0)
merged['y_3yr'] = merged.groupby('standardized_tcad_id')['event'].shift(-3).fillna(0)

# Join with probabilities
full = merged.merge(probs, on=['standardized_tcad_id', 'year'], how='inner')

# FIGURE 3: Gauntlet Precision-Recall Curves (By Horizon)
horizons_map = {4: (1, 'y_1yr'), 8: (2, 'y_2yr'), 12: (3, 'y_3yr')}

for h, (y_yr, y_col) in horizons_map.items():
    print(f"[*] Plotting Figure 3: Academic Gauntlet PR Curves (H={h} Quarters)...")
    plt.figure(figsize=(9, 6))

    targets = [
        (y_col, f'Prob_H={h}', 'Challenger: CatBoost (Swept)', '#1f77b4'),
        (y_col, f'Prob_LGBM_H={h}', 'Challenger: LightGBM (Swept)', '#9467bd'),
        (y_col, f'Prob_LR_H={h}', 'Baseline: Logistic (Dye & McMillen 2007)', '#ff7f0e'),
        (y_col, 'Heuristic_ILR', 'Baseline: ILR Heuristic (Rosenthal & Helsley 1994)', '#2ca02c')
    ]

    for (t_col, prob_col, label, color) in targets:
        if prob_col in full.columns:
            # Handle possible boolean/int inputs and scale explicitly
            proba_arr = full[prob_col].values
            precision, recall, _ = precision_recall_curve(full[t_col], proba_arr)
            auc = average_precision_score(full[t_col], proba_arr)
            plt.plot(recall, precision, label=f"{label} (AUC={auc:.4f})", color=color, lw=2)

    baseline_val = full[y_col].mean()
    plt.axhline(baseline_val, color='red', linestyle='--', alpha=0.5, label=f"Statistical Null (AUC={baseline_val:.4f})")

    plt.xlabel('Recall (Fraction of True Events Captured)')
    plt.ylabel('Precision (Fraction of Predictions that are Events)')
    plt.title(f'Figure 3: Hazard Gauntlet Comparison (H = {h} Quarters)')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'Analysis/Output/Track0_Predictive/StageA_Figure3_H{h}.png', dpi=300)
    plt.close()

# FIGURE 4: Ex-Ante Hotspot Density vs. Realized Events
print("[*] Plotting Figure 4: Spatial Hexbin Map (1-Year Hazard, Top Decile)...")
# We'll plot max probability per parcel across all years for visualization of spatial pressure
map_data = full.groupby('standardized_tcad_id')[['latitude', 'longitude', 'Prob_H=4', 'y_1yr']].max().reset_index()

# Filter out null coords
map_data = map_data.dropna(subset=['latitude', 'longitude', 'Prob_H=4'])
# Simple bounds for Austin
map_data = map_data[(map_data.longitude > -98.1) & (map_data.longitude < -97.5) & 
                    (map_data.latitude > 30.0) & (map_data.latitude < 30.6)]

# Apply statistical significance threshold (Top 10% Hazard Decile)
threshold_90 = np.percentile(map_data['Prob_H=4'], 90)
print(f"    Filtering out ambient noise below the 90th percentile hazard threshold: {threshold_90:.4f}")
significant_hotspots = map_data[map_data['Prob_H=4'] >= threshold_90]

plt.figure(figsize=(10, 10))
plt.hexbin(significant_hotspots['longitude'], significant_hotspots['latitude'], 
           C=significant_hotspots['Prob_H=4'], gridsize=100, cmap='YlOrRd', reduce_C_function=np.mean, mincnt=1, alpha=0.85)
plt.colorbar(label='Average Predicted Development Probability (Top 10% Sites)')

# Overlay true events as small dots
events = map_data[map_data['y_1yr'] == 1]
plt.scatter(events['longitude'], events['latitude'], c='cyan', s=3, alpha=0.5, label='Observed Dev Event')

plt.title('Figure 4: Ex-Ante Development Hotspot Density (Top Decile Filtered)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.tight_layout()
plt.savefig('Analysis/Output/Track0_Predictive/StageA_Figure4_Hotspot.png', dpi=300)
plt.close()

print("[+] Done! Exhibits saved to Analysis/Output/Track0_Predictive/")
