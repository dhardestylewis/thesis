import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
try:
    # Attempt to locate the root Scripts directory
    _curr = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(_curr) != 'Scripts' and os.path.dirname(_curr) != _curr:
        _curr = os.path.dirname(_curr)
    if _curr not in sys.path:
        sys.path.insert(0, _curr)
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

from sklearn.metrics import precision_recall_curve, average_precision_score
import os

def generate_exhibits():
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

    # FIGURE 3: Multi-Horizon Precision-Recall Curves (Combined)
    horizons_map = {4: (1, 'y_1yr'), 8: (2, 'y_2yr'), 12: (3, 'y_3yr')}

    print("[*] Plotting Figure 3: Multi-Horizon PR Curves...")
    plt.figure(figsize=(9, 6))
    
    colors = {4: '#1f77b4', 8: '#9467bd', 12: '#e377c2'}

    for h, (y_yr, y_col) in horizons_map.items():
        prob_col = f'Prob_H={h}'
        if prob_col in full.columns:
            proba_arr = full[prob_col].values
            precision, recall, _ = precision_recall_curve(full[y_col], proba_arr)
            auc_score = average_precision_score(full[y_col], proba_arr)
            plt.plot(recall, precision, label=f"CatBoost (H={h} Qtrs) AUC={auc_score:.4f}", color=colors[h], lw=2)

    # Only one baseline for aesthetic purposes
    baseline_val = full['y_1yr'].mean()
    plt.axhline(baseline_val, color='red', linestyle='--', alpha=0.5, label=f"Null Base Rate (AUC={baseline_val:.4f})")

    plt.xlabel('Recall (Fraction of True Events Captured)')
    plt.ylabel('Precision (Fraction of Predictions that are Events)')
    plt.title('Multi-Horizon Hazard PR Curves (CatBoost)')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Analysis/Output/Track0_Predictive/StageA_Figure3_PR_Curves.png', dpi=300)
    plt.close()

    # FIGURE 4: Ex-Ante Hotspot Density vs. Realized Events
    print("[*] Plotting Figure 4: Spatial Hexbin Map (1-Year Hazard, Top Decile)...")
    map_data = full.groupby('standardized_tcad_id')[['latitude', 'longitude', 'Prob_H=4', 'y_1yr']].max().reset_index()

    map_data = map_data.dropna(subset=['latitude', 'longitude', 'Prob_H=4'])
    map_data = map_data[(map_data.longitude > -98.1) & (map_data.longitude < -97.5) & 
                        (map_data.latitude > 30.0) & (map_data.latitude < 30.6)]

    import contextily as cx
    threshold_90 = np.percentile(map_data['Prob_H=4'], 90)
    print(f"    Filtering out ambient noise below the 90th percentile hazard threshold: {threshold_90:.4f}")
    significant_hotspots = map_data[map_data['Prob_H=4'] >= threshold_90]

    fig, ax = plt.subplots(figsize=(10, 10))
    hb = ax.hexbin(significant_hotspots['longitude'], significant_hotspots['latitude'], 
                   C=significant_hotspots['Prob_H=4'], gridsize=100, cmap='YlOrRd', reduce_C_function=np.mean, mincnt=15, alpha=0.85)
    plt.colorbar(hb, ax=ax, label='Average Predicted Development Probability (Top 10% Sites)')
    
    # Add basemap using contextily (assuming coordinates are WGS84)
    cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron)

    events = map_data[map_data['y_1yr'] == 1]
    ax.scatter(events['longitude'], events['latitude'], c='cyan', s=3, alpha=0.5, label='Observed Dev Event')

    ax.set_title('Ex-ante predicted hotspot density vs. realized development events')
    ax.set_xlabel('City of Austin')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    plt.tight_layout()
    plt.savefig('Analysis/Output/Track0_Predictive/StageA_Figure4_Hotspot.png', dpi=300)
    plt.close()

    print("[+] Done! Exhibits saved to Analysis/Output/Track0_Predictive/")

if __name__ == "__main__":
    generate_exhibits()
