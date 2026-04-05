import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

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

from artifact_registry import ROOT_DIR, TraceabilityRegistry as AR

# Make sure the paths use ROOT
ROOT = str(ROOT_DIR)

def generate_exhibits():
    print("[*] Generating Stage A Exhibits (Figures 3 & 4 and Table 6)...")

    # 1. Load probabilities
    probs_path = str(AR.STAGE_A_HAZARD_RESULTS)
    probs = pd.read_csv(probs_path)

    # 2. Extract Latitude/Longitude and events efficiently
    print("    Loading spatial coordinates and calculating events...")
    panel_geo_path = os.path.join(ROOT, 'Data', 'Panel', 'Output', 'Property_Year_Panel_v3.csv')
    case_tcad_path = os.path.join(ROOT, 'Data', 'Warehouse_As_Of', 'H0_Filing_Master_Enriched.csv')
    
    panel_geo = pd.read_csv(panel_geo_path, usecols=['standardized_tcad_id', 'year', 'latitude', 'longitude'])
    case_tcad = pd.read_csv(case_tcad_path, usecols=['standardized_tcad_id', 'year'], low_memory=False)

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

    # FIGURE 3 & TABLE 6: Multi-Horizon Precision-Recall Curves (Combined)
    horizons_map = {4: (1, 'y_1yr'), 8: (2, 'y_2yr'), 12: (3, 'y_3yr')}

    print("[*] Plotting Figure 3: Multi-Horizon PR Curves...")
    plt.figure(figsize=(9, 6))
    
    colors = {4: '#1f77b4', 8: '#9467bd', 12: '#e377c2'}

    latex_table_rows = []

    for h, (y_yr, y_col) in horizons_map.items():
        prob_col = f'Prob_Optimal_H={h}'
        if prob_col in full.columns:
            proba_arr = full[prob_col].values
            
            # Calculate PR AUC
            precision, recall, _ = precision_recall_curve(full[y_col], proba_arr)
            auc_score = average_precision_score(full[y_col], proba_arr)
            plt.plot(recall, precision, label=f"LightGBM (H={h} Qtrs) AUC={auc_score:.4f}", color=colors[h], lw=2)
            
            # Calculate Top-10% Lift
            top_10_thresh = np.percentile(proba_arr, 90)
            top_10_mask = proba_arr >= top_10_thresh
            top_10_rate = full[y_col][top_10_mask].mean()
            base_rate = full[y_col].mean()
            
            lift = top_10_rate / base_rate if base_rate > 0 else 0
            # dynamically format: if lift is large (e.g > 100), round to ints, else preserve decimals
            if lift > 100:
                lift_str = f"$\\sim${int(lift):,}\\times$"
            else:
                lift_str = f"{lift:.2f}$\\times$"
            
            # Add row to latex table
            latex_table_rows.append(f"{h}-Quarter & {auc_score:.4f} & LightGBM & {lift_str} \\\\")

    # Write Latex Table
    table_path = str(AR.TRACK0_METRICS / 'Table6_multi_horizon.tex')
    with open(table_path, 'w') as f:
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Horizon} & \\textbf{PR-AUC} & \\textbf{Algorithm} & \\textbf{Lift over Baseline} \\\\\n")
        f.write("\\midrule\n")
        for row in latex_table_rows:
            f.write(row + "\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"    [+] Saved dynamic Table 6 to {table_path}")

    # Only one baseline for aesthetic purposes
    baseline_val = full['y_1yr'].mean()
    plt.axhline(baseline_val, color='red', linestyle='--', alpha=0.5, label=f"Null Base Rate (AUC={baseline_val:.4f})")

    plt.xlabel('Recall (Fraction of True Events Captured)')
    plt.ylabel('Precision (Fraction of Predictions that are Events)')
    plt.title('Multi-Horizon Hazard PR Curves (LightGBM)')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = str(AR.TRACK0_FIGURES / 'StageA_Figure3_PR_Curves.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"    [+] Saved Figure 4 (Stage A PR Curves) to {fig_path}")

    # FIGURE 4: Ex-Ante Hotspot Density vs. Realized Events (Multi-Horizon)
    print("[*] Plotting Figure 4: Spatial Hexbin Maps (4, 8, 12-Quarter Hazard, Top Decile)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    import contextily as cx
    
    horizons = [(4, 'y_1yr'), (8, 'y_2yr'), (12, 'y_3yr')]
    
    for idx, (h, y_col) in enumerate(horizons):
        prob_col = f'Prob_Optimal_H={h}'
        ax = axes[idx]
        
        map_data = full.groupby('standardized_tcad_id')[['latitude', 'longitude', prob_col, y_col]].max().reset_index()
        map_data = map_data.dropna(subset=['latitude', 'longitude', prob_col])
        map_data = map_data[(map_data.longitude > -98.1) & (map_data.longitude < -97.5) & 
                            (map_data.latitude > 30.0) & (map_data.latitude < 30.6)]
        
        threshold_90 = np.percentile(map_data[prob_col], 90)
        significant_hotspots = map_data[map_data[prob_col] >= threshold_90]
        
        hb = ax.hexbin(significant_hotspots['longitude'], significant_hotspots['latitude'], 
                       C=significant_hotspots[prob_col], gridsize=100, cmap='YlOrRd', reduce_C_function=np.mean, mincnt=15, alpha=0.85)
        
        if idx == 2:
            fig.colorbar(hb, ax=axes, label='Average Predicted Development Probability (Top 10% Sites)', fraction=0.02, pad=0.04)
            
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron)
        
        events = map_data[map_data[y_col] == 1]
        ax.scatter(events['longitude'], events['latitude'], c='cyan', s=3, alpha=0.5, label='Observed Dev Event')
        
        ax.set_title(f'H={h} Quarters Ex-ante Hotspots')
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.legend(loc='lower left')

    fig.suptitle('Multi-Horizon Hotspot Density vs. Realized Development Events', fontsize=18, fontweight='bold', y=1.02)
    hotspot_path = str(AR.TRACK0_FIGURES / 'StageA_Figure4_Hotspot.png')
    plt.savefig(hotspot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    [+] Saved Figure 4 (Hotspot Map Grid) to {hotspot_path}")

    print("[+] Done! Exhibits saved to Analysis/Output/Track0_Predictive/")

if __name__ == "__main__":
    generate_exhibits()
