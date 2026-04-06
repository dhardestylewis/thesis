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
    print("[*] Generating Development-Proposal Model Exhibits (Figures 3 & 4 and Table 6)...")

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

    print("[*] Plotting Figure 3: Multi-Horizon & Multi-Algorithm PR Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {4: '#1f77b4', 8: '#9467bd', 12: '#e377c2'}

    latex_table_rows = []

    # Panel A: Multi-Horizon
    for h, (y_yr, y_col) in horizons_map.items():
        prob_col = f'Prob_Optimal_H={h}'
        if prob_col in full.columns:
            proba_arr = full[prob_col].values
            
            # Calculate PR AUC
            precision, recall, _ = precision_recall_curve(full[y_col], proba_arr)
            auc_score = average_precision_score(full[y_col], proba_arr)
            ax1.plot(recall, precision, label=f"LightGBM (H={h} Qtrs) AUC={auc_score:.4f}", color=colors[h], lw=2)
            
            # Calculate Top-10% Lift
            top_10_thresh = np.percentile(proba_arr, 90)
            top_10_mask = proba_arr >= top_10_thresh
            top_10_rate = full[y_col][top_10_mask].mean()
            base_rate = full[y_col].mean()
            
            lift = top_10_rate / base_rate if base_rate > 0 else 0
            if lift > 100:
                lift_str = f"$\\sim${int(lift):,}\\times$"
            else:
                lift_str = f"{lift:.2f}$\\times$"
            
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

    # Panel A layout
    baseline_val = full['y_1yr'].mean()
    ax1.axhline(baseline_val, color='red', linestyle='--', alpha=0.5, label=f"Null Base Rate (AUC={baseline_val:.4f})")

    ax1.set_xlabel('Recall (Fraction of True Events Captured)')
    ax1.set_ylabel('Precision (Fraction of Predictions that are Events)')
    ax1.set_title('A: Multi-Horizon Hazard (Optimal Model)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel B: Algorithm Comparison (H=4)
    model_types = {
        'Prob_LGBM_H=4': ('LightGBM', '#1f77b4'),
        'Prob_CB_H=4': ('CatBoost', '#ff7f0e'),
        'Prob_DL_H=4': ('Deep Learning (MLP)', '#2ca02c'),
        'Prob_SAR_H=4': ('SAR-Logistic', '#d62728'),
        'Prob_LR_H=4': ('Logistic Base', '#9467bd')
    }
    
    y_col = 'y_1yr'
    for model_col, (label, color) in model_types.items():
        if model_col in full.columns:
            proba_arr = full[model_col].values
            precision, recall, _ = precision_recall_curve(full[y_col], proba_arr)
            auc_score = average_precision_score(full[y_col], proba_arr)
            ax2.plot(recall, precision, label=f"{label} (AUC={auc_score:.4f})", color=color, lw=2)
            
    ax2.axhline(baseline_val, color='red', linestyle='--', alpha=0.5, label=f"Null Base Rate (AUC={baseline_val:.4f})")
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('B: Multi-Algorithm Validation (1-Year Horizon)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = str(AR.TRACK0_FIGURES / 'StageA_Figure3_PR_Curves.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"    [+] Saved Figure 4 (Development-Proposal Model PR Curves) to {fig_path}")

    # FIGURE 4: Predicted High-Probability Areas vs. Realized Events (Multi-Horizon)
    print("[*] Plotting Figure 4: Spatial Hexbin Maps (4, 8, 12-Quarter Hazard, Top-N Equivalent)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes_flat = axes.flatten()
    import contextily as cx
    
    horizons = [(4, 'y_1yr'), (8, 'y_2yr'), (12, 'y_3yr')]
    
    for idx, (h, y_col) in enumerate(horizons):
        prob_col = f'Prob_Optimal_H={h}'
        ax = axes_flat[idx]
        
        map_data = full.groupby('standardized_tcad_id')[['latitude', 'longitude', prob_col, y_col]].max().reset_index()
        map_data = map_data.dropna(subset=['latitude', 'longitude', prob_col])
        map_data = map_data[(map_data.longitude > -98.1) & (map_data.longitude < -97.5) & 
                            (map_data.latitude > 30.0) & (map_data.latitude < 30.6)]
        
        events = map_data[map_data[y_col] == 1]
        num_events = len(events)
        
        # Select Equivalent N hotspots instead of generic Top 10% Decile
        significant_hotspots = map_data.nlargest(num_events, prob_col)
        
        hb = ax.hexbin(significant_hotspots['longitude'], significant_hotspots['latitude'], 
                       C=significant_hotspots[prob_col], gridsize=100, cmap='YlOrRd', reduce_C_function=np.mean, mincnt=1, alpha=0.85)
        
        if idx == 2:
            fig.colorbar(hb, ax=axes.ravel().tolist(), label=f'Predicted Density (Top N={num_events:,} Sites)', fraction=0.04, pad=0.04)
            
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron)
        
        ax.scatter(events['longitude'], events['latitude'], c='#00FFFF', s=0.7, alpha=0.35, linewidths=0, label='Observed Dev Event')
        
        ax.set_title(f'H={h} Quarters Ex-ante Hotspots')
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.legend(loc='lower left')

    # Turn off the empty bottom-right subplot
    axes_flat[3].axis('off')

    fig.suptitle('Multi-Horizon High-Probability Areas vs. Realized Development Events', fontsize=18, fontweight='bold', y=0.96)
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    hotspot_path = str(AR.TRACK0_FIGURES / 'StageA_Figure4_Hotspot.png')
    plt.savefig(hotspot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    [+] Saved Figure 4 (Hotspot Map Grid) to {hotspot_path}")

    print("[+] Done! Exhibits saved to Analysis/Output/Track0_Predictive/")

if __name__ == "__main__":
    generate_exhibits()
