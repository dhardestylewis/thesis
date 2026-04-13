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

import pandas as pd
import os

import sys
_scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from artifact_registry import ROOT_DIR, TraceabilityRegistry as AR

def generate_exhibits():
    print("[*] Rendering Authentic F22: Joint Policy Map (Expected Contested Units)...")
    ROOT = str(ROOT_DIR)
    STAGE_A_OUT = str(AR.STAGE_A_HAZARD_RESULTS)
    STAGE_C_OUT = str(AR.stage_c_oof("H0"))
    PANEL = os.path.join(ROOT, "Data", "Panel", "Output", "Property_Year_Panel_Enriched.csv")
    out_dir = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter4")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(STAGE_A_OUT) or not os.path.exists(STAGE_C_OUT):
        print(f"[!] F22 Failure: Requires completed Development-Proposal Model and C Hazard pipelines at {STAGE_A_OUT} and {STAGE_C_OUT}")
        return

    # Extract mathematically authentic coordinates mapped to explicit model hazard distributions
    probs_a = pd.read_csv(STAGE_A_OUT, usecols=['standardized_tcad_id', 'Prob_Optimal_H=4'])
    probs_c = pd.read_csv(STAGE_C_OUT, usecols=['standardized_tcad_id', 'y_true', 'y_prob'])
    geo = pd.read_csv(PANEL, usecols=['standardized_tcad_id', 'latitude', 'longitude']).drop_duplicates(subset=['standardized_tcad_id'])
    
    probs_a['standardized_tcad_id'] = probs_a['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    probs_c['standardized_tcad_id'] = probs_c['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    geo['standardized_tcad_id'] = geo['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    
    merged = probs_c.merge(probs_a, on='standardized_tcad_id', how='inner')
    merged = merged.merge(geo, on='standardized_tcad_id', how='inner')
    merged = merged.dropna(subset=['latitude', 'longitude', 'Prob_Optimal_H=4', 'y_prob'])
    
    if merged.empty:
        print("[!] Merged spatial evaluation frame is empty.")
        return
        
    x = merged['longitude'].values
    y = merged['latitude'].values
    
    # Calculate pure Joint Protest Event Probability and the Direct Spatial Residual
    joint_protest_prob = merged['Prob_Optimal_H=4'].values * merged['y_prob'].values
    residual_error = merged['y_true'].values - merged['y_prob'].values

    import contextily as cx
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    titles = [
        "1. Predicted Development Probability $P(D)$",
        "2. Ex-Ante Petition Filing $P(O)$",
        "3. Joint Expected Protest Probability $P(D) \\times P(O)$",
        "4. Opposition Mapping Residual ($Y_{True} - P(O)$)"
    ]
    
    C_arrays = [
        merged['Prob_Optimal_H=4'].values,
        merged['y_prob'].values,
        joint_protest_prob,
        residual_error
    ]

    cmaps = ['viridis', 'viridis', 'viridis', 'coolwarm']
    reduce_funcs = [np.mean, np.mean, np.mean, np.mean]
    
    for i, ax in enumerate(axes):
        c_vals = C_arrays[i]
        
        # Threshold to reduce clutter (user requested)
        mask = np.ones(len(c_vals), dtype=bool)
        if i == 0:
            mask = c_vals > np.percentile(c_vals, 50) # Top 50% expected dev
        elif i == 1:
            mask = c_vals > 0.10 # >10% opposition risk
        elif i == 2:
            mask = c_vals > np.percentile(c_vals, 50) # Top 50% joint probability
        elif i == 3:
            mask = np.abs(c_vals) > 0.15 # Hide near-perfect predictions
            
        x_filtered = x[mask]
        y_filtered = y[mask]
        c_filtered = c_vals[mask]
        
        if len(x_filtered) > 0:
            hb = ax.hexbin(x_filtered, y_filtered, C=c_filtered, gridsize=45, cmap=cmaps[i], reduce_C_function=reduce_funcs[i], mincnt=1, alpha=0.85, zorder=2)
            fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        
        # Display Ground Truths
        if i == 0:
            # For Panel 1, all points in this sample are historical development cases
            ax.scatter(x_filtered, y_filtered, c='cyan', s=10, alpha=0.3, edgecolors='none', label='Observed Dev Event', zorder=3)
            ax.legend(loc='lower right', fontsize=10)
        elif i in [1, 2]:
            events = merged[merged['y_true'] == 1]
            ax.scatter(events['longitude'], events['latitude'], c='cyan', s=15, alpha=0.8, edgecolors='black', linewidths=0.5, label='Actual Valid Petition', zorder=3)
            ax.legend(loc='lower right', fontsize=10)
        
        # Adding Authentic Contextily Basemap
        try:
            cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron, zorder=1)
        except Exception as e:
            print(f"Basemap warning on Panel {i}: {e}")
            
        ax.set_title(titles[i], fontsize=14, pad=15)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Expected Petition Probability (Development Hazard \u00d7 Filing-Date Prediction)", fontsize=18, fontweight='bold', y=0.96)

    plt.tight_layout()

    # --- QR code stamp (consistent with Figure 6) ---
    try:
        from PIL import Image
        qr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qr_temporal_drift.png")
        if os.path.exists(qr_path):
            qr_img  = Image.open(qr_path).convert("RGB")
            qr_size = 0.08   # fraction of figure
            ax_qr = fig.add_axes([0.91, 0.015, qr_size, qr_size])
            ax_qr.imshow(qr_img)
            ax_qr.axis("off")
            ax_qr.set_title("Scan for\ninteractive", fontsize=7, pad=2, color="#444444")
    except Exception as e:
        print(f"QR stamp warning: {e}")

    f22_path = os.path.join(out_dir, "F22_Joint_Policy_Map.png")
    plt.savefig(f22_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    [+] Successfully saved 4-panel explicit {f22_path} via authentic hazard vectors and basemaps (with QR stamp).")

if __name__ == "__main__":
    generate_exhibits()
