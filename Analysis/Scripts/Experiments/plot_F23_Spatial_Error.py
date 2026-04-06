import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import contextily as cx

sys.path.insert(0, os.path.join(r"C:\Users\dhl\data\thesis\thesis", "Analysis", "Scripts"))
from artifact_registry import TraceabilityRegistry as AR
from thesis_style import set_thesis_style, OKABE_ITO

def generate_spatial_error_map():
    set_thesis_style()
    
    PREDS_FILE = str(AR.STAGE_C_OOF_H0)
    DATA_FILE = os.path.join(r"C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
    
    if not os.path.exists(PREDS_FILE) or not os.path.exists(DATA_FILE):
        print("Missing required files for spatial map.")
        return

    df = pd.read_csv(PREDS_FILE)
    
    # Reload H0 to grab spatial coordinates deterministically
    master = pd.read_csv(DATA_FILE, usecols=['year', 'latitude', 'longitude'], low_memory=False)
    master['year'] = pd.to_numeric(master['year'], errors='coerce')
    master = master.dropna(subset=['year']).sort_values('year').copy()
    
    if len(master) != len(df):
        print(f"Row count mismatch! H0 has {len(master)}, OOF has {len(df)}")
        return
        
    df['latitude'] = master['latitude'].values
    df['longitude'] = master['longitude'].values
    
    # Drop rows without geographic markers
    df = df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)
    
    # Define primary model threshold
    global_thresh = df['y_true'].mean()
    df['pred'] = (df['y_prob'] > global_thresh).astype(int)
    
    # Categorize Spatial Errors
    df['error_type'] = 'True Negative' # Correctly predicted no risk
    df.loc[(df['y_true'] == 1) & (df['pred'] == 1), 'error_type'] = 'True Positive' # Correctly predicted risk
    df.loc[(df['y_true'] == 0) & (df['pred'] == 1), 'error_type'] = 'False Positive' # Ghost risk (overprediction)
    df.loc[(df['y_true'] == 1) & (df['pred'] == 0), 'error_type'] = 'False Negative' # Missed risk
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel 1: True Positives (Where organized opposition actually happens)
    ax1 = axes[0]
    tp = df[df['error_type'] == 'True Positive']
    ax1.scatter(tp['longitude'], tp['latitude'], c=OKABE_ITO[4], alpha=0.6, s=25, zorder=2, label='Actual Localized Protests')
    
    try:
        cx.add_basemap(ax1, crs="EPSG:4326", source=cx.providers.CartoDB.PositronNoLabels, zorder=1)
    except Exception as e:
        print(e)
    ax1.set_title("Historical True Positives\n(Accurate Regulatory Friction)", fontsize=16, pad=15)
    ax1.set_xlim(-97.9, -97.6)
    ax1.set_ylim(30.15, 30.5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend(loc='lower right')
    
    # Panel 2: False Positives (Where the model legally overpenalizes)
    ax2 = axes[1]
    fp = df[df['error_type'] == 'False Positive']
    ax2.scatter(fp['longitude'], fp['latitude'], c=OKABE_ITO[5], alpha=0.7, s=25, zorder=2, label='Algorithmic False Positives')
    
    # Draw I-35 roughly
    i35_lons = [-97.790, -97.750, -97.735, -97.730, -97.715, -97.695, -97.670]
    i35_lats = [30.160, 30.210, 30.240, 30.275, 30.300, 30.335, 30.380]
    ax2.plot(i35_lons, i35_lats, 'k--', alpha=0.5, zorder=3, label="I-35 Historical Crescent Divide")
    
    try:
        cx.add_basemap(ax2, crs="EPSG:4326", source=cx.providers.CartoDB.PositronNoLabels, zorder=1)
    except Exception as e:
        print(e)
    ax2.set_title("Algorithmic False Positives\n(Geographic Exposure to Error)", fontsize=16, pad=15)
    ax2.set_xlim(-97.9, -97.6)
    ax2.set_ylim(30.15, 30.5)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    
    artifact_path = r"C:\Users\dhl\.gemini\antigravity\brain\ebf7d3ae-8672-4ccd-9da8-331e25c23773\F23_Spatial_Error.png"
    plt.savefig(artifact_path, dpi=300, bbox_inches='tight')
    
    out_dir = os.path.join(r"C:\Users\dhl\data\thesis\thesis\Thesis_Draft\Draft_v1\Figures\Chapter4")
    os.makedirs(out_dir, exist_ok=True)
    repo_path = os.path.join(out_dir, "F23_Spatial_Error.png")
    plt.savefig(repo_path, dpi=300, bbox_inches='tight')
    
    print(f"[*] Visual rendering complete: saved to {artifact_path} & {repo_path}")

if __name__ == '__main__':
    generate_spatial_error_map()
