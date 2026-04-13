import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# Attempt to locate the root Scripts directory
_curr = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(_curr) != 'Scripts' and os.path.dirname(_curr) != _curr:
    _curr = os.path.dirname(_curr)
if _curr not in sys.path:
    sys.path.insert(0, _curr)
try:
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

ROOT_DIR = r"C:\Users\dhl\data\thesis\thesis"
DATA_IN = os.path.join(ROOT_DIR, "Analysis", "Output", "Track1_Predictive", "rolling_origin_drift.json")
FIGURES_DIR = os.path.join(ROOT_DIR, "Thesis_Draft", "Draft_v1", "Figures", "Chapter4")
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_3d_drift():
    if not os.path.exists(DATA_IN):
        print(f"Data file not found: {DATA_IN}")
        return

    with open(DATA_IN, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    
    # Extract numerical anchor year from 'Pre-YYYY'
    df['Anchor_Year'] = df['Anchor'].str.replace('Pre-', '').astype(int)
    
    # We will plot two surfaces: CatBoost and TabNet(LS+Pruning)
    df_cat = df[df['Model'] == 'CatBoost'].dropna(subset=['PR-AUC'])
    df_tab = df[df['Model'] == 'TabNet(LS+Pruning)'].dropna(subset=['PR-AUC'])

    fig = plt.figure(figsize=(18, 6))
    
    view_angles = [(25, -45), (25, 45), (25, 135)]
    
    for i, (elev, azim) in enumerate(view_angles):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Plot CatBoost surface
        if not df_cat.empty:
            surf1 = ax.plot_trisurf(
                df_cat['Anchor_Year'], 
                df_cat['Evaluate_Year'], 
                df_cat['PR-AUC'], 
                cmap='Blues', 
                alpha=0.8, 
                edgecolor='blue',
                linewidth=0.2,
                label='CatBoost (Structural)'
            )
            surf1._facecolors2d = surf1._facecolor3d
            surf1._edgecolors2d = surf1._edgecolor3d

        # Plot TabNet sparse surface
        if not df_tab.empty:
            surf2 = ax.plot_trisurf(
                df_tab['Anchor_Year'], 
                df_tab['Evaluate_Year'], 
                df_tab['PR-AUC'], 
                cmap='Reds', 
                alpha=0.6, 
                edgecolor='red',
                linewidth=0.2,
                label='TabNet Pruned'
            )
            surf2._facecolors2d = surf2._facecolor3d
            surf2._edgecolors2d = surf2._edgecolor3d

        ax.set_xlabel('Anchor Year', labelpad=10)
        ax.set_ylabel('Horizon Year', labelpad=10)
        ax.set_zlabel('PR-AUC Performance', labelpad=10)
        
        # Set view angle to emphasize the drop-off
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Viewing Angle: {azim}°")
        
        if i == 1:
            # Place legend centrally on the middle plot
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    plt.suptitle("3D Temporal Drift Topology: Structural Guardrails vs Deep Interpolation", fontsize=16, y=1.02)
    
    out_path = os.path.join(FIGURES_DIR, "Fig_3D_Temporal_Drift.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] 3D Temporal Drift Visualization saved to {out_path}")

if __name__ == "__main__":
    plot_3d_drift()
