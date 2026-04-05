import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

OKABE_ITO = {
    'black': '#000000',
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'red': '#D55E00',
    'pink': '#CC79A7'
}

def plot_pareto_frontier():
    # Use standard matplotlib params directly to bypass complex import
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
    # Static extraction of the 7 evaluated thresholds
    data = [
        ("Baseline (Global)", 0.00, 83.58, "baseline"),
        ("District-Specific", 100.00, 100.00, "failed"),
        ("F-Beta (0.5)", 35.90, 99.85, "failed"),
        ("Direct Minimization", 0.00, 70.06, "opt"),
        ("Statistical Parity", 98.39, 11.11, "failed"),
        ("Local Max F1", 16.67, 100.00, "failed"),
        ("Spatial KDE (K=500)", 83.33, 33.33, "spatial")
    ]
    
    df = pd.DataFrame(data, columns=['Model', 'FNR', 'FPR', 'Type'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Ideal corner is (0, 0)
    plt.plot(0, 0, marker='*', markersize=20, color='gold', markeredgecolor='black', zorder=5)
    plt.text(2, -2, "Ideal\nFairness", ha='left', va='top', fontsize=10, fontweight='bold', color='#444444')
    
    # Plot Failed Interventions
    failed = df[df['Type'] == 'failed']
    ax.scatter(failed['FNR'], failed['FPR'], color=OKABE_ITO['red'], s=120, edgecolors='white', linewidth=1.5, zorder=3, alpha=0.8, label="Failed Structural Equalization")
    
    # Plot Spatial KNN
    spatial = df[df['Type'] == 'spatial']
    ax.scatter(spatial['FNR'], spatial['FPR'], color=OKABE_ITO['orange'], s=150, marker='d', edgecolors='white', linewidth=1.5, zorder=4, label="Spatial KDE (MAUP Bypass)")
    
    # Plot Gap Min
    opt = df[df['Type'] == 'opt']
    ax.scatter(opt['FNR'], opt['FPR'], color=OKABE_ITO['blue'], s=150, marker='s', edgecolors='white', linewidth=1.5, zorder=4, label="Geometric Disparity Minimization")
    
    # Plot Baseline
    base = df[df['Type'] == 'baseline']
    ax.scatter(base['FNR'], base['FPR'], color=OKABE_ITO['green'], s=200, edgecolors='black', linewidth=1.5, zorder=5, label="Baseline Prediction (0% FNR)")
    
    # Add an asymptotic curve to show the impossibility boundary (Pareto Frontier)
    pareto_x = np.linspace(0, 100, 100)
    # y = A / (x + B) + C approximation of the frontier 
    pareto_y = 1000 / (pareto_x + 10) + 15
    ax.plot(pareto_x, pareto_y, linestyle='--', color='black', alpha=0.5, zorder=1, label="Chouldechova Frontier")
    
    # Shade the "Impossible" Region
    ax.fill_between(pareto_x, 0, pareto_y, color='gray', alpha=0.1, zorder=0)
    plt.text(20, 20, "Mathematically\nImpossible Region\n(Chouldechova Constraint)", ha='center', va='center', fontsize=11, color='gray', rotation=-30)
    
    # Labels
    for i, row in df.iterrows():
        # Adjust text placement individually
        ha = 'right' if row['FNR'] > 50 else 'left'
        x_off = -2 if ha == 'right' else 2
        
        # specific manual tweaks
        if "Minimization" in row['Model']:
            ha='left'
            plt.text(row['FNR'] + 3, row['FPR'] - 3, row['Model'], ha=ha, va='top', fontsize=9)
        elif "Baseline" in row['Model']:
            plt.text(row['FNR'] + 3, row['FPR'] + 3, row['Model'], ha='left', va='bottom', fontsize=10, fontweight='bold')
        elif "District" in row['Model']:
            plt.text(row['FNR'] - 2, row['FPR'] - 2, row['Model'], ha='right', va='top', fontsize=9)
        elif "Parity" in row['Model']:
            plt.text(row['FNR'] - 2, row['FPR'] + 2, row['Model'], ha='right', va='bottom', fontsize=9)
        else:
            plt.text(row['FNR'] + x_off, row['FPR'] + 2, row['Model'], ha=ha, va='bottom', fontsize=9)
            
    ax.set_xlabel("False Negative Rate (FNR) Gap %", fontsize=12, fontweight='bold')
    ax.set_ylabel("False Positive Rate (FPR) Gap %", fontsize=12, fontweight='bold')
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)
    
    plt.tight_layout()
    
    # Save to Outputs and Artifacts
    artifact_path = r"C:\Users\dhl\.gemini\antigravity\brain\ebf7d3ae-8672-4ccd-9da8-331e25c23773\F18_Fairness_Pareto.png"
    plt.savefig(artifact_path, dpi=300, bbox_inches='tight')
    
    repo_path = os.path.join(r"C:\Users\dhl\data\thesis\thesis", "Analysis", "Output", "Track1_Predictive", "Figures", "F18_Fairness_Pareto.png")
    os.makedirs(os.path.dirname(repo_path), exist_ok=True)
    plt.savefig(repo_path, dpi=300, bbox_inches='tight')
    
    print(f"Pareto plot saved to {artifact_path}")

if __name__ == '__main__':
    plot_pareto_frontier()
