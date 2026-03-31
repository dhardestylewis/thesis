import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


ROOT_DIR = r"C:\Users\dhl\data\thesis\thesis"
WORK_DIR = os.path.join(ROOT_DIR, "Data", "Warehouse_As_Of")
FIGURES_DIR = os.path.join(ROOT_DIR, "Thesis_Draft", "Draft_v1", "Figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
# Removed local style: sns.set_theme(style="whitegrid", context="paper")

def generate_descriptive_stats():
    print("Generating Descriptive Statistics and Panel Visualizations...")
    
    try:
        df = pd.read_csv(os.path.join(WORK_DIR, "H0_Filing_Master_Enriched.csv"), low_memory=False)
    except:
        print("Required H0 Master Enriched dataset not found.")
        return
        
    df['organized_opposition'] = df['is_protested'].fillna(0).astype(int)
    
    # 1. LaTeX Summary Statistics Table
    desc_vars = [
        'gross_site_area_acres', 'delta_max_height_ft', 'delta_max_far', 'delta_max_bldg_cov_pct', 
        'year', 'organized_opposition', 'zoning_case_nearby', 'distance_to_core_m', 'nearest_park_dist_m',
        'median_income_fill', 'pct_renter_fill', 'pct_white_fill', 'pct_bachelor_fill', 'appraised_val_per_sqft_fill'
    ]
    
    # Ensure columns exist before describing to prevent key errors
    available_vars = [v for v in desc_vars if v in df.columns]
    desc_df = df[available_vars].describe().T
    desc_df = desc_df[['count', 'mean', 'std', 'min', '50%', 'max']]
    desc_df.rename(columns={'50%': 'Median'}, inplace=True)
    
    # Rename for academic presentation conditionally
    rename_dict = {
        'gross_site_area_acres': 'Gross Site Area (Acres)',
        'delta_max_height_ft': 'Requested Height Delta (ft)',
        'delta_max_far': 'Requested FAR Delta',
        'delta_max_bldg_cov_pct': 'Requested Bldg Cov Delta (\\%)',
        'year': 'Filing Year',
        'organized_opposition': 'Organized Opposition (Binary)',
        'zoning_case_nearby': 'Contemporaneous Nearby Cases (200ft)',
        'distance_to_core_m': 'Distance to Urban Core (m)',
        'nearest_park_dist_m': 'Distance to Nearest Park (m)',
        'median_income_fill': 'ACS Median Household Income (\\$)',
        'pct_renter_fill': 'ACS Renter Occupied (\\%)',
        'pct_white_fill': 'ACS White Population (\\%)',
        'pct_bachelor_fill': 'ACS Bachelor Degree or Higher (\\%)',
        'appraised_val_per_sqft_fill': 'TCAD Appraised Value per SqFt (\\$)'
    }
    desc_df.index = [rename_dict.get(i, i) for i in desc_df.index]
    
    # Generate standard booktabs
    raw_latex = desc_df.to_latex(float_format="%.2f", caption="Historical Panel Descriptive Statistics (V2 Full Dimensional Array)", label="tab:desc_stats")
    
    # Inject resizebox to prevent violent margin overflows natively
    latex_table = raw_latex.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}").replace("\\end{tabular}", "\\end{tabular}\n}")
    
    table_path = os.path.join(ROOT_DIR, "Thesis_Draft", "Draft_v1", "summary_stats_table.tex")
    with open(table_path, "w") as f:
        f.write(latex_table)
    print("Exported LaTeX Summary Statistics Table.")

    # 2. Historical Volume and Opposition Rate Visualization
    temporal_df = df[df['year'] >= 1990].copy()
    temporal_grouped = temporal_df.groupby('year').agg(
        total_cases=('case_number', 'count'),
        opposition_rate=('organized_opposition', 'mean')
    ).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar chart for volume
    ax1.bar(temporal_grouped['year'], temporal_grouped['total_cases'], color='lightgray', edgecolor='black', alpha=0.7, label="Total Zoning Cases Filed")
    ax1.set_xlabel("Filing Year", fontsize=12)
    ax1.set_ylabel("Total Case Volume", fontsize=12, color='dimgray')
    ax1.tick_params(axis='y', labelcolor='dimgray')
    
    # Line chart for opposition rate
    ax2 = ax1.twinx()
    ax2.plot(temporal_grouped['year'], temporal_grouped['opposition_rate'] * 100, color='crimson', marker='o', lw=2, label="Organized Opposition Rate (%)")
    ax2.set_ylabel("Opposition Rate (%)", fontsize=12, color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    
    # Formatting
    plt.title("Track 1 Data Assembly: Austin Zoning Case Volume and Historical Opposition Dynamics (1990-2024)", fontsize=13, pad=15)
    
    # Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig13_Cases_Over_Time.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Exported Temporal Volume & Opposition Visualization to Fig13.")

if __name__ == "__main__":
    generate_descriptive_stats()
