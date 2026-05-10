import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DRAFT_DIR = os.path.join(ROOT, "Thesis_Draft")

df = pd.read_csv(os.path.join(DRAFT_DIR, "Recursive_LTR_Omni_Clustermap.csv"), index_col=0)

def assign_semantic_name(col):
    text = col.lower()
    if 'meta' in text:
        return "Meta-Stack Parity Failures (Regression/Cliff Models)"
    if 'lag' in text:
        if 'rent' in text or 'population' in text:
            return "Lagged Demographic Shift (1-6 Yr)"
        elif 'appraise' in text or 'value' in text:
            return "Historical Valuation Momentum Lags"
        else:
            return "Macro-Historical Base Geometries"
    else:
        if 'lat' in text or 'lon' in text:
            return "Absolute Geographic Lat/Long Coordinates"
        elif 'gross' in text or 'lotsize' in text:
            return "Rigid Density Boundaries (Acreage/Scale)"
        elif 'apprais' in text or 'value' in text:
            return "Current Valuation Gradients"
        elif 'rent' in text or 'population' in text:
            return "Immediate Structural Demographics"
        else:
            return "Administrative Artifacts & Agenda Text"
            
# Create aggregated matrix
semantic_df_map = pd.DataFrame()

# Sum attribution across raw features mapping to the same semantic cluster
for col in df.columns:
    cluster_name = assign_semantic_name(col)
    if cluster_name not in semantic_df_map:
        semantic_df_map[cluster_name] = df[col]
    else:
        semantic_df_map[cluster_name] += df[col]

# Re-normalize just to be safe
semantic_df_map = semantic_df_map.div(semantic_df_map.sum(axis=1), axis=0).fillna(0.0) * 100.0

sns.set_theme(style="white", context="paper", font_scale=1.1)

# Plot Semantically Clustered Matrix
cg = sns.clustermap(
    semantic_df_map, 
    cmap="magma", 
    figsize=(18, 14), 
    linewidths=.5, 
    annot=True,
    fmt=".1f",
    cbar_kws={'label': 'Clustered Relational Attribution (%)'}
)

cg.fig.suptitle("Recursive Meta-Meta Attribution via Semantic Clustering\nValidating that Relational Architectures isolate Momentum & Density Boundaries over Continuous Absolutes", 
                fontsize=16, weight='bold', y=1.02)

out_png = os.path.join(DRAFT_DIR, "plot_recursive_semantic_omni_clustermap.png")
cg.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close()

print(f"Generated successfully to: {out_png}")
