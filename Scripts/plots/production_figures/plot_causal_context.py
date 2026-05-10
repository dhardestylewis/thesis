import geopandas as gpd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.colors as mcolors

print("Loading Austin 10-1 Council Districts GeoJSON...")
import sys, os
ROOT_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR_PATH not in sys.path: sys.path.append(ROOT_DIR_PATH)
from pipeline.config.paths import GIS_DIR

districts_path = GIS_DIR / 'council_districts.geojson'

try:
    gdf = gpd.read_file(districts_path)
except Exception as e:
    print(f"Failed to load: {e}")
    exit(1)

if gdf.crs and gdf.crs.to_epsg() == 4326:
    gdf = gdf.to_crs(epsg=3857)

dist_col = None
for col in gdf.columns:
    if 'dist' in col.lower() or 'council' in col.lower():
        dist_col = col
        break
if not dist_col:
    dist_col = gdf.columns[0]

# Rather than a categorical Treatment vs Control, we map the actual estimated
# Marginal Effect coefficient from the Difference-in-Differences regression:
# Flipped District (4,9) x Post-2022 resulted in beta = -0.546 petition rate.
# All other districts represent the reference baseline (0.00).

def get_did_effect(dist_val):
    if str(dist_val) in ['4', '9', '4.0', '9.0', 4, 9]:
        return -0.546
    else:
        return 0.000

gdf['DiD_Effect'] = gdf[dist_col].apply(get_did_effect)

fig, ax = plt.subplots(1, 1, figsize=(9, 8))

# Define a custom diverging colormap centered at 0
# 0 -> Light Grey (Status Quo)
# Negative -> Dark Orange (Decrease in Petition Filing Rate)
num_colors = 100
colors = ['#dd8452', '#f4d1b6', '#e0e0e0'] 
cmap = mcolors.LinearSegmentedColormap.from_list('did_cmap', colors)

# Normalize colormap between -0.6 and +0.1 to center the grey on 0
norm = mcolors.Normalize(vmin=-0.6, vmax=0.1)

cax = gdf.plot(
    column='DiD_Effect',
    ax=ax,
    linewidth=0.8,
    edgecolor='white',
    cmap=cmap,
    norm=norm,
    legend=False
)

# Label the districts with coefficient explicitly
for idx, row in gdf.iterrows():
    centroid = row.geometry.centroid
    dist_val = str(row[dist_col]).split('.')[0]
    
    if float(row['DiD_Effect']) < 0:
        eff_str = f"$\Delta$ -0.54"
        font_w = 'bold'
        c = 'white'
    else:
        eff_str = "Control"
        font_w = 'normal'
        c = '#444444'
        
    ax.annotate(text=f"D{dist_val}\n{eff_str}", xy=(centroid.x, centroid.y),
                xytext=(-10, -5), textcoords="offset points",
                fontsize=11, fontweight=font_w, color=c, alpha=0.9, ha='center')

# Add legend colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
cbar = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.04)
cbar.set_label('Difference-in-Differences Coefficient ($\Delta$ Petition Filing Rate)', fontsize=12)

plt.title("2022 Electoral Transition: Geographic Treatment Effects", fontsize=15, pad=15)
ax.axis('off')

out_path = "c:/Users/dhl/data/thesis/thesis/Thesis_Draft/Draft_v1/Figures/Chapter5/fig_causal_context_did.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Generated geospatial DiD effect map at: {out_path}")
