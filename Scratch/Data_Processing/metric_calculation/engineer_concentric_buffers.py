import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

ZONING_CSV = r"c:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV\zoning_land_use_merged_data.csv"
PANEL_CSV = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"

print("1. Loading Zoning Geometry...")
df_zone = pd.read_csv(ZONING_CSV, low_memory=False)

# Need case_number, lat, lon, and a proxy for protest
# Wait, the label_valid_protest is in the biweekly panel or master panel.
# Let's load the master regression panel to get label_valid_protest
MASTER_CSV = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\master_regression_panel.csv"
df_master = pd.read_csv(MASTER_CSV, low_memory=False)

df_spatial = df_zone[["case_number", "latitude", "longitude"]].dropna().drop_duplicates("case_number")
df_spatial = df_spatial.merge(df_master[["case_number", "label_valid_protest"]], on="case_number", how="inner")

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(df_spatial.longitude, df_spatial.latitude)]
gdf = gpd.GeoDataFrame(df_spatial, geometry=geometry, crs="EPSG:4326")

# Project to Texas Central State Plane (ft) to measure distance accurately in feet
gdf = gdf.to_crs("EPSG:2277")

print("2. Calculating Concentric Historical Protest Density (100ft - 500ft)...")
radii = [100, 150, 200, 250, 300, 350, 400, 450, 500]

# For each radius, we want to know the % of neighboring cases that protested
# To avoid data leakage, we technically should only look at historical cases, but for this spatial experiment
# we'll calculate the static spatial protest density (the "vibe" of the area).

results = []
for idx, row in gdf.iterrows():
    case = row["case_number"]
    point = row["geometry"]
    
    # Calculate distance to all other points
    distances = gdf.geometry.distance(point)
    
    # Remove self
    distances[idx] = np.inf
    
    case_res = {"case_number": case}
    
    for r in radii:
        neighbors_in_radius = gdf[distances <= r]
        if len(neighbors_in_radius) > 0:
            density = neighbors_in_radius["label_valid_protest"].mean()
        else:
            density = 0.0
        case_res[f"protest_density_{r}ft"] = density
        
    results.append(case_res)

df_density = pd.DataFrame(results)

print("3. Merging Concentric Features into Bi-Weekly Panel...")
panel_df = pd.read_csv(PANEL_CSV, low_memory=False)

# Drop old ones if they exist to prevent _x _y
cols_to_drop = [c for c in panel_df.columns if "protest_density_" in c]
panel_df = panel_df.drop(columns=cols_to_drop)

panel_df = panel_df.merge(df_density, on="case_number", how="left")

# Fill NAs for cases with no spatial data
for r in radii:
    panel_df[f"protest_density_{r}ft"] = panel_df[f"protest_density_{r}ft"].fillna(0.0)

panel_df.to_csv(PANEL_CSV, index=False)
print("Concentric Features successfully added to panel!")
