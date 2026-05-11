"""
08j_generate_modern_LUI_baseline.py

Generates the ultimate 3D Citywide Causal Surface baseline matrix (`X_base`) 
using the modern Land Use Inventory (LUI). 
Since the LUI lacks financial intrinsic data, this script performs a massive spatial 
fusion, joining the LUI against the Land Database (LDB) to inherit property values, 
and against the ACS Census Tracts to inherit demographics.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
import joblib
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')

ROOT = Path(r"c:\Users\dhl\data\Thesis\thesis")

# ── 1. Load Modern LUI Geometries ─────────────────────────────────────────
print("1. Loading Modern Land Use Inventory (LUI)...", flush=True)
lui_path = ROOT / "Data/CoA_Open_Data/Land_Use_Inventory_Detailed.geojson"
lui_gdf = gpd.read_file(lui_path)
if lui_gdf.crs and lui_gdf.crs != "EPSG:4326":
    lui_gdf = lui_gdf.to_crs("EPSG:4326")

lui_gdf = lui_gdf[lui_gdf.geometry.notna()]
lui_gdf = lui_gdf[~lui_gdf.geometry.is_empty]

# Use centroids for fast point-in-polygon spatial joins
centroids_gdf = gpd.GeoDataFrame(geometry=lui_gdf.geometry.centroid, crs="EPSG:4326")
lui_gdf['latitude'] = centroids_gdf.geometry.y
lui_gdf['longitude'] = centroids_gdf.geometry.x

# ── 2. Spatial Fusion: Inherit Financials from LDB ──────────────────────
print("2. Fusing with Land Database (LDB) to inherit Appraised Values...", flush=True)
ldb_path = ROOT / "Data/CoA_Open_Data/Land_Database_2021.geojson"
ldb_gdf = gpd.read_file(ldb_path, columns=['appraised_', 'market_val', 'yr_built', 'geometry'])
if ldb_gdf.crs and ldb_gdf.crs != "EPSG:4326":
    ldb_gdf = ldb_gdf.to_crs("EPSG:4326")

if 'appraised_' in ldb_gdf.columns:
    ldb_gdf['appraised_value'] = pd.to_numeric(ldb_gdf['appraised_'], errors='coerce')
else:
    ldb_gdf['appraised_value'] = pd.to_numeric(ldb_gdf.get('market_val', 500000), errors='coerce')

ldb_gdf['yr_built'] = pd.to_numeric(ldb_gdf['yr_built'], errors='coerce')
ldb_gdf['building_age'] = 2021 - ldb_gdf['yr_built']
ldb_gdf['building_age'] = ldb_gdf['building_age'].clip(lower=0)

ldb_subset = ldb_gdf[['appraised_value', 'building_age', 'geometry']]

# Spatial join: For each LUI centroid, find the LDB polygon it falls inside
joined_ldb = gpd.sjoin(centroids_gdf, ldb_subset, how='left', predicate='within')
# Drop duplicates if a centroid falls exactly on a boundary
joined_ldb = joined_ldb[~joined_ldb.index.duplicated(keep='first')]

lui_gdf['appraised_value'] = joined_ldb['appraised_value']
lui_gdf['building_age'] = joined_ldb['building_age']

# ── 3. Spatial Fusion: Inherit Modern ACS Demographics ──────────────────
print("3. Fetching 2021 Census Tract Geometries...", flush=True)
tract_url = "https://www2.census.gov/geo/tiger/TIGER2021/TRACT/tl_2021_48_tract.zip"
try:
    tracts_gdf = gpd.read_file(tract_url)
    tracts_gdf = tracts_gdf.to_crs("EPSG:4326")
    bbox = lui_gdf.total_bounds
    tracts_gdf = tracts_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
except Exception as e:
    print(f"Error loading tracts: {e}")
    exit(1)

print("4. Spatial Joining LUI to Tracts...", flush=True)
tract_id_col = 'GEOID' if 'GEOID' in tracts_gdf.columns else 'GEOID20'
joined_acs = gpd.sjoin(centroids_gdf, tracts_gdf[[tract_id_col, 'geometry']], how='left', predicate='within')
lui_gdf['census_tract'] = joined_acs[tract_id_col].astype(str)

print("5. Merging 2021 ACS Data...", flush=True)
acs_path = ROOT / "Data/Panel/census/acs_tract_timeseries.csv"
acs_df = pd.read_csv(acs_path)
acs_2021 = acs_df[acs_df['vintage'] == 2021].copy()
acs_2021['census_tract'] = acs_2021['geoid_tract'].astype(str)

acs_2021['total_units'] = acs_2021['owner_occupied_units'].fillna(0) + acs_2021['renter_occupied_units'].fillna(0)
acs_2021['renter_share'] = acs_2021['renter_occupied_units'] / acs_2021['total_units'].replace(0, 1)

pop_safe = acs_2021['total_population'].replace(0, 1)
acs_2021['race_white'] = acs_2021['race_white'].fillna(0) / pop_safe
acs_2021['race_black'] = acs_2021['race_black'].fillna(0) / pop_safe
acs_2021['race_hispanic'] = acs_2021['race_hispanic'].fillna(0) / pop_safe

inc_safe = acs_2021['median_household_income'].replace(0, 1).fillna(1)
acs_2021['rent_burden'] = (acs_2021['median_gross_rent'].fillna(0) * 12) / inc_safe

acs_2021_features = acs_2021[[
    'census_tract', 'median_household_income', 'race_white', 'race_black', 'race_hispanic',
    'renter_share', 'rent_burden', 'total_population', 'median_age'
]]

lui_gdf = lui_gdf.merge(acs_2021_features, on='census_tract', how='left')

# ── 4. Set Macroeconomics and Spatial Contagion ───────────────────────────
print("6. Setting 2021 Macroeconomic baselines...", flush=True)
lui_gdf['mortgage_rate_30yr'] = 2.96  
lui_gdf['fed_funds_rate'] = 0.08      
lui_gdf['local_unemployment_rate'] = 4.3  

print("7. Interpolating Spatial Contagion metrics...", flush=True)
cs_path = ROOT / "Data/Panel/cross_sectional_dml_panel.csv"
cs = pd.read_csv(cs_path)
cs_geo = cs.dropna(subset=['latitude', 'longitude', 'knn_petition_rate_1km', 'dist_petition_rate_lag1'])

knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn.fit(cs_geo[['latitude', 'longitude']].values, cs_geo[['knn_petition_rate_1km', 'dist_petition_rate_lag1']].values)

contagion_preds = knn.predict(lui_gdf[['latitude', 'longitude']].fillna(0).values)
lui_gdf['knn_petition_rate_1km'] = contagion_preds[:, 0]
lui_gdf['dist_petition_rate_lag1'] = contagion_preds[:, 1]

# Set baseline petition/protest mechanics (Zero-Treatment Counterfactual)
lui_gdf['Delta_Requested_Height'] = 29.0  
lui_gdf['cumulative_min_signer_dist'] = 0.0
lui_gdf['cumulative_signers_outside_200ft'] = 0.0
lui_gdf['cumulative_protester_embed_dim1'] = 0.0
lui_gdf['cumulative_protester_embed_dim2'] = 0.0
lui_gdf['cumulative_petition_attempted'] = 0.0
lui_gdf['cumulative_mobilization_failure'] = 0.0

# ── 4b. Spatial Fusion: WUI and Imagine Austin Corridors ──────────────────
print("7b. Applying structural spatial constraints (WUI & Corridors)...", flush=True)
try:
    # 1. WUI
    wui_path = ROOT / "Data/CoA_Open_Data/BOUNDARIES_wildland_urban_interface_code.geojson"
    wui_gdf = gpd.read_file("https://data.austintexas.gov/api/geospatial/ti8v-kzst?method=export&format=GeoJSON") if not wui_path.exists() else gpd.read_file(wui_path)
    if wui_gdf.crs and wui_gdf.crs != "EPSG:4326": wui_gdf = wui_gdf.to_crs("EPSG:4326")
    
    wui_cols = [c for c in wui_gdf.columns if c.lower() in ['fire_hazard_severity', 'slope_degree', 'geometry']]
    joined_wui = gpd.sjoin(centroids_gdf, wui_gdf[wui_cols], how='left', predicate='within')
    joined_wui = joined_wui[~joined_wui.index.duplicated(keep='first')]
    
    fh_col = next((c for c in joined_wui.columns if 'fire_hazard' in c.lower()), None)
    if fh_col:
        fh_series = joined_wui[fh_col].astype(str).str.title()
        fh_map = {'Low': 1, 'Moderate': 2, 'Medium': 2, 'High': 3, 'Extreme': 4}
        lui_gdf['fire_hazard_severity'] = fh_series.map(fh_map).fillna(0.0)
    else:
        lui_gdf['fire_hazard_severity'] = 0.0

    slope_col = next((c for c in joined_wui.columns if 'slope' in c.lower()), None)
    if slope_col:
        lui_gdf['slope_degree'] = pd.to_numeric(joined_wui[slope_col], errors='coerce').fillna(0.0)
    else:
        lui_gdf['slope_degree'] = 0.0

    # 2. Corridors
    corr_path = ROOT / "Data/CoA_Open_Data/Imagine_Austin_Corridors.geojson"
    corr_gdf = gpd.read_file("https://data.austintexas.gov/api/geospatial/gsvs-ypi7?method=export&format=GeoJSON") if not corr_path.exists() else gpd.read_file(corr_path)
    if corr_gdf.crs and corr_gdf.crs != "EPSG:4326": corr_gdf = corr_gdf.to_crs("EPSG:4326")
    
    if corr_gdf.geometry.type.isin(['LineString', 'MultiLineString']).any():
        corr_gdf = corr_gdf.to_crs("EPSG:3857")
        corr_gdf.geometry = corr_gdf.geometry.buffer(50)
        corr_gdf = corr_gdf.to_crs("EPSG:4326")

    joined_corr = gpd.sjoin(centroids_gdf, corr_gdf[['geometry']], how='left', predicate='within')
    joined_corr = joined_corr[~joined_corr.index.duplicated(keep='first')]
    lui_gdf['is_imagine_corridor'] = joined_corr['index_right'].notna().astype(float)

except Exception as e:
    print(f"Failed to spatial join WUI/Corridors: {e}")
    lui_gdf['fire_hazard_severity'] = 0.0
    lui_gdf['slope_degree'] = 0.0
    lui_gdf['is_imagine_corridor'] = 0.0

# ── 5. Hurdle Propensity Link ─────────────────────────────────────────────
print("8. Calculating Baseline Withdrawal Propensity...", flush=True)
confounders = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'race_black', 'race_hispanic',
    'renter_share', 'rent_burden', 'total_population', 'median_age',
    'appraised_value', 'building_age',
    'mortgage_rate_30yr', 'fed_funds_rate', 'local_unemployment_rate',
    'knn_petition_rate_1km', 'dist_petition_rate_lag1',
    'cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
    'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
    'cumulative_petition_attempted', 'cumulative_mobilization_failure',
    'fire_hazard_severity', 'slope_degree', 'is_imagine_corridor'
]

# Impute lingering NaNs (due to geometries falling outside LDB/Tract bounds)
for c in confounders:
    lui_gdf[c] = lui_gdf[c].fillna(lui_gdf[c].median())

X_base_raw = lui_gdf[confounders].values

models_path = ROOT / "Data/Zoning_Cases/causal_models.pkl"
try:
    models = joblib.load(models_path)
    survival_clf = models.get('survival_clf', None)
    if survival_clf:
        P_withdraw = survival_clf.predict_proba(X_base_raw)[:, 1]
    else:
        P_withdraw = np.zeros(len(lui_gdf))
except:
    P_withdraw = np.zeros(len(lui_gdf))

lui_gdf['P_withdraw'] = P_withdraw
joint_confounders = confounders + ['P_withdraw']

X_base = lui_gdf[joint_confounders].values

# ── 6. Save Artifacts ──────────────────────────────────────────────────────
X_out = ROOT / "Data/Zoning_Cases/X_base.npy"
print(f"Saving pristine LUI X_base array to {X_out}...", flush=True)
np.save(X_out, X_base)

geom_out = ROOT / "Data/Zoning_Cases/austin_base_geometries.fgb"
print(f"Exporting LUI geometry to {geom_out}...", flush=True)

ZONE_HEIGHT = {
    'SF':5,'MF':60,'MH':20,'RR':20,'LA':20,'LR':30,'LO':40,'GO':60,'MO':60,
    'GR':60,'CS':60,'CR':60,'DMU':90,'CBD':180,'P':60,'CH':90,'TOD':90,
    'VMU':60,'LI':50,'MI':50,'HI':60,'IP':60,'R&D':60,'PUD':90,'ETOD':90,
}
def map_height(z):
    if not isinstance(z, str): return 30
    z_upper = z.strip().upper()
    for key, h in ZONE_HEIGHT.items():
        if z_upper.startswith(key): return h
    return 30

# LUI uses 'land_use' or 'general_land_use', map rough height limits
lui_gdf['basezone_h'] = lui_gdf['land_use'].apply(map_height)
gdf_geom = lui_gdf[['parcel_id_10', 'land_use', 'basezone_h', 'geometry']].copy()
# Rename to match legacy downstream pipelines
gdf_geom.rename(columns={'parcel_id_10': 'prop_id', 'land_use': 'basezone'}, inplace=True)

gdf_geom.to_file(geom_out, driver='FlatGeobuf')
print("Modern LUI Baseline Generation Complete!", flush=True)
