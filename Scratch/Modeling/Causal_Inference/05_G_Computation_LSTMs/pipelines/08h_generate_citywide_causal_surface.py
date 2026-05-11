"""
08h_generate_citywide_causal_surface.py

Trains the Causal Forest on the 3,416 historical zoning cases,
interpolates spatial demographics to all 300,000+ Austin parcels,
and runs a simulated inference for the entire city.
Outputs a FlatGeobuf file for ultra-fast WebGL rendering.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from econml.dml import CausalForestDML
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
import geopandas as gpd

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    model_y_multi = CatBoostRegressor(iterations=200, depth=5, loss_function='MultiRMSE', verbose=0)
    model_t = CatBoostClassifier(iterations=200, depth=5, verbose=0)
    model_y_bin = CatBoostRegressor(iterations=200, depth=5, verbose=0)
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.multioutput import MultiOutputRegressor
    model_y_multi = MultiOutputRegressor(GradientBoostingRegressor(max_depth=4, n_estimators=100))
    model_t = GradientBoostingClassifier(max_depth=4, n_estimators=100)
    model_y_bin = GradientBoostingRegressor(max_depth=4, n_estimators=100)

ROOT = Path(r"c:\Users\dhl\data\Thesis\thesis")
DOSE_THRESHOLD = 0.20
SIMULATED_HEIGHT = 29.0

# ── 1. Load Historical Data & Train Model ──────────────────────────────────
print("Loading historical panel...", flush=True)
panel_path = ROOT / "Data/Panel/biweekly_panel_patched.csv"
if not panel_path.exists():
    panel_path = ROOT / "Data/Panel/biweekly_panel.csv"
df = pd.read_csv(panel_path, low_memory=False)

zoning_df = pd.read_csv(ROOT / "Data/Zoning_Cases/Processed_Data/CSV/zoning_land_use_merged_data.csv", low_memory=False)
zoning_df['start'] = pd.to_datetime(zoning_df['application_start_date'], errors='coerce')
zoning_df['end'] = pd.to_datetime(zoning_df['status_date'], errors='coerce')
zoning_df['days_to_resolution'] = (zoning_df['end'] - zoning_df['start']).dt.days.clip(0, 3650)
zoning_dates = zoning_df[['case_number', 'days_to_resolution']].drop_duplicates('case_number')

status_df = pd.read_csv(ROOT / "Data/Zoning_Cases/Processed_Data/CSV/zoning_case_statuses.csv", low_memory=False)

print("Collapsing panel...", flush=True)
cs = df.groupby('case_number').agg({
    'cumulative_unofficial_protest_intensity': 'max',
    'Delta_Approved_Height': 'last',
    'Delta_Requested_Height': 'last',
    'latitude': 'first',
    'longitude': 'first',
    'median_household_income': 'first',
    'race_white': 'first',
    'renter_share': 'first',
    'cumulative_min_signer_dist': 'max',
    'cumulative_signers_outside_200ft': 'max',
    'cumulative_protester_embed_dim1': 'max',
    'cumulative_protester_embed_dim2': 'max',
    'cumulative_petition_attempted': 'max',
    'cumulative_mobilization_failure': 'max'
}).reset_index()

mask_withdrawn = cs['Delta_Requested_Height'].notna() & cs['Delta_Approved_Height'].isna()
cs.loc[mask_withdrawn, 'Delta_Approved_Height'] = 0

cs = pd.merge(cs, zoning_dates, on='case_number', how='left')
cs = pd.merge(cs, status_df[['case_number', 'detailed_status']], on='case_number', how='left')

def fraction_01(s):
    x = pd.to_numeric(s, errors='coerce').fillna(0.0)
    non_zero = x[x > 0]
    if len(non_zero) > 0 and non_zero.quantile(0.99) > 1.0:
        x = x / 100.0
    return x.clip(0.0, 1.0)

cs['petition_dose'] = fraction_01(cs['cumulative_unofficial_protest_intensity'])
cs['Height_Attrition'] = cs['Delta_Requested_Height'] - cs['Delta_Approved_Height']
cs['Withdrawal_Binary'] = (cs['detailed_status'] == 'Withdrawn').astype(float)

for c in ['median_household_income', 'race_white', 'renter_share']:
    cs[c] = cs[c].fillna(cs[c].median())
for c in ['cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
          'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
          'cumulative_petition_attempted', 'cumulative_mobilization_failure']:
    cs[c] = cs[c].fillna(0.0)

confounders = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'renter_share',
    'cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
    'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
    'cumulative_petition_attempted', 'cumulative_mobilization_failure'
]

cs = cs.dropna(subset=confounders + ['Delta_Approved_Height', 'Height_Attrition', 'petition_dose', 'days_to_resolution', 'latitude', 'longitude'])

print("Fitting Causal Forest on historical cases...", flush=True)
X = cs[confounders].values
D = cs['petition_dose'].values
D_bin = (D >= DOSE_THRESHOLD).astype(float)

surv_mask = ~cs['detailed_status'].isin(['Withdrawn', 'Denied', 'Expired', 'VOID'])
cs_surv = cs[surv_mask]
X_surv = cs_surv[confounders].values
Y_surv_joint = cs_surv[['Height_Attrition', 'days_to_resolution']].values
D_bin_surv = (cs_surv['petition_dose'] >= DOSE_THRESHOLD).astype(float).values
Y_withd = cs['Withdrawal_Binary'].values

cf_joint = CausalForestDML(
    model_y=model_y_multi, model_t=model_t,
    discrete_treatment=True, n_estimators=100,
    cv=StratifiedKFold(n_splits=2), random_state=42
)
cf_withd = CausalForestDML(
    model_y=model_y_bin, model_t=model_t,
    discrete_treatment=True, n_estimators=100,
    cv=StratifiedKFold(n_splits=2), random_state=42
)

cf_joint.fit(Y_surv_joint, D_bin_surv, X=X_surv)
cf_withd.fit(Y_withd, D_bin, X=X)

# ── 2. Train Spatial KNN Interpolator ──────────────────────────────────────
print("Training KNN Demographic Interpolator...", flush=True)
# Predict [income, race, renter] from [lat, lon]
knn_X = cs[['latitude', 'longitude']].values
knn_Y = cs[['median_household_income', 'race_white', 'renter_share']].values
knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn.fit(knn_X, knn_Y)

# ── 3. Load 300k Parcels & Interpolate ─────────────────────────────────────
print("Loading 300k+ Land Database Parcels...", flush=True)
ldb_path = ROOT / "Data/CoA_Open_Data/Land_Database_2021.geojson"
gdf = gpd.read_file(ldb_path)

print(f"Loaded {len(gdf)} parcels. Converting CRS...", flush=True)
if gdf.crs and gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

print("Calculating parcel centroids...", flush=True)
centroids = gdf.geometry.centroid
gdf['latitude'] = centroids.y
gdf['longitude'] = centroids.x

# Drop parcels with invalid geometries/centroids
gdf = gdf.dropna(subset=['latitude', 'longitude'])

print("Interpolating demographics for all parcels...", flush=True)
demo_preds = knn.predict(gdf[['latitude', 'longitude']].values)
gdf['median_household_income'] = demo_preds[:, 0]
gdf['race_white'] = demo_preds[:, 1]
gdf['renter_share'] = demo_preds[:, 2]

# ── 4. Build Synthetic Features for Inference ──────────────────────────────
print("Building synthetic X matrix for inference...", flush=True)
gdf['Delta_Requested_Height'] = SIMULATED_HEIGHT
gdf['cumulative_min_signer_dist'] = 0.0
gdf['cumulative_signers_outside_200ft'] = 0.0
gdf['cumulative_protester_embed_dim1'] = 0.0
gdf['cumulative_protester_embed_dim2'] = 0.0
gdf['cumulative_petition_attempted'] = 0.0
gdf['cumulative_mobilization_failure'] = 0.0

X_city = gdf[confounders].values

# ── 5. Run City-Wide Inference ─────────────────────────────────────────────
print("Running city-wide Causal Forest inference...", flush=True)

# Chunk the inference to avoid any massive memory spikes, though 300k is usually fine
chunk_size = 50000
delay_preds = []
height_preds = []
withd_preds = []

for i in range(0, len(X_city), chunk_size):
    print(f"  Inference chunk {i} to {i + chunk_size}...")
    X_chunk = X_city[i:i+chunk_size]
    cate_multi = cf_joint.effect(X_chunk)
    cate_w = cf_withd.effect(X_chunk)
    
    height_preds.append(cate_multi[:, 0])
    delay_preds.append(cate_multi[:, 1])
    withd_preds.append(cate_w)

gdf['cate_height'] = np.clip(np.concatenate(height_preds), -500, 1500)
gdf['cate_delay']  = np.clip(np.concatenate(delay_preds), -365, 3650)
gdf['cate_withd']  = np.clip(np.concatenate(withd_preds), -1.0, 1.0)
gdf['is_dead'] = (gdf['cate_withd'] > 0.10).astype(int)

# Clean up output GeoDataFrame to keep it tiny and fast
cols_to_keep = ['prop_id', 'geometry', 'cate_height', 'cate_delay', 'cate_withd', 'is_dead']
# Keep only necessary columns
out_gdf = gdf[[c for c in cols_to_keep if c in gdf.columns]]

# ── 6. Export FlatGeobuf ───────────────────────────────────────────────────
out_fgb = ROOT / "Data/Zoning_Cases/austin_causal_surface.fgb"
print(f"Exporting to FlatGeobuf: {out_fgb}...", flush=True)
out_gdf.to_file(out_fgb, driver='FlatGeobuf')

print("Pipeline completed successfully!", flush=True)
