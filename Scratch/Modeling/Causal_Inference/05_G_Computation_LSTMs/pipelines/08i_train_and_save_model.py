"""
08i_train_and_save_model.py

Trains the Causal Forest using CONTINUOUS treatment so we can dynamically
evaluate the CATE for any arbitrary Petition Dose on the fly.
Saves the trained models, the base X matrix, and the geometry FlatGeobuf.
"""

import pandas as pd
import numpy as np
import warnings
import joblib
warnings.filterwarnings('ignore')
from pathlib import Path
from econml.dml import CausalForestDML
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsRegressor
import geopandas as gpd

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    model_y_multi = CatBoostRegressor(iterations=200, depth=5, loss_function='MultiRMSE', verbose=0)
    model_t_cont = CatBoostRegressor(iterations=200, depth=5, verbose=0)
    model_y_bin = CatBoostRegressor(iterations=200, depth=5, verbose=0)
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    model_y_multi = MultiOutputRegressor(GradientBoostingRegressor(max_depth=4, n_estimators=100))
    model_t_cont = GradientBoostingRegressor(max_depth=4, n_estimators=100)
    model_y_bin = GradientBoostingRegressor(max_depth=4, n_estimators=100)

ROOT = Path(r"c:\Users\dhl\data\Thesis\thesis")

# ── 1. Load Historical Data ──────────────────────────────────
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

# ── 2. Fit CONTINUOUS Causal Forest ──────────────────────────────────────
print("Fitting Continuous Causal Forest on historical cases...", flush=True)
X = cs[confounders].values
T_cont = cs['petition_dose'].values # Continuous treatment!

surv_mask = ~cs['detailed_status'].isin(['Withdrawn', 'Denied', 'Expired', 'VOID'])
cs_surv = cs[surv_mask]
X_surv = cs_surv[confounders].values
Y_surv_joint = cs_surv[['Height_Attrition', 'days_to_resolution']].values
T_cont_surv = cs_surv['petition_dose'].values
Y_withd = cs['Withdrawal_Binary'].values

# Notice discrete_treatment=False so we can dynamically query T1=dose
cf_joint = CausalForestDML(
    model_y=model_y_multi, model_t=model_t_cont,
    discrete_treatment=False, n_estimators=100,
    cv=KFold(n_splits=2), random_state=42
)
cf_withd = CausalForestDML(
    model_y=model_y_bin, model_t=model_t_cont,
    discrete_treatment=False, n_estimators=100,
    cv=KFold(n_splits=2), random_state=42
)

cf_joint.fit(Y_surv_joint, T_cont_surv, X=X_surv)
cf_withd.fit(Y_withd, T_cont, X=X)

print("Training KNN Demographic Interpolator...", flush=True)
knn_X = cs[['latitude', 'longitude']].values
knn_Y = cs[['median_household_income', 'race_white', 'renter_share']].values
knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn.fit(knn_X, knn_Y)

# ── 3. Prepare 300k X_base Matrix ──────────────────────────────────────────
print("Loading 300k+ Land Database Parcels...", flush=True)
ldb_path = ROOT / "Data/CoA_Open_Data/Land_Database_2021.geojson"
gdf = gpd.read_file(ldb_path)
if gdf.crs and gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

centroids = gdf.geometry.centroid
gdf['latitude'] = centroids.y
gdf['longitude'] = centroids.x
gdf = gdf.dropna(subset=['latitude', 'longitude'])

print("Interpolating demographics for 300k parcels...", flush=True)
demo_preds = knn.predict(gdf[['latitude', 'longitude']].values)
gdf['median_household_income'] = demo_preds[:, 0]
gdf['race_white'] = demo_preds[:, 1]
gdf['renter_share'] = demo_preds[:, 2]

gdf['Delta_Requested_Height'] = 29.0
gdf['cumulative_min_signer_dist'] = 0.0
gdf['cumulative_signers_outside_200ft'] = 0.0
gdf['cumulative_protester_embed_dim1'] = 0.0
gdf['cumulative_protester_embed_dim2'] = 0.0
gdf['cumulative_petition_attempted'] = 0.0
gdf['cumulative_mobilization_failure'] = 0.0

X_base = gdf[confounders].values

# ── 4. Save Artifacts ──────────────────────────────────────────────────────
models_out = ROOT / "Data/Zoning_Cases/causal_models.pkl"
print(f"Saving models to {models_out}...", flush=True)
joblib.dump({
    'cf_joint': cf_joint,
    'cf_withd': cf_withd
}, models_out)

X_out = ROOT / "Data/Zoning_Cases/X_base.npy"
print(f"Saving X_base array to {X_out}...", flush=True)
np.save(X_out, X_base)

geom_out = ROOT / "Data/Zoning_Cases/austin_base_geometries.fgb"
print(f"Exporting geometry with basezone to {geom_out}...", flush=True)

# Austin LDC basezone → approximate height limit (ft)
ZONE_HEIGHT = {
    # Single Family
    'SF-1':5,'SF-1A':5,'SF-2':5,'SF-2A':5,'SF-3':5,'SF-4A':5,'SF-4B':5,'SF-5':5,'SF-6':5,
    # Multifamily
    'MF-1':40,'MF-2':40,'MF-3':60,'MF-4':60,'MF-5':75,'MF-6':90,
    # Mixed Use / Commercial
    'MH':20,'RR':20,'LA':20,'LR':30,
    'LO':40,'GO':60,'MO':60,
    'GR':60,'CS':60,'CS-1':75,'CR':60,
    # Downtown / Urban
    'DMU':90,'CBD':180,'P':60,
    'CH':90,'TOD':90,'VMU':60,'VMU2':75,
    'W/LO':40,'W/GO':60,
    # Industrial / Warehouse
    'LI':50,'MI':50,'HI':60,
    'IP':60,'R&D':60,
    # Civic / Special
    'PUD':90,'ETOD':90,
}

def map_height(z):
    if not isinstance(z, str):
        return 30  # default
    z_upper = z.strip().upper()
    for key, h in ZONE_HEIGHT.items():
        if z_upper.startswith(key.upper()):
            return h
    return 30  # default for unknown zones

gdf['basezone_h'] = gdf['basezone'].apply(map_height)
gdf_geom = gdf[['prop_id', 'basezone', 'basezone_h', 'geometry']].copy()
gdf_geom.to_file(geom_out, driver='FlatGeobuf')

print("Phase 1 Persistence complete!", flush=True)
