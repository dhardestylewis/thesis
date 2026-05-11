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
# STRICT ARTIFACT CAP: Clip delay to 5 years (1825 days). Anything longer is 
# an administrative ledger artifact (e.g. 30-year abandoned cases closed in mass purges)
zoning_df['days_to_resolution'] = (zoning_df['end'] - zoning_df['start']).dt.days.clip(0, 1825)
zoning_dates = zoning_df[['case_number', 'days_to_resolution']].drop_duplicates('case_number')

# Load the enriched causal CSV — has delta_max_height_ft + lat/lon for more cases
enriched_path = ROOT / "Data/Zoning_Cases/Processed_Data/CSV/enriched_zoning_data_causal.csv"
enriched_df = pd.read_csv(enriched_path, low_memory=False) if enriched_path.exists() else pd.DataFrame()

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

# Withdrawn cases → approved height = 0 (nothing was approved)
mask_withdrawn = cs['Delta_Requested_Height'].notna() & cs['Delta_Approved_Height'].isna()
cs.loc[mask_withdrawn, 'Delta_Approved_Height'] = 0

cs = pd.merge(cs, zoning_dates, on='case_number', how='left')
cs = pd.merge(cs, status_df[['case_number', 'detailed_status']], on='case_number', how='left')

# ── FIX 1: Backfill lat/lon from zoning CSV (63 cases) ──────────────────────
if 'latitude' in zoning_df.columns and 'longitude' in zoning_df.columns:
    geo_fallback = zoning_df[['case_number', 'latitude', 'longitude']].dropna().drop_duplicates('case_number')
    cs = cs.merge(geo_fallback.rename(columns={'latitude': '_lat_z', 'longitude': '_lon_z'}),
                  on='case_number', how='left')
    missing_geo = cs['latitude'].isna()
    cs.loc[missing_geo, 'latitude']  = cs.loc[missing_geo, '_lat_z']
    cs.loc[missing_geo, 'longitude'] = cs.loc[missing_geo, '_lon_z']
    n1 = (missing_geo & cs['latitude'].notna()).sum()
    cs.drop(columns=['_lat_z', '_lon_z'], inplace=True)
    print(f"FIX 1a: Backfilled lat/lon from zoning CSV for {n1:,} cases.", flush=True)

# ── FIX 1b: Backfill lat/lon from enriched_zoning_data_causal.csv (19 more) ─
if not enriched_df.empty and 'latitude' in enriched_df.columns:
    geo_enriched = enriched_df[['case_number', 'latitude', 'longitude']].dropna().drop_duplicates('case_number')
    cs = cs.merge(geo_enriched.rename(columns={'latitude': '_lat_e', 'longitude': '_lon_e'}),
                  on='case_number', how='left')
    missing_geo2 = cs['latitude'].isna()
    cs.loc[missing_geo2, 'latitude']  = cs.loc[missing_geo2, '_lat_e']
    cs.loc[missing_geo2, 'longitude'] = cs.loc[missing_geo2, '_lon_e']
    n1b = (missing_geo2 & cs['latitude'].notna()).sum()
    cs.drop(columns=['_lat_e', '_lon_e'], inplace=True)
    print(f"FIX 1b: Backfilled lat/lon from enriched_causal CSV for {n1b:,} more cases.", flush=True)

# ── FIX 1c: Backfill lat/lon from external geocoder (866 cases) ─────────────
geocoded_path = ROOT / "Data/Zoning_Cases/Processed_Data/CSV/geocoded_missing_cases.csv"
if geocoded_path.exists():
    geocoded_df = pd.read_csv(geocoded_path, low_memory=False)
    if 'lat_geocoded' in geocoded_df.columns:
        cs = cs.merge(geocoded_df[['case_number', 'lat_geocoded', 'lon_geocoded']].drop_duplicates('case_number'),
                      on='case_number', how='left')
        missing_geo3 = cs['latitude'].isna()
        cs.loc[missing_geo3, 'latitude']  = cs.loc[missing_geo3, 'lat_geocoded']
        cs.loc[missing_geo3, 'longitude'] = cs.loc[missing_geo3, 'lon_geocoded']
        n1c = (missing_geo3 & cs['latitude'].notna()).sum()
        cs.drop(columns=['lat_geocoded', 'lon_geocoded'], inplace=True)
        print(f"FIX 1c: Backfilled lat/lon from Geocoder for {n1c:,} more cases.", flush=True)

# ── FIX 2: Height recovery from enriched_zoning_data_causal.csv ─────────────
# Cases where Delta_Requested_Height is null may have height data in the
# enriched CSV under delta_max_height_ft / proposed_max_height_ft.
if not enriched_df.empty and 'delta_max_height_ft' in enriched_df.columns:
    height_enriched = enriched_df[['case_number', 'delta_max_height_ft',
                                    'proposed_max_height_ft', 'existing_max_height_ft']].drop_duplicates('case_number')
    cs = cs.merge(height_enriched.add_prefix('_e_').rename(columns={'_e_case_number': 'case_number'}),
                  on='case_number', how='left')
    # Fill missing Delta_Requested_Height from delta_max_height_ft
    missing_req = cs['Delta_Requested_Height'].isna()
    cs.loc[missing_req, 'Delta_Requested_Height'] = cs.loc[missing_req, '_e_delta_max_height_ft']
    n2a = (missing_req & cs['Delta_Requested_Height'].notna()).sum()
    # For those with proposed/existing but no delta, compute it
    still_missing = cs['Delta_Requested_Height'].isna()
    cs.loc[still_missing, 'Delta_Requested_Height'] = (
        cs.loc[still_missing, '_e_proposed_max_height_ft'] - cs.loc[still_missing, '_e_existing_max_height_ft']
    )
    n2b = (still_missing & cs['Delta_Requested_Height'].notna()).sum()
    cs.drop(columns=[c for c in cs.columns if c.startswith('_e_')], inplace=True)
    print(f"FIX 2: Recovered Delta_Requested_Height for {n2a+n2b:,} cases from enriched CSV.", flush=True)

# ── FIX 3: No-petition, no-height cases → zero-treatment controls ────────────
# If a case had no petition AND still no recorded height request, it's a
# pure control observation. Attrition = 0 (full approval), approved = requested.
no_petition = cs['cumulative_unofficial_protest_intensity'].fillna(0) == 0
no_height   = cs['Delta_Requested_Height'].isna()
zero_controls = no_petition & no_height
cs.loc[zero_controls, 'Delta_Requested_Height'] = 0.0
cs.loc[zero_controls, 'Delta_Approved_Height']  = cs.loc[zero_controls, 'Delta_Approved_Height'].fillna(0.0)
print(f"FIX 3: Set {zero_controls.sum():,} no-petition/no-height cases as zero-treatment controls.", flush=True)

# Recompute derived columns after all recovery
def fraction_01(s):
    x = pd.to_numeric(s, errors='coerce').fillna(0.0)
    non_zero = x[x > 0]
    if len(non_zero) > 0 and non_zero.quantile(0.99) > 1.0:
        x = x / 100.0
    return x.clip(0.0, 1.0)

cs['petition_dose']     = fraction_01(cs['cumulative_unofficial_protest_intensity'])
cs['Height_Attrition']  = cs['Delta_Requested_Height'] - cs['Delta_Approved_Height']
cs['Withdrawal_Binary'] = (cs['detailed_status'] == 'Withdrawn').astype(float)

# Zero-fill petition features (genuinely zero when no petition occurred)
for c in ['cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
          'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
          'cumulative_petition_attempted', 'cumulative_mobilization_failure']:
    cs[c] = cs[c].fillna(0.0)

# ── Spatially-aware KNN imputation for census demographics ──────────────────
demo_cols = ['median_household_income', 'race_white', 'renter_share']
cs_geo = cs.dropna(subset=['latitude', 'longitude'] + demo_cols)
if len(cs_geo) > 5:
    knn_imputer = KNeighborsRegressor(n_neighbors=min(5, len(cs_geo)), weights='distance')
    knn_imputer.fit(cs_geo[['latitude', 'longitude']].values, cs_geo[demo_cols].values)
    missing_demo = cs[demo_cols].isna().any(axis=1) & cs['latitude'].notna()
    if missing_demo.sum() > 0:
        cs.loc[missing_demo, demo_cols] = knn_imputer.predict(
            cs.loc[missing_demo, ['latitude', 'longitude']].values
        )
        print(f"KNN-imputed demographics for {missing_demo.sum():,} training cases.", flush=True)

confounders = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'renter_share',
    'cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
    'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
    'cumulative_petition_attempted', 'cumulative_mobilization_failure'
]

# Hard minimum: must have lat/lon (spatial context) and petition_dose (treatment)
cs = cs.dropna(subset=['latitude', 'longitude', 'petition_dose'])
cs = cs.dropna(subset=confounders)  # after all imputations, remaining nulls are unrecoverable

print(f"\nFinal case counts after all recovery:", flush=True)
print(f"  Total with geo + confounders:  {len(cs):,}", flush=True)
print(f"  With days_to_resolution:       {cs['days_to_resolution'].notna().sum():,}  (joint model)", flush=True)
print(f"  With height attrition:         {cs['Height_Attrition'].notna().sum():,}  (joint model)", flush=True)
print(f"  All cases (withdrawal model):  {len(cs):,}", flush=True)

# ── 2. Fit CONTINUOUS Causal Forest ──────────────────────────────────────
print("\nFitting Continuous Causal Forest on historical cases...", flush=True)
X = cs[confounders].values
T_cont = cs['petition_dose'].values

# ── FIX 4: Split joint vs withdrawal training sets ────────────────────────────
# Joint model (delay + attrition): requires resolved outcome — exclude pending cases
# Withdrawal model: all cases including pending (Withdrawal_Binary = 0 for pending)
surv_mask = (
    ~cs['detailed_status'].isin(['Withdrawn', 'Denied', 'Expired', 'VOID']) &
    cs['days_to_resolution'].notna() &
    cs['Height_Attrition'].notna()
)
cs_surv = cs[surv_mask]
X_surv = cs_surv[confounders].values
Y_surv_joint = cs_surv[['Height_Attrition', 'days_to_resolution']].values
T_cont_surv = cs_surv['petition_dose'].values
Y_withd = cs['Withdrawal_Binary'].values
print(f"  Joint model training N:      {len(cs_surv):,}", flush=True)
print(f"  Withdrawal model training N: {len(cs):,}", flush=True)


# Notice discrete_treatment=False so we can dynamically query T1=dose
cf_joint = CausalForestDML(
    model_y=model_y_multi, model_t=model_t_cont,
    discrete_treatment=False, n_estimators=1000, # Crank to 1000 for asymptote
    cv=KFold(n_splits=5), random_state=42 # 5-fold for stability
)
cf_withd = CausalForestDML(
    model_y=model_y_bin, model_t=model_t_cont,
    discrete_treatment=False, n_estimators=1000,
    cv=KFold(n_splits=5), random_state=42
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
