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

print("Loading cleaned cross-sectional panel...", flush=True)
cs_path = ROOT / "Data/Panel/cross_sectional_dml_panel.csv"
cs = pd.read_csv(cs_path)

demo_cols = ['median_household_income', 'race_white', 'race_black', 'race_hispanic', 
             'renter_share', 'rent_burden', 'total_population', 'median_age',
             'appraised_value', 'building_age', 'mortgage_rate_30yr', 
             'fed_funds_rate', 'local_unemployment_rate',
             'knn_petition_rate_1km', 'dist_petition_rate_lag1']

confounders = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'race_black', 'race_hispanic',
    'renter_share', 'rent_burden', 'total_population', 'median_age',
    'appraised_value', 'building_age',
    'mortgage_rate_30yr', 'fed_funds_rate', 'local_unemployment_rate',
    'knn_petition_rate_1km', 'dist_petition_rate_lag1',
    'cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
    'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
    'cumulative_petition_attempted', 'cumulative_mobilization_failure'
]

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
knn_Y = cs[demo_cols].values
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
for i, col in enumerate(demo_cols):
    gdf[col] = demo_preds[:, i]

# Set baseline petition/protest mechanics (what if zero protest?)
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
