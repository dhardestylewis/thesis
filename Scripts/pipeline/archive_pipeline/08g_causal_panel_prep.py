"""
08g_prepare_causal_panel.py

Production version of the Causal Inference data engineering pipeline.
Collapses biweekly longitudinal panel to cross-sectional cases, backfills spatial coordinates,
intersects area-weighted demographics, and ensures strictly pre-treatment conditioning (X).
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')

# Root resolution for production Scripts/pipeline/ location
ROOT = Path(__file__).resolve().parents[2]

# ── 1. Load Historical Data ──────────────────────────────────
print("Loading historical panel...", flush=True)
panel_path = ROOT / "Data/Panel/biweekly_panel_patched.csv"
if not panel_path.exists():
    panel_path = ROOT / "Data/Panel/biweekly_panel.csv"
df = pd.read_csv(panel_path, low_memory=False)

zoning_df = pd.read_csv(ROOT / "Data/Zoning_Cases/Processed_Data/CSV/zoning_land_use_merged_data.csv", low_memory=False)

zoning_df['start'] = pd.to_datetime(zoning_df['application_start_date'], errors='coerce')
zoning_df['end_status'] = pd.to_datetime(zoning_df['status_date'], errors='coerce')
zoning_df['end_approval'] = pd.to_datetime(zoning_df['approval_date'], errors='coerce')
zoning_df['end_final'] = pd.to_datetime(zoning_df['final_date'], errors='coerce')

# Cascade to find the true resolution date
zoning_df['days_final'] = (zoning_df['end_final'] - zoning_df['start']).dt.days
zoning_df['days_approval'] = (zoning_df['end_approval'] - zoning_df['start']).dt.days
zoning_df['days_status'] = (zoning_df['end_status'] - zoning_df['start']).dt.days

# Condition 1: Use final_date if it's valid (between 0 and 1825 days)
cond_final = zoning_df['days_final'].between(0, 1825)
# Condition 2: Use approval_date if final_date is invalid, but approval_date is valid
cond_app = ~cond_final & zoning_df['days_approval'].between(0, 1825)

# Build the true delay column
zoning_df['days_to_resolution'] = zoning_df['days_status']
zoning_df.loc[cond_app, 'days_to_resolution'] = zoning_df.loc[cond_app, 'days_approval']
zoning_df.loc[cond_final, 'days_to_resolution'] = zoning_df.loc[cond_final, 'days_final']

# Finally, apply the 5-year strict statutory cap to everything that remains
zoning_df['days_to_resolution'] = zoning_df['days_to_resolution'].clip(0, 1825)
zoning_dates = zoning_df[['case_number', 'days_to_resolution', 'application_start_date']].drop_duplicates('case_number')

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
    'race_black': 'first',
    'race_hispanic': 'first',
    'renter_share': 'first',
    'rent_burden': 'first',
    'total_population': 'first',
    'median_age': 'first',
    'appraised_value': 'first',
    'building_age': 'first',
    'mortgage_rate_30yr': 'first',
    'fed_funds_rate': 'first',
    'local_unemployment_rate': 'first',
    'knn_petition_rate_1km': 'first',
    'dist_petition_rate_lag1': 'first',
    'cumulative_min_signer_dist': 'max',
    'cumulative_signers_outside_200ft': 'max',
    'cumulative_protester_embed_dim1': 'max',
    'cumulative_protester_embed_dim2': 'max',
    'cumulative_petition_attempted': 'max',
    'cumulative_mobilization_failure': 'max'
}).reset_index()

# ── INJECT AREA-WEIGHTED DEMOGRAPHICS ──────────────────────────────────────────
area_weighted_path = ROOT / "Data/Zoning_Cases/Processed_Data/CSV/area_weighted_demographics.csv"
if area_weighted_path.exists():
    aw_df = pd.read_csv(area_weighted_path)
    demo_cols = ['median_household_income', 'renter_share', 'race_white', 'total_population', 'median_age']
    aw_df = aw_df[['case_number'] + [c for c in demo_cols if c in aw_df.columns]]
    cs = cs.set_index('case_number')
    aw_df = aw_df.set_index('case_number')
    cs.update(aw_df)
    cs = cs.reset_index()
    print(f"Injected exact area-weighted demographics for {len(aw_df)} cases.", flush=True)

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

# ── FIX 1b: Backfill lat/lon from enriched_causal CSV (19 more) ────────────
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

# ── FIX 2: Height recovery from enriched_causal CSV ─────────────
if not enriched_df.empty and 'delta_max_height_ft' in enriched_df.columns:
    height_enriched = enriched_df[['case_number', 'delta_max_height_ft',
                                    'proposed_max_height_ft', 'existing_max_height_ft']].drop_duplicates('case_number')
    cs = cs.merge(height_enriched.add_prefix('_e_').rename(columns={'_e_case_number': 'case_number'}),
                  on='case_number', how='left')
    missing_req = cs['Delta_Requested_Height'].isna()
    cs.loc[missing_req, 'Delta_Requested_Height'] = cs.loc[missing_req, '_e_delta_max_height_ft']
    n2a = (missing_req & cs['Delta_Requested_Height'].notna()).sum()
    still_missing = cs['Delta_Requested_Height'].isna()
    cs.loc[still_missing, 'Delta_Requested_Height'] = (
        cs.loc[still_missing, '_e_proposed_max_height_ft'] - cs.loc[still_missing, '_e_existing_max_height_ft']
    )
    n2b = (still_missing & cs['Delta_Requested_Height'].notna()).sum()
    cs.drop(columns=[c for c in cs.columns if c.startswith('_e_')], inplace=True)
    print(f"FIX 2: Recovered Delta_Requested_Height for {n2a+n2b:,} cases from enriched CSV.", flush=True)

# ── FIX 3: No-petition, no-height cases → zero-treatment controls ────────────
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
cs['log_days_to_resolution'] = np.log1p(cs['days_to_resolution'])

# ── SPATIAL JOIN: WUI and Imagine Austin Corridors ──────────────────────────
import geopandas as gpd
print("Applying structural spatial constraints (WUI & Corridors)...", flush=True)
try:
    cs_gdf = gpd.GeoDataFrame(cs, geometry=gpd.points_from_xy(cs.longitude, cs.latitude), crs="EPSG:4326")
    
    # 1. WUI
    wui_path = ROOT / "Data/CoA_Open_Data/BOUNDARIES_wildland_urban_interface_code.geojson"
    wui_gdf = gpd.read_file("https://data.austintexas.gov/api/geospatial/ti8v-kzst?method=export&format=GeoJSON") if not wui_path.exists() else gpd.read_file(wui_path)
    if wui_gdf.crs and wui_gdf.crs != "EPSG:4326": wui_gdf = wui_gdf.to_crs("EPSG:4326")
    
    wui_cols = [c for c in wui_gdf.columns if c.lower() in ['fire_hazard_severity', 'slope_degree', 'geometry']]
    joined_wui = gpd.sjoin(cs_gdf, wui_gdf[wui_cols], how='left', predicate='within')
    joined_wui = joined_wui[~joined_wui.index.duplicated(keep='first')]
    
    fh_col = next((c for c in joined_wui.columns if 'fire_hazard' in c.lower()), None)
    if fh_col:
        fh_series = joined_wui[fh_col].astype(str).str.title()
        fh_map = {'Low': 1, 'Moderate': 2, 'Medium': 2, 'High': 3, 'Extreme': 4}
        cs['fire_hazard_severity'] = fh_series.map(fh_map).fillna(0.0)
    else:
        cs['fire_hazard_severity'] = 0.0

    slope_col = next((c for c in joined_wui.columns if 'slope' in c.lower()), None)
    if slope_col:
        cs['slope_degree'] = pd.to_numeric(joined_wui[slope_col], errors='coerce').fillna(0.0)
    else:
        cs['slope_degree'] = 0.0

    # 2. Corridors
    corr_path = ROOT / "Data/CoA_Open_Data/Imagine_Austin_Corridors.geojson"
    corr_gdf = gpd.read_file("https://data.austintexas.gov/api/geospatial/gsvs-ypi7?method=export&format=GeoJSON") if not corr_path.exists() else gpd.read_file(corr_path)
    if corr_gdf.crs and corr_gdf.crs != "EPSG:4326": corr_gdf = corr_gdf.to_crs("EPSG:4326")
    
    if corr_gdf.geometry.type.isin(['LineString', 'MultiLineString']).any():
        corr_gdf = corr_gdf.to_crs("EPSG:3857")
        corr_gdf.geometry = corr_gdf.geometry.buffer(50)
        corr_gdf = corr_gdf.to_crs("EPSG:4326")

    joined_corr = gpd.sjoin(cs_gdf, corr_gdf[['geometry']], how='left', predicate='within')
    joined_corr = joined_corr[~joined_corr.index.duplicated(keep='first')]
    cs['is_imagine_corridor'] = joined_corr['index_right'].notna().astype(float)

except Exception as e:
    print(f"Failed to spatial join WUI/Corridors: {e}")
    cs['fire_hazard_severity'] = 0.0
    cs['slope_degree'] = 0.0
    cs['is_imagine_corridor'] = 0.0


# ── Spatially-aware KNN imputation for census demographics ──────────────────
demo_cols = ['median_household_income', 'race_white', 'race_black', 'race_hispanic', 
             'renter_share', 'rent_burden', 'total_population', 'median_age',
             'appraised_value', 'building_age', 'mortgage_rate_30yr', 
             'fed_funds_rate', 'local_unemployment_rate',
             'knn_petition_rate_1km', 'dist_petition_rate_lag1']
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

# ── IDENTIFICATION SET (X) ──────────────────────────────────────────────────
# STRICT PRE-TREATMENT COVARIATES ONLY.
confounders = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'race_black', 'race_hispanic',
    'renter_share', 'rent_burden', 'total_population', 'median_age',
    'appraised_value', 'building_age',
    'mortgage_rate_30yr', 'fed_funds_rate', 'local_unemployment_rate',
    'knn_petition_rate_1km', 'dist_petition_rate_lag1',
    'fire_hazard_severity', 'slope_degree', 'is_imagine_corridor'
]

# RETENTION FOR DESCRIPTIVE ANALYSIS (NOT USED IN DML IDENTIFICATION)
mediators = [
    'cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
    'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
    'cumulative_petition_attempted', 'cumulative_mobilization_failure'
]

# Hard minimum: must have lat/lon and petition_dose
cs = cs.dropna(subset=['latitude', 'longitude', 'petition_dose'])
cs = cs.dropna(subset=confounders) 

print(f"\nFinal case counts after all recovery:", flush=True)
print(f"  Total with geo + confounders:  {len(cs):,}")
print(f"  With days_to_resolution:       {cs['days_to_resolution'].notna().sum():,} (joint model)")
print(f"  With height attrition:         {cs['Height_Attrition'].notna().sum():,} (joint model)")

out_path = ROOT / "Data/Panel/cross_sectional_dml_panel.csv"
print(f"Saving fully prepared cross-sectional panel to {out_path}...", flush=True)
cs.to_csv(out_path, index=False)
