import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
import geopandas as gpd

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]

PANEL_PATH = ROOT / "Data/Panel/biweekly_panel.csv"
GEOM_PATH = ROOT / "Data/Panel/exact_geometric_petition_intensity.csv"
SPATIAL_PATH = ROOT / "Data/Panel/spatial_attribution_2024.csv"
ZONING_PATH = ROOT / "Data/Zoning_Cases/Processed_Data/CSV/zoning_land_use_merged_data.csv"
ENRICHED_PATH = ROOT / "Data/Zoning_Cases/Processed_Data/CSV/enriched_zoning_data_causal.csv"
GEOCODED_PATH = ROOT / "Data/Zoning_Cases/Processed_Data/CSV/geocoded_missing_cases.csv"
AREA_WEIGHTED_PATH = ROOT / "Data/Zoning_Cases/Processed_Data/CSV/area_weighted_demographics.csv"

def get_base_case_year(panel):
    # To join temporally, we need the year of the case
    if 'year' in panel.columns:
        return panel[['case_number', 'year']].groupby('case_number').min().reset_index()
    return pd.DataFrame({'case_number': panel['case_number'].unique(), 'year': 2016})

def main():
    print(f"1. Loading Base Biweekly Panel from {PANEL_PATH}...", flush=True)
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["case_number"] = panel["case_number"].astype(str).str.strip()
    
    # Track cases
    unique_cases = panel['case_number'].nunique()
    print(f"   {panel.shape[0]} rows, {unique_cases} unique cases.", flush=True)
    
    # ── 1. Merge Geometric Petition Pct ──────────────────────────────────────
    print("2. Merging Exact Geometric Petition Intensity...", flush=True)
    if GEOM_PATH.exists():
        geom_df = pd.read_csv(GEOM_PATH)
        geom_df["case_number"] = geom_df["case_number"].astype(str).str.strip()
        if "label_exact_geometric_petition_pct" in panel.columns:
            panel = panel.drop(columns=["label_exact_geometric_petition_pct"])
        panel = panel.merge(geom_df[["case_number", "label_exact_geometric_petition_pct"]], on="case_number", how="left")
        panel["label_exact_geometric_petition_pct"] = panel["label_exact_geometric_petition_pct"].fillna(0)
    
    # ── 2. Merge Spatial Blight ──────────────────────────────────────────────
    print("3. Merging Spatial Blight Indices...", flush=True)
    if SPATIAL_PATH.exists():
        spatial_df = pd.read_csv(SPATIAL_PATH, low_memory=False)
        spatial_df["case_number"] = spatial_df["case_number"].astype(str).str.strip()
        spatial_df = spatial_df.drop_duplicates(subset=["case_number"])
        spatial_cols = ["archetype_pct_Architectural", "archetype_pct_Bureaucratic", "archetype_pct_Economic", "archetype_pct_Spatial_Gravity"]
        
        for c in spatial_cols:
            if c in panel.columns:
                panel = panel.drop(columns=[c])
                
        panel = panel.merge(spatial_df[["case_number"] + spatial_cols], on="case_number", how="left")
        for c in spatial_cols:
            panel[c] = panel[c].fillna(0)
            
    # ── 3. Geo-Coordinate Backfill ───────────────────────────────────────────
    print("4. Backfilling Spatial Coordinates...", flush=True)
    
    # Ensure latitude and longitude columns exist
    if 'latitude' not in panel.columns: panel['latitude'] = np.nan
    if 'longitude' not in panel.columns: panel['longitude'] = np.nan
    
    # Load fallback sources
    zoning_df = pd.read_csv(ZONING_PATH, low_memory=False) if ZONING_PATH.exists() else pd.DataFrame()
    enriched_df = pd.read_csv(ENRICHED_PATH, low_memory=False) if ENRICHED_PATH.exists() else pd.DataFrame()
    geocoded_df = pd.read_csv(GEOCODED_PATH, low_memory=False) if GEOCODED_PATH.exists() else pd.DataFrame()
    
    # 3a. Zoning CSV Fallback
    if not zoning_df.empty and 'latitude' in zoning_df.columns:
        geo_z = zoning_df[['case_number', 'latitude', 'longitude']].dropna().drop_duplicates('case_number')
        panel = panel.merge(geo_z.rename(columns={'latitude': '_lat_z', 'longitude': '_lon_z'}), on='case_number', how='left')
        missing = panel['latitude'].isna()
        panel.loc[missing, 'latitude'] = panel.loc[missing, '_lat_z']
        panel.loc[missing, 'longitude'] = panel.loc[missing, '_lon_z']
        panel.drop(columns=['_lat_z', '_lon_z'], inplace=True)
    
    # 3b. Enriched Causal CSV Fallback
    if not enriched_df.empty and 'latitude' in enriched_df.columns:
        geo_e = enriched_df[['case_number', 'latitude', 'longitude']].dropna().drop_duplicates('case_number')
        panel = panel.merge(geo_e.rename(columns={'latitude': '_lat_e', 'longitude': '_lon_e'}), on='case_number', how='left')
        missing = panel['latitude'].isna()
        panel.loc[missing, 'latitude'] = panel.loc[missing, '_lat_e']
        panel.loc[missing, 'longitude'] = panel.loc[missing, '_lon_e']
        panel.drop(columns=['_lat_e', '_lon_e'], inplace=True)
        
    # 3c. Geocoder Fallback
    if not geocoded_df.empty and 'lat_geocoded' in geocoded_df.columns:
        geo_g = geocoded_df[['case_number', 'lat_geocoded', 'lon_geocoded']].drop_duplicates('case_number')
        panel = panel.merge(geo_g, on='case_number', how='left')
        missing = panel['latitude'].isna()
        panel.loc[missing, 'latitude'] = panel.loc[missing, 'lat_geocoded']
        panel.loc[missing, 'longitude'] = panel.loc[missing, 'lon_geocoded']
        panel.drop(columns=['lat_geocoded', 'lon_geocoded'], inplace=True)

    cases_with_geo = panel.dropna(subset=['latitude', 'longitude'])['case_number'].nunique()
    print(f"   Geo-coordinates mapped for {cases_with_geo:,} / {unique_cases:,} cases.", flush=True)

    # ── 4. Area-Weighted Demographics ────────────────────────────────────────
    print("5. Injecting Pre-Treatment Area-Weighted Demographics...", flush=True)
    if AREA_WEIGHTED_PATH.exists():
        aw_df = pd.read_csv(AREA_WEIGHTED_PATH)
        demo_cols = ['median_household_income', 'renter_share', 'race_white', 'total_population', 'median_age']
        
        aw_df = aw_df[['case_number'] + [c for c in demo_cols if c in aw_df.columns]].drop_duplicates('case_number')
        
        # We update existing columns to prioritize the exact area-weighted values over point-based intersections
        for col in demo_cols:
            if col not in panel.columns:
                panel[col] = np.nan
                
        panel = panel.set_index('case_number')
        aw_df = aw_df.set_index('case_number')
        panel.update(aw_df)
        panel = panel.reset_index()

    # ── 5. Structural Spatial Joins (WUI / Corridors) ────────────────────────
    print("6. Executing Structural Spatial Joins (WUI & Corridors)...", flush=True)
    # We only need to do this once per case to save memory, then merge back
    cases_df = panel[['case_number', 'latitude', 'longitude']].drop_duplicates('case_number').dropna()
    try:
        cs_gdf = gpd.GeoDataFrame(cases_df, geometry=gpd.points_from_xy(cases_df.longitude, cases_df.latitude), crs="EPSG:4326")
        
        # 5a. WUI
        wui_path = ROOT / "Data/CoA_Open_Data/BOUNDARIES_wildland_urban_interface_code.geojson"
        wui_gdf = gpd.read_file(wui_path) if wui_path.exists() else gpd.read_file("https://data.austintexas.gov/api/geospatial/ti8v-kzst?method=export&format=GeoJSON")
        if wui_gdf.crs and wui_gdf.crs != "EPSG:4326": wui_gdf = wui_gdf.to_crs("EPSG:4326")
        
        wui_cols = [c for c in wui_gdf.columns if c.lower() in ['fire_hazard_severity', 'slope_degree', 'geometry']]
        joined_wui = gpd.sjoin(cs_gdf, wui_gdf[wui_cols], how='left', predicate='within')
        joined_wui = joined_wui[~joined_wui.index.duplicated(keep='first')]
        
        fh_col = next((c for c in joined_wui.columns if 'fire_hazard' in c.lower()), None)
        if fh_col:
            fh_series = joined_wui[fh_col].astype(str).str.title()
            fh_map = {'Low': 1, 'Moderate': 2, 'Medium': 2, 'High': 3, 'Extreme': 4}
            cases_df['fire_hazard_severity'] = fh_series.map(fh_map).fillna(0.0)
        else:
            cases_df['fire_hazard_severity'] = 0.0

        slope_col = next((c for c in joined_wui.columns if 'slope' in c.lower()), None)
        if slope_col:
            cases_df['slope_degree'] = pd.to_numeric(joined_wui[slope_col], errors='coerce').fillna(0.0)
        else:
            cases_df['slope_degree'] = 0.0

        # 5b. Corridors
        corr_path = ROOT / "Data/CoA_Open_Data/Imagine_Austin_Corridors.geojson"
        corr_gdf = gpd.read_file(corr_path) if corr_path.exists() else gpd.read_file("https://data.austintexas.gov/api/geospatial/gsvs-ypi7?method=export&format=GeoJSON")
        if corr_gdf.crs and corr_gdf.crs != "EPSG:4326": corr_gdf = corr_gdf.to_crs("EPSG:4326")
        
        if corr_gdf.geometry.type.isin(['LineString', 'MultiLineString']).any():
            corr_gdf = corr_gdf.to_crs("EPSG:3857")
            corr_gdf.geometry = corr_gdf.geometry.buffer(50)
            corr_gdf = corr_gdf.to_crs("EPSG:4326")

        joined_corr = gpd.sjoin(cs_gdf, corr_gdf[['geometry']], how='left', predicate='within')
        joined_corr = joined_corr[~joined_corr.index.duplicated(keep='first')]
        cases_df['is_imagine_corridor'] = joined_corr['index_right'].notna().astype(float)
        
        # Merge back to full panel
        for col in ['fire_hazard_severity', 'slope_degree', 'is_imagine_corridor']:
            if col in panel.columns: panel.drop(columns=[col], inplace=True)
        panel = panel.merge(cases_df[['case_number', 'fire_hazard_severity', 'slope_degree', 'is_imagine_corridor']], on='case_number', how='left')
        panel['fire_hazard_severity'] = panel['fire_hazard_severity'].fillna(0.0)
        panel['slope_degree'] = panel['slope_degree'].fillna(0.0)
        panel['is_imagine_corridor'] = panel['is_imagine_corridor'].fillna(0.0)

    except Exception as e:
        print(f"   [!] Failed to spatial join WUI/Corridors: {e}", flush=True)
        panel['fire_hazard_severity'] = 0.0
        panel['slope_degree'] = 0.0
        panel['is_imagine_corridor'] = 0.0

    # ── 6. Temporally-Safe KNN Demographic Imputation ────────────────────────
    print("7. Performing Temporally-Safe KNN Imputation...", flush=True)
    demo_cols = ['median_household_income', 'race_white', 'race_black', 'race_hispanic', 
                 'renter_share', 'rent_burden', 'total_population', 'median_age',
                 'appraised_value', 'building_age', 'mortgage_rate_30yr', 
                 'fed_funds_rate', 'local_unemployment_rate',
                 'knn_petition_rate_1km', 'dist_petition_rate_lag1']
                 
    # Ensure all exist
    for c in demo_cols:
        if c not in panel.columns: panel[c] = np.nan
        
    # Get base year per case
    case_years = get_base_case_year(panel)
    panel = panel.merge(case_years.rename(columns={'year': 'base_case_year'}), on='case_number', how='left')
    
    # We will build a single case-level dataframe for the imputation base
    cs_base = panel[['case_number', 'base_case_year', 'latitude', 'longitude'] + demo_cols].drop_duplicates('case_number')
    cs_imputed = cs_base.copy()
    
    unique_years = sorted(cs_base['base_case_year'].dropna().unique())
    imputed_count = 0
    
    for t_year in unique_years:
        for col in demo_cols:
            # Train pool: cases <= t_year with valid lat/lon AND valid col
            train_pool = cs_imputed[(cs_imputed['base_case_year'] <= t_year) & cs_imputed['latitude'].notna() & cs_imputed[col].notna()]
            
            # Target pool: cases == t_year with valid lat/lon AND missing col
            target_pool = cs_imputed[(cs_imputed['base_case_year'] == t_year) & cs_imputed['latitude'].notna()]
            missing_idx = target_pool[target_pool[col].isna()].index
            
            if len(train_pool) > 5 and len(missing_idx) > 0:
                knn = KNeighborsRegressor(n_neighbors=min(5, len(train_pool)), weights='distance')
                knn.fit(train_pool[['latitude', 'longitude']].values, train_pool[col].values)
                
                imputed_vals = knn.predict(cs_imputed.loc[missing_idx, ['latitude', 'longitude']].values)
                cs_imputed.loc[missing_idx, col] = imputed_vals
                imputed_count += len(missing_idx)
                
    print(f"   Chronologically KNN-imputed missing demographics for {imputed_count:,} data points.", flush=True)
    
    # Merge imputed case-level data back onto the full panel
    panel = panel.drop(columns=demo_cols)
    panel = panel.merge(cs_imputed[['case_number'] + demo_cols], on='case_number', how='left')
    panel = panel.drop(columns=['base_case_year'])
    
    # ── 7. Explicitly Drop Target Leakage (NLP / Remands) ────────────────────
    print("8. Final Cleanup (Removing Target Leakage NLP/Remands)...", flush=True)
    nlp_cols = ["Aggregate_Sentiment", "Opposition_Volume", "Support_Volume", "Remand_Count"]
    for c in nlp_cols:
        if c in panel.columns:
            panel = panel.drop(columns=[c])

    print(f"9. Saving Hydrated Panel to {PANEL_PATH}...", flush=True)
    panel.to_csv(PANEL_PATH, index=False)
    print(f"Done! New panel has {panel.shape[0]:,} rows and {panel.shape[1]} columns.", flush=True)

if __name__ == "__main__":
    main()
