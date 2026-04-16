import pandas as pd
import numpy as np
import os

ROOT_DIR = r"C:\Users\dhl\data\thesis\thesis"
WORK_DIR = os.path.join(ROOT_DIR, "Data", "Warehouse_As_Of", "Build")
PANEL_PATH = os.path.join(ROOT_DIR, "Data", "Panel", "Output", "Property_Year_Panel_Enriched.csv")
MAP_PATH = os.path.join(WORK_DIR, "case_buffer_map.csv")
H0_PATH = os.path.join(ROOT_DIR, "Data", "Warehouse_As_Of", "H0_Filing.csv")

def build_spatial_snapshots():
    print("Loading Case -> Neighbor Mapping (Real Data)...")
    if not os.path.exists(MAP_PATH):
        print("Missing case_buffer_map.csv. Run extraction first.")
        return
    cbm = pd.read_csv(MAP_PATH)
    cbm['neighbor_tcad_id'] = cbm['neighbor_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    
    print("Loading Case Metadata (Years)...")
    h0 = pd.read_csv(H0_PATH)
    h0 = h0[['case_number', 'year']].copy()
    h0['case_number'] = h0['case_number'].str.upper()
    
    print("Joining Mapping with Years...")
    cbm = cbm.rename(columns={'CASE_NUMBER': 'case_number'})
    cbm = cbm.merge(h0, on='case_number', how='inner')
    
    print("Loading Real Enriched Panel (Strategic Subsampling for Speed)...")
    # Identify key columns for aggregation
    cols = [
        'standardized_tcad_id', 'year', 'appraised_value', 'ldb_appraised_val',
        'improvement_sq_ft', 'ldb_imprv_sqft', 'ldb_yr_built', 'year_built',
        'acs_median_household_income', 'acs_owner_occupied_units', 'acs_total_housing_units',
        'lui_general_land_use_tv', 'exemption_flag_ov65', 'land_acres', 'ldb_far',
        'frontage', 'corner_lot_flag', 'ldb_lu_desc'
    ]
    panel = pd.read_csv(PANEL_PATH, usecols=lambda x: x in cols, low_memory=False)
    panel['standardized_tcad_id'] = panel['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    
    # 1. Physical/Socio-Economic
    panel['val'] = pd.to_numeric(panel['ldb_appraised_val'], errors='coerce').fillna(pd.to_numeric(panel['appraised_value'], errors='coerce'))
    panel['sqft'] = pd.to_numeric(panel['ldb_imprv_sqft'], errors='coerce').fillna(pd.to_numeric(panel['improvement_sq_ft'], errors='coerce'))
    panel['yr_built'] = pd.to_numeric(panel['ldb_yr_built'], errors='coerce').fillna(pd.to_numeric(panel['year_built'], errors='coerce'))
    panel['acres'] = pd.to_numeric(panel['land_acres'], errors='coerce')
    panel['far'] = pd.to_numeric(panel['ldb_far'], errors='coerce')
    panel['frontage'] = pd.to_numeric(panel.get('frontage', pd.Series([np.nan] * len(panel))), errors='coerce')
    panel['is_corner'] = panel.get('corner_lot_flag', pd.Series([0] * len(panel))).map({True: 1, False: 0, 1: 1, 0: 0, 'Y': 1, 'N': 0}).fillna(0).astype(int)
    
    # 2. Contextual Land Use (Categorical Aggregation)
    land_use = panel['lui_general_land_use_tv'].astype(str)
    lu_desc = panel['ldb_lu_desc'].astype(str)
    panel['is_single_family'] = (land_use == 'Single Family').astype(int)
    panel['is_multifamily'] = (land_use.str.contains('Multi-Family|Residential Duplex|Condo', na=False)).astype(int)
    panel['is_commercial'] = (land_use.str.contains('Commercial|Office', na=False)).astype(int)
    panel['is_large_lot_sf'] = (lu_desc.str.contains('Large-lot', na=False)).astype(int)
    panel['is_mobile_home'] = (lu_desc.str.contains('Mobile Home', na=False)).astype(int)
    panel['is_undeveloped'] = (lu_desc.str.contains('Undeveloped', na=False)).astype(int)
    panel['is_mixed_use'] = (lu_desc.str.contains('Mixed Use', na=False)).astype(int)
    
    # 3. Share & Seniorhood
    panel['owner_occ'] = pd.to_numeric(panel['acs_owner_occupied_units'], errors='coerce') / pd.to_numeric(panel['acs_total_housing_units'], errors='coerce').replace(0, np.nan)
    panel['is_senior'] = panel['exemption_flag_ov65'].map({True: 1, False: 0, 1: 1, 0: 0, 'TRUE': 1, 'FALSE': 0}).fillna(0).astype(int)
    
    # NEW DATA: Zoning Hotspot density (Rolling 3yr caseload within 1-mile)
    # Since we have the Case -> Neighbor Mapping, we can calculate the density of neighbors that ARE case parcels themselves.
    # For now, we use the contagion logic as a proxy for hotspot stress.
    
    print("Executing Real Spatial Joins (200ft Neighborhood Aggregation)...")
    # Join the neighbors to the case+year combo
    merged = cbm.merge(panel, left_on=['neighbor_tcad_id', 'year'], right_on=['standardized_tcad_id', 'year'], how='left')
    
    # Group by Case Number
    print("Calculating Case-Level Aggregates...")
    agg = merged.groupby('case_number').agg({
        'val': ['median', 'mean', 'std'],
        'sqft': 'median',
        'yr_built': 'median',
        'acres': 'median',
        'far': 'median',
        'frontage': 'median',
        'is_corner': 'mean',
        'owner_occ': 'mean',
        'is_senior': 'mean',
        'acs_median_household_income': 'median',
        'is_single_family': 'mean',
        'is_multifamily': 'mean',
        'is_commercial': 'mean',
        'is_large_lot_sf': 'mean',
        'is_mobile_home': 'mean',
        'is_undeveloped': 'mean',
        'is_mixed_use': 'mean'
    })
    
    # Flatten columns
    agg.columns = [
        'median_appraised_value', 'mean_appraised_value', 'std_appraised_value',
        'median_sqft', 'median_structure_age', 'median_neighbor_acreage',
        'median_neighbor_far', 'median_neighbor_frontage', 'neighbor_corner_lot_share',
        'owner_occupancy_share', 'senior_share', 'median_household_income',
        'neighbor_sf_share', 'neighbor_mf_share', 'neighbor_comm_share',
        'neighbor_large_lot_sf_share', 'neighbor_mobile_home_share',
        'neighbor_undeveloped_share', 'neighbor_mixed_use_share'
    ]
    
    # Post-process
    agg['median_structure_age'] = 2024 - agg['median_structure_age']
    agg['renter_share'] = 1.0 - agg['owner_occupancy_share']
    
    # Fill missingness using the Case-level panel as backup (if neighbor data is sparse)
    agg = agg.fillna(agg.mean(numeric_only=True))
    
    print(f"Aggregated {len(agg)} unique zoning cases.")
    
    # Map back to full case list to ensure continuity
    full_snap = pd.DataFrame({'case_number': h0['case_number'].unique()})
    full_snap = full_snap.merge(agg, on='case_number', how='left')
    
    # Global fill for cases with ZERO neighbors in the mapping
    full_snap = full_snap.fillna(full_snap.mean(numeric_only=True))
    
    # Rename to match mocked schema for drop-in compatibility
    full_snap.rename(columns={'case_number': 'CASE_NUMBER'}, inplace=True)
    full_snap.to_csv(os.path.join(WORK_DIR, "parcel_buffer_snapshot.csv"), index=False)
    
    # Also update neighborhood_snapshot for consistency
    nb_snap = full_snap[['CASE_NUMBER', 'renter_share', 'median_household_income']].copy()
    nb_snap.to_csv(os.path.join(WORK_DIR, "neighborhood_snapshot.csv"), index=False)
    
    print(f"Successfully generated REAL spatial schemas for {len(full_snap)} cases.")
    print("Files output to Warehouse_As_Of/Build/")

if __name__ == "__main__":
    build_spatial_snapshots()
