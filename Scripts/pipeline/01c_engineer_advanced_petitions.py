import pandas as pd
import numpy as np
import os
import sys

ROOT = r"C:\Users\dhl\data\Thesis\thesis"
sys.path.append(os.path.join(ROOT, "Scripts"))

from pipeline.advanced_spatial_modules.calc_vectors import build_spatial_vectors
from pipeline.advanced_spatial_modules.engineer_neighbor_differentials import build_neighbor_differentials
from pipeline.advanced_spatial_modules.generate_pca_embeddings import build_pca_embeddings
from pipeline.advanced_spatial_modules.engineer_ears_differentials import build_ears_differentials
from pipeline.advanced_spatial_modules.engineer_temporal_differentials import build_temporal_differentials

ROOT = r"C:\Users\dhl\data\Thesis\thesis"
DATA = os.path.join(ROOT, "Data")

def engineer_advanced_petitions():
    panel_path = os.path.join(ROOT, "Data", "Panel", "biweekly_panel.csv")
    petitions_path = os.path.join(DATA, "Protest_Petitions", "advanced_geometric_petition_intensity.csv")
    ocr_path = os.path.join(ROOT, "Scratch", "Data_Exports", "ocr_petition_results.csv")
    
    if not os.path.exists(panel_path):
        print(f"Skipping 01c: {panel_path} does not exist locally.")
        return
        
    print(f"Loading official panel from {panel_path}...")
    panel = pd.read_csv(panel_path, low_memory=False)
    
    if not os.path.exists(petitions_path):
        import geopandas as gpd
        import time
        
        print(f"advanced_geometric_petition_intensity.csv not found!")
        print(f"Dynamically generating PCA embeddings, EARS differentials, and exact spatial bounding polygons...")
        print(f"Loading 526MB TCAD geometry into memory (this happens only ONCE)...")
        
        t0 = time.time()
        petitions = pd.read_csv("Data/Protest_Petitions/petition_signers_from_pdf.csv", dtype=str)
        petitions = petitions[petitions['signed'] == '1']
        
        tcad = gpd.read_file("Data/GIS/TCAD/tcad_parcels.geojson")
        tcad = tcad.to_crs(epsg=2277).set_index('geo_id')
        
        cases_gdf = gpd.read_file("Data/Zoning_Cases/zoning_cases_master_polygons.geojson")
        cases_gdf = cases_gdf.to_crs(epsg=2277).set_index('case_number')
        
        props = pd.read_csv("Data/Panel/parcel/property_universe.csv", dtype={'standardized_tcad_id': str})
        props['standardized_tcad_id'] = props['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
        props = props.set_index('standardized_tcad_id')
        print(f"Datasets loaded in {time.time() - t0:.1f}s. Proceeding with spatial vectors...")
        
        build_spatial_vectors(petitions, tcad, cases_gdf)
        build_neighbor_differentials(petitions, tcad, cases_gdf, props)
        build_temporal_differentials(petitions, tcad, cases_gdf, props)
        build_ears_differentials(petitions, tcad, cases_gdf, props)
        build_pca_embeddings(petitions, tcad, cases_gdf, props)
        
    petitions = pd.read_csv(petitions_path)
    
    cols_to_keep = ['case_number', 
                    'min_signer_dist', 'max_signer_dist', 'median_signer_dist', 
                    'signers_within_200ft', 'signers_outside_200ft', 
                    'unofficial_protest_intensity', 'signer_distance_vector',
                    'protesting_pct_single_family', 'silent_pct_single_family',
                    'protesting_pct_commercial', 'silent_pct_commercial',
                    'protesting_pct_multifamily', 'silent_pct_multifamily',
                    'protesting_mean_parcel_sqft', 'silent_mean_parcel_sqft',
                    'protester_embed_dim1', 'protester_embed_dim2', 'protester_embed_dim3', 'protester_embed_dim4',
                    'temporal_protesting_pct_sf', 'temporal_silent_pct_sf',
                    'temporal_protesting_pct_com', 'temporal_silent_pct_com',
                    'temporal_protesting_pct_mf', 'temporal_silent_pct_mf',
                    'delta_protesting_friction', 'delta_silent_friction']
    
    petitions = petitions[cols_to_keep].drop_duplicates(subset=['case_number'])
    
    # Identify primary injection period (First Council Hearing)
    first_council = panel[panel['council_hearings_this_period'] > 0].groupby('case_number')['period_seq'].min().reset_index()
    first_council = first_council.rename(columns={'period_seq': 'council_period'})
    
    # Identify secondary injection period (First Commission Hearing)
    first_comm = panel[panel['commission_hearings_this_period'] > 0].groupby('case_number')['period_seq'].min().reset_index()
    first_comm = first_comm.rename(columns={'period_seq': 'comm_period'})
    
    petitions = petitions.merge(first_council, on='case_number', how='left')
    petitions = petitions.merge(first_comm, on='case_number', how='left')
    
    # Load EDIMS OCR Ground Truth to align injection precisely
    if os.path.exists(ocr_path):
        ocr = pd.read_csv(ocr_path)
        
        def extract_date(url):
            date_str = str(url).split('/')[-1].split('-')[0]
            try:
                return pd.to_datetime(date_str, format='%Y%m%d')
            except:
                return pd.NaT
                
        ocr['Petition_Date'] = ocr['Meeting_URL'].apply(extract_date)
        petition_map_date = ocr.set_index('Case_Number')['Petition_Date'].to_dict()
        
        # We must find the `period_seq` that corresponds to the `Petition_Date` for each case
        edims_period_map = {}
        panel['period_start_dt'] = pd.to_datetime(panel['period_start'])
        for case, p_date in petition_map_date.items():
            if pd.isna(p_date): continue
            case_data = panel[panel['case_number'] == case]
            if case_data.empty: continue
            mask = case_data['period_start_dt'] >= p_date
            if mask.any():
                edims_period_map[case] = case_data[mask]['period_seq'].iloc[0]
            else:
                edims_period_map[case] = case_data['period_seq'].iloc[-1]
                
        petitions['edims_period'] = petitions['case_number'].map(edims_period_map)
    else:
        petitions['edims_period'] = np.nan
        
    # Priority: EDIMS -> Council -> Commission -> 1
    petitions['injection_period'] = petitions['edims_period'].fillna(petitions['council_period']).fillna(petitions['comm_period']).fillna(1).astype(int)
    
    # Initialize advanced feature columns
    adv_features = ['min_signer_dist', 'max_signer_dist', 'median_signer_dist', 
                    'signers_within_200ft', 'signers_outside_200ft', 
                    'unofficial_protest_intensity', 'signer_distance_vector',
                    'protesting_pct_single_family', 'silent_pct_single_family',
                    'protesting_pct_commercial', 'silent_pct_commercial',
                    'protesting_pct_multifamily', 'silent_pct_multifamily',
                    'protesting_mean_parcel_sqft', 'silent_mean_parcel_sqft',
                    'protester_embed_dim1', 'protester_embed_dim2', 'protester_embed_dim3', 'protester_embed_dim4',
                    'temporal_protesting_pct_sf', 'temporal_silent_pct_sf',
                    'temporal_protesting_pct_com', 'temporal_silent_pct_com',
                    'temporal_protesting_pct_mf', 'temporal_silent_pct_mf',
                    'delta_protesting_friction', 'delta_silent_friction']
    
    for f in adv_features:
        panel[f] = 0.0 if f != 'signer_distance_vector' else '[]'
        
    # Maps for advanced features
    adv_maps = {f: petitions.set_index(['case_number', 'injection_period'])[f].to_dict() for f in adv_features}
    
    print(f"Preparing to inject advanced spatial vectors for {len(petitions)} cases...")
    
    for f in adv_features:
        def apply_adv(row):
            key = (row['case_number'], row['period_seq'])
            val = adv_maps[f].get(key, 0.0 if f != 'signer_distance_vector' else '[]')
            return val
        panel[f + "_this_period"] = panel.apply(apply_adv, axis=1)
    
    print("Forward-filling cumulative advanced features...")
    
    for f in adv_features:
        if f != 'signer_distance_vector':
            panel['cumulative_' + f] = panel.groupby('case_number')[f + "_this_period"].transform(lambda x: x.cumsum().shift(1).fillna(0))
        else:
            # For JSON string vector, just carry forward the last non-empty one
            panel['cumulative_' + f] = panel[f + "_this_period"].replace('[]', np.nan)
            panel['cumulative_' + f] = panel.groupby('case_number')['cumulative_' + f].transform(lambda x: x.ffill().shift(1).fillna('[]'))
    
    # Drop intermediate columns
    for f in adv_features:
        del panel[f]
        del panel[f + "_this_period"]
    
    unofficial_protested = panel.groupby('case_number').last()['cumulative_unofficial_protest_intensity'] > 0
    print(f"Final UNOFFICIALLY protested cases (including >200ft): {unofficial_protested.sum()}")
    
    print("Saving enriched biweekly panel...")
    panel.to_csv(panel_path, index=False)
    print(f"Successfully integrated advanced petition features into {panel_path}")

if __name__ == "__main__":
    engineer_advanced_petitions()
