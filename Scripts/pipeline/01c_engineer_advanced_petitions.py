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

from config.paths import ROOT_DIR, DATA_DIR, PANEL_DIR, PROTEST_PETITIONS_DIR, GIS_DIR, ZONING_CASES_DIR

def engineer_advanced_petitions():
    panel_path = PANEL_DIR / "biweekly_panel.csv"
    petitions_path = PROTEST_PETITIONS_DIR / "advanced_geometric_petition_intensity.csv"
    ocr_path = ROOT_DIR / "Scratch" / "Data_Exports" / "ocr_petition_results.csv"
    
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
        
        print("Loading 45,000+ raw signatures and normalizing TCADs...")
        t0 = time.time()
        petitions = pd.read_csv(PROTEST_PETITIONS_DIR / "petition_signers_from_pdf.csv", dtype=str)
        petitions['tcad_normalized'] = petitions['tcad_normalized'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
        petitions = petitions[petitions['signed'] == '1']
        
        # Load TCAD GeoJSON — keep geo_id column intact for geo_norm matching
        tcad_geo = gpd.read_file(GIS_DIR / "TCAD" / "tcad_parcels.geojson")
        tcad_geo = tcad_geo.to_crs(epsg=2277)
        # Legacy alias for modules that still expect tcad with numeric index
        tcad = tcad_geo.copy().set_index('geo_id')
        
        # Load development polygons from enriched_zoning (primary)
        import json
        from shapely import wkt
        from shapely.geometry import shape
        
        def parse_geom(g_str):
            try:
                if str(g_str).startswith('{'):
                    g_dict = json.loads(str(g_str).replace("'", '"'))
                    return shape(g_dict)
                return wkt.loads(str(g_str))
            except:
                return None
        
        cases_df = pd.read_csv(ZONING_CASES_DIR / "Processed_Data" / "CSV" / "enriched_zoning_data_updated.csv")
        cases_df['geometry'] = cases_df['the_geom'].apply(parse_geom)
        cases_df = cases_df.dropna(subset=['geometry'])
        
        # Also load zoning_land_use_merged as supplementary geometry source
        merged_z = pd.read_csv(ZONING_CASES_DIR / "Processed_Data" / "CSV" / "zoning_land_use_merged_data.csv", low_memory=False)
        merged_z['geometry'] = merged_z['the_geom'].apply(parse_geom)
        merged_z_valid = merged_z.dropna(subset=['geometry'])
        
        # Combine: enriched first, merged fills gaps
        combined = pd.concat([cases_df[['case_number','geometry']], merged_z_valid[['case_number','geometry']]], ignore_index=True)
        combined = combined.drop_duplicates(subset=['case_number'], keep='first')
        
        cases_gdf = gpd.GeoDataFrame(combined, geometry='geometry').set_crs("EPSG:4326").to_crs("EPSG:2277")
        cases_gdf['case_number'] = cases_gdf['case_number'].astype(str).str.strip()
        cases_gdf = cases_gdf.set_index('case_number')
        
        props = pd.read_csv(PANEL_DIR / "parcel" / "property_universe.csv", dtype={'standardized_tcad_id': str})
        props['standardized_tcad_id'] = props['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
        props = props.set_index('standardized_tcad_id')
        print(f"Datasets loaded in {time.time() - t0:.1f}s. cases_gdf has {len(cases_gdf)} development polygons.")
        print(f"Proceeding with spatial vectors using geo_id primary signer matching...")
        
        build_spatial_vectors(petitions, tcad_geo, cases_gdf, props)
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
    
    # Compute petition mobilization attempt metrics from raw PDF signers
    raw_petitions = pd.read_csv(PROTEST_PETITIONS_DIR / "petition_signers_from_pdf.csv", dtype=str)
    mobilization = raw_petitions.groupby('case_number').apply(
        lambda g: pd.Series({
            'petition_attempted': 1,
            'mobilization_failure': (g['signed'] == '0').sum()
        })
    ).reset_index()
    
    # Also handle .SH aliases for the petition mobilization metrics
    alias_map = {
        'C14-2016-0063': 'C14-2016-0063.SH', 'C14-2018-0100': 'C14-2018-0100.SH',
        'C14-2021-0008': 'C14-2021-0008.SH', 'C14-2008-0057': 'C14-2008-0057.SH',
        'C14-2016-0023': 'C14-2016-0023.SH', 'C14-2022-0018': 'C14-2022-0018.SH',
        'C14-2014-0031': 'C14-2014-0031.SH', 'C14-2023-0007': 'C14-2023-0007.SH'
    }
    aliased_mob = mobilization.copy()
    aliased_mob['case_number'] = aliased_mob['case_number'].map(lambda x: alias_map.get(x, x))
    mobilization = pd.concat([mobilization, aliased_mob]).drop_duplicates(subset=['case_number'])
    
    petitions = petitions.merge(mobilization, on='case_number', how='left')
    petitions['petition_attempted'] = petitions['petition_attempted'].fillna(0).astype(int)
    petitions['mobilization_failure'] = petitions['mobilization_failure'].fillna(0).astype(int)
    
    # Identify primary injection period (The period where the patched petition occurred)
    petition_periods = panel[panel['petition_event'] == 1].groupby('case_number')['period_seq'].min().reset_index()
    petition_periods = petition_periods.rename(columns={'period_seq': 'petition_period'})
    
    # Identify secondary injection period (First Council Hearing)
    first_council = panel[panel['council_hearings_this_period'] > 0].groupby('case_number')['period_seq'].min().reset_index()
    first_council = first_council.rename(columns={'period_seq': 'council_period'})
    
    # Identify tertiary injection period (First Commission Hearing)
    first_comm = panel[panel['commission_hearings_this_period'] > 0].groupby('case_number')['period_seq'].min().reset_index()
    first_comm = first_comm.rename(columns={'period_seq': 'comm_period'})
    
    petitions = petitions.merge(petition_periods, on='case_number', how='left')
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
        
    # Priority: Petition Event -> EDIMS -> Council -> Commission -> 1
    petitions['injection_period'] = petitions['petition_period'].fillna(petitions['edims_period']).fillna(petitions['council_period']).fillna(petitions['comm_period']).fillna(1).astype(int)
    
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
                    'delta_protesting_friction', 'delta_silent_friction',
                    'petition_attempted', 'mobilization_failure']
    
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
            panel['cumulative_' + f] = panel.groupby('case_number')[f + "_this_period"].transform(lambda x: x.cumsum().fillna(0))
        else:
            # For JSON string vector, just carry forward the last non-empty one
            panel['cumulative_' + f] = panel[f + "_this_period"].replace('[]', np.nan)
            panel['cumulative_' + f] = panel.groupby('case_number')['cumulative_' + f].transform(lambda x: x.ffill().fillna('[]'))

    # Clean up non-cumulative tracking variables
    for f in adv_features:
        del panel[f]
        del panel[f + "_this_period"]
    
    final_protests = panel[panel['cumulative_unofficial_protest_intensity'] > 0]['case_number'].nunique()
    
    print(f"Final UNOFFICIALLY protested cases (including >200ft): {final_protests}")
    
    print("Saving enriched biweekly panel...")
    panel.to_csv(panel_path, index=False)
    print(f"Successfully integrated advanced petition features into {panel_path}")

if __name__ == "__main__":
    engineer_advanced_petitions()
