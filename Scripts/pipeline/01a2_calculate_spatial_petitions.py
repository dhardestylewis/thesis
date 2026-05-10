"""
01c_calculate_spatial_petitions.py
==================================
Calculates the true spatial footprint of all protest petitions by intersecting 
a 200ft buffer of the zoning cases with the TCAD parcel universe.

Outputs:
- Data/Protest_Petitions/petition_summary_spatial_true.csv
"""

import pandas as pd
import geopandas as gpd
from shapely import wkt
import os
import ast
from shapely.geometry import shape

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data")
OUT_FILE = os.path.join(DATA, "Protest_Petitions", "petition_summary_spatial_true.csv")

ZONING_CSV = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "CSV", "enriched_zoning_data_updated.csv")
PET_SIGNERS = os.path.join(DATA, "Protest_Petitions", "petition_signers_from_pdf.csv")
PANEL_CSV = os.path.join(DATA, "Panel", "Output", "Property_Year_Panel_Enriched.csv")

def run_spatial_calculation():
    print("=== CALCULATING TRUE SPATIAL PETITION FOOTPRINTS ===")
    
    # 1. Load explicit signers
    print("Loading explicit PDF signers...")
    signers = pd.read_csv(PET_SIGNERS)
    
    # CRITICAL FIX: Only keep parcels that actually signed (Format B PDFs include 'No' signers)
    signers['signed'] = pd.to_numeric(signers['signed'], errors='coerce').fillna(0)
    signers = signers[signers['signed'] == 1].copy()
    
    signers['case_number'] = signers['case_number'].astype(str).str.strip()
    signers['tcad_normalized'] = signers['tcad_normalized'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    
    # 2. Extract TCAD universe
    print("Loading TCAD universe from Land Database GeoJSON...")
    PANEL_CSV = os.path.join(DATA, "CoA_Open_Data", "Land_Database_2021.geojson")
    
    tcad_df = gpd.read_file(PANEL_CSV)
    tcad_df['standardized_tcad_id'] = tcad_df['pid_10'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    
    tcad_df['land_acres'] = pd.to_numeric(tcad_df['land_acres'], errors='coerce').fillna(0)
    
    # Average the land_acres across years to collapse the panel to unique parcels
    print("Collapsing TCAD universe to unique parcels...")
    tcad_unique = tcad_df.groupby('standardized_tcad_id').agg({
        'land_acres': 'mean'
    }).reset_index()
    
    print(f"Loaded {len(tcad_unique)} unique TCAD parcels.")
    
    # 3. Create TCAD GeoDataFrame (we use the original geometries directly)
    # Get the first geometry for each tcad_id
    tcad_geoms = tcad_df.groupby('standardized_tcad_id').first().reset_index()
    
    gdf_tcad = gpd.GeoDataFrame(
        tcad_unique, 
        geometry=tcad_geoms.geometry,
        crs="EPSG:4326"
    )
    # Project to Texas Central (ft)
    gdf_tcad = gdf_tcad.to_crs("EPSG:2277")
    
    # 4. Load Zoning Geometries
    print("Loading zoning cases...")
    zoning = pd.read_csv(ZONING_CSV, low_memory=False)
    zoning['case_number'] = zoning['Case Number'].fillna(zoning['case_number']).astype(str).str.strip()
    
    cases_geom = []
    for idx, row in zoning.drop_duplicates(subset=['case_number']).iterrows():
        g_str = str(row['the_geom'])
        if g_str == 'nan' or g_str.strip() == '':
            continue
        try:
            if g_str.startswith('{'):
                geom_dict = ast.literal_eval(g_str)
                geom = shape(geom_dict)
                cases_geom.append((row['case_number'], geom))
            elif g_str.startswith(('MULTI', 'POLY')):
                geom = wkt.loads(g_str)
                cases_geom.append((row['case_number'], geom))
        except Exception:
            pass
            
    print(f"Extracted valid geometries for {len(cases_geom)} cases.")
    
    df_geom = pd.DataFrame(cases_geom, columns=['case_number', 'geometry'])
    gdf_zoning = gpd.GeoDataFrame(df_geom, geometry='geometry', crs="EPSG:4326")
    gdf_zoning = gdf_zoning.to_crs("EPSG:2277")
    
    # 5. Buffer and Spatial Join
    print("Buffering zoning cases by 200 feet...")
    gdf_zoning['buffer_200ft'] = gdf_zoning.geometry.buffer(200)
    gdf_zoning = gdf_zoning.set_geometry('buffer_200ft')
    
    print("Performing spatial join...")
    intersected = gpd.sjoin(gdf_tcad, gdf_zoning, how="inner", predicate="intersects")
    print(f"Intersections found: {len(intersected)}")
    
    # 6. Calculate True Spatial Protest Footprint
    results_summary = []
    
    print("Computing true spatial footprint metrics...")
    for case, group in intersected.groupby('case_number'):
        # Total area of all TCADs in the 200ft buffer
        total_area = group['land_acres'].sum()
        
        # Explicit Protesting TCADs (from PDF)
        case_signers = signers[signers['case_number'] == case]
        explicit_tcads = set(case_signers['tcad_normalized'])
        
        # Intersection: TCADs in the 200ft buffer that actually signed
        matched_explicit = set(group['standardized_tcad_id']).intersection(explicit_tcads)
        
        # Calculate protested area
        df_protest_subset = group[group['standardized_tcad_id'].isin(matched_explicit)]
        protest_area = df_protest_subset['land_acres'].sum()
        
        calc_pct = (protest_area / total_area * 100.0) if total_area > 0 else 0.0
        
        results_summary.append({
            'case_number': case,
            'spatial_total_parcels': len(group),
            'spatial_signer_parcels': len(matched_explicit),
            'spatial_petition_pct': calc_pct
        })
        
    df_results = pd.DataFrame(results_summary)
    
    # Add back cases that had signers but no geometry/intersection (to preserve dates)
    all_signed_cases = set(signers['case_number'].unique())
    processed_cases = set(df_results['case_number'])
    missing_cases = all_signed_cases - processed_cases
    
    missing_rows = []
    for case in missing_cases:
        missing_rows.append({
            'case_number': case,
            'spatial_total_parcels': 0,
            'spatial_signer_parcels': len(signers[signers['case_number'] == case]),
            'spatial_petition_pct': 0.0 # Can't calculate footprint without geometry
        })
    
    if missing_rows:
        df_results = pd.concat([df_results, pd.DataFrame(missing_rows)], ignore_index=True)
    
    df_results.to_csv(OUT_FILE, index=False)
    print(f"Saved true spatial footprint to: {OUT_FILE} ({len(df_results)} cases)")

if __name__ == "__main__":
    run_spatial_calculation()
