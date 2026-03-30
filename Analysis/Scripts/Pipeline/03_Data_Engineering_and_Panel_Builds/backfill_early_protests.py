"""
backfill_early_protests.py
==========================
Deduces omitted non-protesting properties from early-year protest petitions
by spatially intersecting a 200ft buffer of the zoning case geometry with 
the universe of TCAD parcel points from the Property-Year Panel.

Outputs:
- petition_summary_backfilled.csv
- deduced_nonprotesting_properties.csv
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkb, wkt
import os

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data")
OUT_DIR = os.path.join(DATA, "Protest_Petitions", "Backfilled")
os.makedirs(OUT_DIR, exist_ok=True)

ZONING_CSV = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "enriched_zoning_data_updated.csv")
PET_SUMMARY = os.path.join(DATA, "Protest_Petitions", "petition_summary_from_pdf.csv")
PET_SIGNERS = os.path.join(DATA, "Protest_Petitions", "petition_signers_from_pdf.csv")
PANEL_CSV = os.path.join(DATA, "Panel", "Output", "Property_Year_Panel_v3.csv")

def run_backfill():
    print("=== DEDUCING NON-PROTESTING PROPERTIES ===")
    
    # 1. Load Petition Summaries
    summary = pd.read_csv(PET_SUMMARY)
    summary['case_number'] = summary['case_number'].astype(str).str.strip()
    
    # Find early years where signer_pct exactly 100.0 or closely mimics "Total == Signers"
    mask_100 = (summary['signer_pct'] == 100.0) | (summary['total_parcels'] == summary['signers'])
    cases_to_fix = summary.loc[mask_100, 'case_number'].unique()
    print(f"Detected {len(cases_to_fix)} cases with exactly 100% signers (likely missing non-protesting properties).")
    
    # 2. Extract TCAD base from Panel (chunked to save memory)
    print("Loading TCAD universe from Panel...")
    chunksize = 200000
    tcad_list = []
    usecols = ['standardized_tcad_id', 'latitude', 'longitude', 'land_acres']
    for chunk in pd.read_csv(PANEL_CSV, chunksize=chunksize, usecols=usecols, low_memory=False):
        chunk = chunk.dropna(subset=['latitude', 'longitude', 'standardized_tcad_id'])
        tcad_list.append(chunk)
        
    tcad_df = pd.concat(tcad_list, ignore_index=True)
    tcad_df['standardized_tcad_id'] = tcad_df['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    tcad_df = tcad_df.drop_duplicates(subset=['standardized_tcad_id'])
    
    print(f"Loaded {len(tcad_df)} unique TCAD parcels.")
    
    # 3. Create TCAD GeoDataFrame
    gdf_tcad = gpd.GeoDataFrame(
        tcad_df, 
        geometry=gpd.points_from_xy(tcad_df.longitude, tcad_df.latitude),
        crs="EPSG:4326"
    )
    # Project to Texas Central (ft)
    gdf_tcad = gdf_tcad.to_crs("EPSG:2277")
    
    # 4. Load Zoning Geometries
    print("Loading zoning cases...")
    zoning = pd.read_csv(ZONING_CSV, low_memory=False)
    zoning['case_number'] = zoning['Case Number'].fillna(zoning['case_number']).astype(str).str.strip()
    
    zoning_fix = zoning[zoning['case_number'].isin(cases_to_fix)].drop_duplicates(subset=['case_number']).copy()
    
    cases_geom = []
    import ast
    from shapely.geometry import shape
    
    for idx, row in zoning_fix.iterrows():
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
        except Exception as e:
            # print(f"Error parsing geometry for {row['case_number']}: {e}")
            pass
            
    print(f"Extracted valid geometries for {len(cases_geom)} cases.")
    
    # Create Zoning GeoDataFrame
    df_geom = pd.DataFrame(cases_geom, columns=['case_number', 'geometry'])
    gdf_zoning = gpd.GeoDataFrame(df_geom, geometry='geometry', crs="EPSG:4326")
    gdf_zoning = gdf_zoning.to_crs("EPSG:2277")
    
    # BUFFER BY 200 FEET
    print("Buffering zoning cases by 200 feet...")
    gdf_zoning['buffer_200ft'] = gdf_zoning.geometry.buffer(200)
    gdf_zoning = gdf_zoning.set_geometry('buffer_200ft')
    
    # 5. Spatial Join
    print("Performing spatial join...")
    # Buffer in Texas Central ensures 200 feet is used, geometry becomes polygons
    intersected = gpd.sjoin(gdf_tcad, gdf_zoning, how="inner", predicate="intersects")
    print(f"Intersections found: {len(intersected)}")
    
    # 6. Load Explicit Signers
    signers = pd.read_csv(PET_SIGNERS)
    signers['case_number'] = signers['case_number'].astype(str).str.strip()
    signers['tcad_normalized'] = signers['tcad_normalized'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    
    # 7. Deduce Non-Protesting
    results_summary = []
    deduced_nonprotesting = []
    
    print("Computing metrics...")
    for case, group in intersected.groupby('case_number'):
        # All TCADs in 200ft
        all_tcads_in_buffer = set(group['standardized_tcad_id'])
        
        # Explicit Protesting TCADs
        case_signers = signers[signers['case_number'] == case]
        explicit_tcads = set(case_signers['tcad_normalized'])
        
        # Intersection: TCADs that exist in our panel that signed
        matched_explicit = all_tcads_in_buffer.intersection(explicit_tcads)
        
        # Set Difference: Missing Non-Protesting
        missing_tcads = all_tcads_in_buffer - matched_explicit
        
        # Recompute totals
        total_parcels = len(all_tcads_in_buffer)
        num_signers = len(case_signers) # use the raw number of signers from PDF as the trusted count
        
        # Calculate areas
        # Note: land_acres is in acres. 
        total_area = group['land_acres'].sum()
        
        df_protest_subset = group[group['standardized_tcad_id'].isin(matched_explicit)]
        protest_area = df_protest_subset['land_acres'].sum()
        
        calc_pct = (protest_area / total_area * 100) if total_area > 0 else 0.0
        
        results_summary.append({
            'case_number': case,
            'total_parcels_backfilled': total_parcels,
            'signers_trusted': num_signers,
            'signer_pct_backfilled': calc_pct
        })
        
        for t in missing_tcads:
            deduced_nonprotesting.append({
                'case_number': case,
                'tcad_id': t,
                'protested': 0
            })
            
    df_results = pd.DataFrame(results_summary, columns=['case_number', 'total_parcels_backfilled', 'signers_trusted', 'signer_pct_backfilled'])
    df_deduced = pd.DataFrame(deduced_nonprotesting, columns=['case_number', 'tcad_id', 'protested'])
    
    if len(df_results) == 0:
        print("No results to backfill!")
        return
        
    # Merge back to summary
    summary_updated = summary.merge(df_results, on='case_number', how='left')
    
    # Replace columns where backfilled data exists
    mask = summary_updated['total_parcels_backfilled'].notna()
    summary_updated.loc[mask, 'total_parcels'] = summary_updated.loc[mask, 'total_parcels_backfilled'].astype(int)
    summary_updated.loc[mask, 'signer_pct'] = summary_updated.loc[mask, 'signer_pct_backfilled']
    
    summary_updated = summary_updated.drop(columns=['total_parcels_backfilled', 'signers_trusted', 'signer_pct_backfilled'])
    
    # Output
    out_sum = os.path.join(OUT_DIR, "petition_summary_backfilled.csv")
    out_np = os.path.join(OUT_DIR, "deduced_nonprotesting_properties.csv")
    
    summary_updated.to_csv(out_sum, index=False)
    df_deduced.to_csv(out_np, index=False)
    
    print(f"Saved {len(df_deduced)} deduced non-protesting properties.")
    print(f"Saved backfilled summary to: {out_sum}")
    
    # Let's print a few before/after examples
    print("\nSample Backfill Results:")
    sample = df_results.head(5)
    for _, row in sample.iterrows():
        old_val = summary.loc[summary['case_number'] == row['case_number'], 'signer_pct'].values[0]
        print(f"{row['case_number']}: was {old_val}%, now {row['signer_pct_backfilled']:.2f}% (Total Parcels: {row['total_parcels_backfilled']})")

if __name__ == "__main__":
    run_backfill()
