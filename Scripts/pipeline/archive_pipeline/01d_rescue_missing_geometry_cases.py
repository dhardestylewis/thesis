"""
01d_rescue_missing_geometry_cases.py
=====================================
Rescues petition intensity for cases that failed in 01c due to missing geometry.

Three rescue tiers:
  Tier 1 (19 cases): Valid polygon in zoning DB, signers found in tcad_parcels.geojson via geo_id
  Tier 2 (29 cases): No polygon in zoning DB but lat/lon centroid → nearest parcel → 200ft buffer
  Tier 3 (7 cases):  In zoning_land_use_merged with polygon/lat_lon (missed by enriched_zoning)
  Tier 4 (11 cases): Completely absent — synthesize centroid from matched signer lat/lons

For unmatched signers (those not in geo_id): nearest-parcel lookup within 500ft.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import json, sys, os
from pathlib import Path
from shapely.geometry import shape, Point
from shapely import wkt

ROOT = Path(r"C:\Users\dhl\data\Thesis\thesis")
sys.path.append(str(ROOT / "Scripts"))
from config.paths import ROOT_DIR, DATA_DIR, PANEL_DIR, PROTEST_PETITIONS_DIR, GIS_DIR, ZONING_CASES_DIR

# ── Load all sources ────────────────────────────────────────────────────────
print("Loading datasets...")

petitions_raw = pd.read_csv(PROTEST_PETITIONS_DIR / "petition_signers_from_pdf.csv", dtype=str)
petitions_raw['norm'] = petitions_raw['tcad_normalized'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
petitions_signed = petitions_raw[petitions_raw['signed'] == '1'].copy()

pet_intensity = pd.read_csv(PROTEST_PETITIONS_DIR / "advanced_geometric_petition_intensity.csv")
missing_geom = pet_intensity[pet_intensity['spatial_total_parcels'] == 0]['case_number'].tolist()
signed_cases  = set(petitions_signed['case_number'].unique())
missing_signed = [c for c in missing_geom if c in signed_cases]
print(f"Cases needing rescue: {len(missing_signed)}")

# Load TCAD GeoJSON (has geo_id matching signer TCAD IDs)
print("Loading TCAD GeoJSON (slow, 330MB)...")
tcad_geo = gpd.read_file(GIS_DIR / "TCAD" / "tcad_parcels.geojson")
tcad_geo['geo_norm'] = tcad_geo['geo_id'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
tcad_geo_crs = tcad_geo.to_crs(epsg=2277)   # Texas State Plane ft
tcad_geo_wgs = tcad_geo.copy()              # WGS84 for lat/lon ops

# Load property_universe for fallback lat/lon of unmatched signers
props = pd.read_csv(PANEL_DIR / "parcel" / "property_universe.csv", dtype=str, low_memory=False)
props['norm'] = props['standardized_tcad_id'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
props_latlon = props.dropna(subset=['latitude','longitude']).copy()
props_latlon['latitude']  = pd.to_numeric(props_latlon['latitude'],  errors='coerce')
props_latlon['longitude'] = pd.to_numeric(props_latlon['longitude'], errors='coerce')

# Load all zoning data sources for development geometry
enriched = pd.read_csv(ZONING_CASES_DIR / "Processed_Data" / "CSV" / "enriched_zoning_data_updated.csv", low_memory=False)
merged_z  = pd.read_csv(ZONING_CASES_DIR / "Processed_Data" / "CSV" / "zoning_land_use_merged_data.csv", low_memory=False)

def parse_geom(g_str):
    try:
        if str(g_str).startswith('{'):
            g_dict = json.loads(str(g_str).replace("'", '"'))
            return shape(g_dict)
        return wkt.loads(str(g_str))
    except:
        return None

# ── Build development geometry lookup from ALL sources ────────────────────────
dev_geom = {}   # case_number → shapely geometry (EPSG:4326)
dev_latlon = {} # case_number → (lat, lon)

# Source A: enriched_zoning_data_updated
for _, row in enriched.drop_duplicates('case_number').iterrows():
    cn = str(row['case_number']).strip()
    g = parse_geom(row.get('the_geom',''))
    if g is not None:
        dev_geom[cn] = g
    if cn not in dev_latlon:
        lat = pd.to_numeric(row.get('latitude'), errors='coerce')
        lon = pd.to_numeric(row.get('longitude'), errors='coerce')
        if pd.notna(lat) and pd.notna(lon):
            dev_latlon[cn] = (lat, lon)

# Source B: zoning_land_use_merged
for _, row in merged_z.drop_duplicates('case_number').iterrows():
    cn = str(row['case_number']).strip()
    if cn not in dev_geom:
        g = parse_geom(row.get('the_geom',''))
        if g is not None:
            dev_geom[cn] = g
    if cn not in dev_latlon:
        lat = pd.to_numeric(row.get('latitude'), errors='coerce')
        lon = pd.to_numeric(row.get('longitude'), errors='coerce')
        if pd.notna(lat) and pd.notna(lon):
            dev_latlon[cn] = (lat, lon)

print(f"Development geometries available: {len(dev_geom)} | lat/lons: {len(dev_latlon)}")

# ── Build tcad_geo spatial index for nearest-parcel lookup ───────────────────
print("Building spatial index on TCAD parcels...")
tcad_geo_crs_idx = tcad_geo_crs.copy()
tcad_geo_crs_idx['centroid'] = tcad_geo_crs_idx.geometry.centroid

# ── Helper: nearest parcel to a point ────────────────────────────────────────
def nearest_parcel_to_point(lat, lon, max_dist_ft=500):
    """Return (geo_norm, geometry_ft) of nearest TCAD parcel to lat/lon within max_dist_ft."""
    pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=2277).geometry.iloc[0]
    candidates = tcad_geo_crs_idx[tcad_geo_crs_idx['centroid'].distance(pt) < max_dist_ft]
    if len(candidates) == 0:
        return None, None
    nearest = candidates.loc[candidates['centroid'].distance(pt).idxmin()]
    return nearest['geo_norm'], nearest['geometry']

# ── Main rescue loop ─────────────────────────────────────────────────────────
results = []

for cn in missing_signed:
    signers = petitions_signed[petitions_signed['case_number'] == cn].copy()

    # Step 1: Match signers to GeoJSON via geo_id (TCAD 10-digit)
    matched = signers.merge(tcad_geo_crs[['geo_norm','geometry']].rename(columns={'geometry':'parcel_geom'}),
                            left_on='norm', right_on='geo_norm', how='inner')
    
    # Step 2: For unmatched signers, try property_universe lat/lon → nearest parcel
    unmatched_norms = set(signers['norm']) - set(matched['geo_norm'])
    extra_rows = []
    for norm in unmatched_norms:
        # Try to get lat/lon from property_universe
        pu = props_latlon[props_latlon['norm'] == norm]
        if len(pu) == 0:
            continue
        lat = float(pu['latitude'].iloc[0])
        lon = float(pu['longitude'].iloc[0])
        geo_norm, parcel_geom = nearest_parcel_to_point(lat, lon, max_dist_ft=300)
        if geo_norm is not None:
            extra_rows.append({'norm': norm, 'geo_norm': geo_norm, 'parcel_geom': parcel_geom})
    
    if extra_rows:
        extra_df = pd.DataFrame(extra_rows)
        extra_signers = signers.merge(extra_df[['norm','geo_norm','parcel_geom']], on='norm', how='inner')
        matched = pd.concat([matched, extra_signers], ignore_index=True)

    if len(matched) == 0:
        print(f"  {cn}: 0 signers resolved → skipping")
        continue

    # Step 3: Get development polygon
    dev_poly_wgs = dev_geom.get(cn)
    
    if dev_poly_wgs is None and cn in dev_latlon:
        # Tier 2/3: lat/lon → nearest parcel as dev footprint
        lat, lon = dev_latlon[cn]
        geo_norm, dev_parcel_geom_ft = nearest_parcel_to_point(lat, lon, max_dist_ft=200)
        if dev_parcel_geom_ft is not None:
            dev_poly_ft = dev_parcel_geom_ft
        else:
            # fallback: 200ft circle buffer from centroid
            pt_ft = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=2277).geometry.iloc[0]
            dev_poly_ft = pt_ft.buffer(200)
    elif dev_poly_wgs is not None:
        dev_poly_ft = gpd.GeoDataFrame(geometry=[dev_poly_wgs], crs="EPSG:4326").to_crs(epsg=2277).geometry.iloc[0]
    else:
        # Tier 4: synthesize centroid from signer centroids
        signer_pts = matched['parcel_geom'].apply(lambda g: g.centroid)
        if len(signer_pts) == 0:
            print(f"  {cn}: no signer geometry fallback → skipping")
            continue
        all_x = [p.x for p in signer_pts]
        all_y = [p.y for p in signer_pts]
        synth_centroid = Point(np.median(all_x), np.median(all_y))
        dev_poly_ft = synth_centroid.buffer(200)
        print(f"  {cn}: SYNTHESIZED centroid from {len(signer_pts)} signers")

    # Step 4: Calculate intensities
    dev_area_sqft = dev_poly_ft.area
    buffer_200ft   = dev_poly_ft.buffer(200)

    signer_areas_within  = []
    signer_areas_outside = []

    for _, srow in matched.iterrows():
        parcel = srow['parcel_geom']
        area   = parcel.area
        if buffer_200ft.intersects(parcel):
            signer_areas_within.append(area)
        else:
            signer_areas_outside.append(area)

    total_signer_area = sum(signer_areas_within) + sum(signer_areas_outside)
    unofficial_intensity = (total_signer_area / dev_area_sqft) if dev_area_sqft > 0 else 0.0

    result = {
        'case_number': cn,
        'rescue_tier': 'geojson_geo_id',
        'signers_resolved': len(matched),
        'signers_within_200ft': len(signer_areas_within),
        'signers_outside_200ft': len(signer_areas_outside),
        'dev_area_sqft': dev_area_sqft,
        'total_signer_area_sqft': total_signer_area,
        'unofficial_protest_intensity': unofficial_intensity,
    }
    results.append(result)
    print(f"  {cn}: {len(matched)} signers resolved → intensity={unofficial_intensity:.4f}")

# ── Save results ─────────────────────────────────────────────────────────────
out = pd.DataFrame(results)
out_path = PROTEST_PETITIONS_DIR / "rescued_petition_intensity.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved {len(out)} rescued cases to {out_path}")
print(out[['case_number','signers_resolved','unofficial_protest_intensity']].to_string())
