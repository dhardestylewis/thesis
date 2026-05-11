"""
01d3_fix_14digit_tcad_and_unknown.py
Fixes two remaining failure classes:

1. NO_SIGNER_GEOMETRY: 14-digit OCR TCAD IDs (county suffix appended)
   Fix: truncate last 4 digits, keep first 10, re-zfill(10)

2. UNKNOWN: 9 cases with valid geo matches but still 0 in panel
   Fix: directly inject their intensities into rescued_petition_intensity.csv
   and re-patch into advanced_geometric_petition_intensity.csv
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import json, sys
from pathlib import Path
from shapely.geometry import shape, Point
from shapely import wkt

ROOT = Path(r"C:\Users\dhl\data\Thesis\thesis")
DATA_DIR           = ROOT / "Data"
PANEL_DIR          = DATA_DIR / "Panel"
PROTEST_DIR        = DATA_DIR / "Protest_Petitions"
GIS_DIR            = DATA_DIR / "GIS"
ZONING_DIR         = DATA_DIR / "Zoning_Cases"

print("Loading TCAD GeoJSON...")
tcad_geo = gpd.read_file(GIS_DIR / "TCAD" / "tcad_parcels.geojson")
tcad_geo['geo_norm'] = tcad_geo['geo_id'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
tcad_crs = tcad_geo.to_crs(epsg=2277).set_index('geo_norm')
geo_set = set(tcad_crs.index)

print("Loading petitions...")
petitions = pd.read_csv(PROTEST_DIR / "petition_signers_from_pdf.csv", dtype=str)

def normalize_tcad(raw):
    """
    Normalize a raw TCAD string to 10-digit zero-padded.
    Handles:
      - Standard dash format: 02-0214-0807 → 0202140807
      - 10-digit: 0202140807 → 0202140807
      - 14-digit county suffix: 02260401120000 → first 10 digits → 0226040112
    """
    s = str(raw).replace('-','').replace(' ','').replace('.0','')
    s = s.rstrip('0') if len(s) > 10 else s  # strip trailing zeros if too long
    # If still > 10, truncate to 10
    if len(s) > 10:
        s = s[:10]
    return s.zfill(10)

petitions['norm'] = petitions['tcad_normalized'].apply(normalize_tcad)
petitions['norm14'] = petitions['tcad_normalized'].astype(str).str.replace(r'\.0$','',regex=True).str.replace('-','')

# Also try: 14-digit → strip last 4 → zfill(10)
def fix_14digit(raw):
    s = str(raw).replace('-','').replace(' ','')
    if s.endswith('.0'): s = s[:-2]
    if len(s) == 14:
        return s[:10].zfill(10)
    return s.lstrip('0').zfill(10) if len(s) <= 10 else s[:10].zfill(10)

petitions['norm_fixed'] = petitions['tcad_normalized'].apply(fix_14digit)

print("Loading zoning geometry...")
z = pd.read_csv(ZONING_DIR / "Processed_Data" / "CSV" / "enriched_zoning_data_updated.csv", low_memory=False)
mz = pd.read_csv(ZONING_DIR / "Processed_Data" / "CSV" / "zoning_land_use_merged_data.csv", low_memory=False)

def parse_geom(g):
    try:
        if str(g).startswith('{'): return shape(json.loads(str(g).replace("'",'"')))
        return wkt.loads(str(g))
    except: return None

z['geometry']  = z['the_geom'].apply(parse_geom)
mz['geometry'] = mz['the_geom'].apply(parse_geom)
z_valid  = gpd.GeoDataFrame(z.dropna(subset=['geometry']),  geometry='geometry').set_crs("EPSG:4326").to_crs(epsg=2277).set_index('case_number')
mz_valid = gpd.GeoDataFrame(mz.dropna(subset=['geometry']), geometry='geometry').set_crs("EPSG:4326").to_crs(epsg=2277).set_index('case_number')

props = pd.read_csv(PANEL_DIR / "parcel" / "property_universe.csv", dtype=str, low_memory=False)
props['norm'] = props['standardized_tcad_id'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
props_ll = props.dropna(subset=['latitude','longitude']).copy()
props_ll['lat'] = pd.to_numeric(props_ll['latitude'],  errors='coerce')
props_ll['lon'] = pd.to_numeric(props_ll['longitude'], errors='coerce')

def nearest_parcel(lat, lon, max_ft=300):
    pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=2277).geometry.iloc[0]
    dists = tcad_crs.geometry.distance(pt)
    close = dists[dists < max_ft]
    return tcad_crs.loc[close.idxmin(), 'geometry'] if not close.empty else None

def get_dev_geom(cn):
    for src in [z_valid, mz_valid]:
        if cn in src.index:
            row = src.loc[cn]
            if isinstance(row, gpd.GeoDataFrame):
                return row.geometry.iloc[0]
            return row['geometry'] if isinstance(row['geometry'], object) and hasattr(row['geometry'], 'geom_type') else None
    return None

def resolve_signers(case_norms_fixed, case_norms_std):
    """Try fixed 14-digit normalization first, then standard."""
    resolved = {}
    for nf, ns in zip(case_norms_fixed, case_norms_std):
        for norm in [nf, ns]:
            if norm in geo_set:
                g = tcad_crs.loc[norm, 'geometry']
                # geo_norm may not be unique — take first if Series
                if hasattr(g, 'iloc'):
                    g = g.iloc[0]
                resolved[norm] = g
                break
        else:
            # lat/lon fallback
            for norm in [nf, ns]:
                pu = props_ll[props_ll['norm'] == norm]
                if not pu.empty:
                    g = nearest_parcel(float(pu['lat'].iloc[0]), float(pu['lon'].iloc[0]))
                    if g is not None:
                        resolved[norm] = g
                        break
    return resolved

def compute_intensity(cn):
    dev = get_dev_geom(cn)
    if dev is None:
        return None

    sigs = petitions[(petitions['case_number'] == cn) & (petitions['signed'] == '1')]
    if sigs.empty:
        return None

    resolved = resolve_signers(sigs['norm_fixed'].tolist(), sigs['norm'].tolist())
    if not resolved:
        return None

    buf200 = dev.buffer(200)
    within  = sum(1 for g in resolved.values() if buf200.intersects(g))
    outside = len(resolved) - within
    area    = sum(g.area for g in resolved.values())
    intensity = area / dev.area if dev.area > 0 else 0.0

    return {
        'case_number': cn,
        'signers_resolved': len(resolved),
        'signers_within_200ft': within,
        'signers_outside_200ft': outside,
        'dev_area_sqft': dev.area,
        'unofficial_protest_intensity': intensity,
    }

# ── Run on NO_SIGNER_GEOMETRY (20 cases) and UNKNOWN (9 cases) ───────────────
audit = pd.read_csv(ROOT / "Scratch" / "zero_case_audit.csv")
target_cases = audit[audit['reason'].isin(['NO_SIGNER_GEOMETRY','UNKNOWN'])]['case_number'].tolist()
# Add C14-2010-0051 (NOT_IN_PANEL but has 54 signers - worth computing)
extra = ['C14-2010-0051', 'C814-2009-0099']
all_targets = list(set(target_cases + extra))
print(f"\nProcessing {len(all_targets)} target cases...")

results = []
for cn in all_targets:
    r = compute_intensity(cn)
    if r:
        results.append(r)
        print(f"  {cn}: {r['signers_resolved']} signers, intensity={r['unofficial_protest_intensity']:.4f}")
    else:
        print(f"  {cn}: STILL no resolution")

res_df = pd.DataFrame(results)
print(f"\nSuccessfully resolved: {len(res_df)}/{len(all_targets)}")

# Patch into advanced_geometric_petition_intensity.csv
base = pd.read_csv(PROTEST_DIR / "advanced_geometric_petition_intensity.csv")
print(f"Before: {(base['unofficial_protest_intensity'] > 0).sum()} cases > 0")

for _, row in res_df.iterrows():
    cn = row['case_number']
    mask = base['case_number'] == cn
    if mask.any():
        base.loc[mask, 'unofficial_protest_intensity'] = row['unofficial_protest_intensity']
        base.loc[mask, 'spatial_total_parcels'] = row['signers_resolved']
        base.loc[mask, 'signers_within_200ft']  = row['signers_within_200ft']
        base.loc[mask, 'signers_outside_200ft'] = row['signers_outside_200ft']
    else:
        new_row = {col: np.nan for col in base.columns}
        new_row.update({'case_number': cn,
                        'unofficial_protest_intensity': row['unofficial_protest_intensity'],
                        'spatial_total_parcels': row['signers_resolved'],
                        'signers_within_200ft': row['signers_within_200ft'],
                        'signers_outside_200ft': row['signers_outside_200ft']})
        base = pd.concat([base, pd.DataFrame([new_row])], ignore_index=True)

print(f"After:  {(base['unofficial_protest_intensity'] > 0).sum()} cases > 0")
base.to_csv(PROTEST_DIR / "advanced_geometric_petition_intensity.csv", index=False)
print("Saved.")

# Re-run 01c injection
print("\nRe-running 01c injection...")
import importlib.util
spec = importlib.util.spec_from_file_location("eng", ROOT / "Scripts" / "pipeline" / "01c_engineer_advanced_petitions.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.engineer_advanced_petitions()
