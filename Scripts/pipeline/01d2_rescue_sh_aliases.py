"""
01d2_rescue_sh_aliases.py
==========================
Rescues the 8 cases whose petition was filed under the base case number
but whose development is tracked in the panel under the .SH suffix variant.

Strategy: compute intensity using base case signers + .SH development geometry,
then inject into the .SH panel rows.
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

# ── Alias map: petition filed under base, development tracked under .SH ─────
ALIAS_MAP = {
    'C14-2016-0063': 'C14-2016-0063.SH',
    'C14-2018-0100': 'C14-2018-0100.SH',
    'C14-2021-0008': 'C14-2021-0008.SH',
    'C14-2008-0057': 'C14-2008-0057.SH',
    'C14-2016-0023': 'C14-2016-0023.SH',
    'C14-2022-0018': 'C14-2022-0018.SH',
    'C14-2014-0031': 'C14-2014-0031.SH',
    'C14-2023-0007': 'C14-2023-0007.SH',
}

print("Loading datasets...")
petitions = pd.read_csv(PROTEST_PETITIONS_DIR / "petition_signers_from_pdf.csv", dtype=str)
petitions['norm'] = petitions['tcad_normalized'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
petitions = petitions[petitions['signed'] == '1'].copy()

# TCAD GeoJSON – primary signer geometry source
tcad_geo = gpd.read_file(GIS_DIR / "TCAD" / "tcad_parcels.geojson")
tcad_geo['geo_norm'] = tcad_geo['geo_id'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
tcad_crs  = tcad_geo.to_crs(epsg=2277).set_index('geo_norm')

# Zoning for .SH development polygons
def parse_geom(g_str):
    try:
        if str(g_str).startswith('{'):
            return shape(json.loads(str(g_str).replace("'",'"')))
        return wkt.loads(str(g_str))
    except:
        return None

z = pd.read_csv(ZONING_CASES_DIR / "Processed_Data" / "CSV" / "enriched_zoning_data_updated.csv", low_memory=False)
z['geometry'] = z['the_geom'].apply(parse_geom)
z_valid = gpd.GeoDataFrame(z.dropna(subset=['geometry']), geometry='geometry').set_crs("EPSG:4326").to_crs("EPSG:2277")
z_valid = z_valid.set_index('case_number')

# property_universe lat/lon fallback
props = pd.read_csv(PANEL_DIR / "parcel" / "property_universe.csv", dtype=str, low_memory=False)
props['norm'] = props['standardized_tcad_id'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
props_ll = props.dropna(subset=['latitude','longitude']).copy()
props_ll['latitude']  = pd.to_numeric(props_ll['latitude'],  errors='coerce')
props_ll['longitude'] = pd.to_numeric(props_ll['longitude'], errors='coerce')

def nearest_parcel(lat, lon, max_ft=300):
    pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=2277).geometry.iloc[0]
    dists = tcad_crs.geometry.distance(pt)
    close = dists[dists < max_ft]
    return tcad_crs.loc[close.idxmin(), 'geometry'] if not close.empty else None

# ── Compute intensity for each aliased case ───────────────────────────────────
results = []

for base_cn, sh_cn in ALIAS_MAP.items():
    signers = petitions[petitions['case_number'] == base_cn].copy()
    if signers.empty:
        print(f"  {base_cn}: no signers")
        continue

    # Get .SH development polygon
    if sh_cn not in z_valid.index:
        print(f"  {base_cn} -> {sh_cn}: no .SH geometry found")
        continue
    dev_geom = z_valid.loc[sh_cn]
    if isinstance(dev_geom, gpd.GeoDataFrame):
        dev_poly = dev_geom.geometry.iloc[0]
    else:
        dev_poly = dev_geom.geometry
    dev_area_sqft = dev_poly.area

    # Resolve signer geometries: geo_id first, lat/lon fallback
    signer_geoms = {}
    for norm in signers['norm'].unique():
        if norm in tcad_crs.index:
            signer_geoms[norm] = tcad_crs.loc[norm, 'geometry']
        else:
            pu = props_ll[props_ll['norm'] == norm]
            if not pu.empty:
                g = nearest_parcel(float(pu['latitude'].iloc[0]), float(pu['longitude'].iloc[0]))
                if g is not None:
                    signer_geoms[norm] = g

    if not signer_geoms:
        print(f"  {base_cn}: no signer geometries resolved")
        continue

    buffer_200 = dev_poly.buffer(200)
    within  = [g for g in signer_geoms.values() if buffer_200.intersects(g)]
    outside = [g for g in signer_geoms.values() if not buffer_200.intersects(g)]

    total_signer_area = sum(g.area for g in within) + sum(g.area for g in outside)
    intensity = total_signer_area / dev_area_sqft if dev_area_sqft > 0 else 0.0

    results.append({
        'base_case_number': base_cn,
        'panel_case_number': sh_cn,
        'signers_resolved': len(signer_geoms),
        'signers_within_200ft': len(within),
        'signers_outside_200ft': len(outside),
        'dev_area_sqft': dev_area_sqft,
        'total_signer_area_sqft': total_signer_area,
        'unofficial_protest_intensity': intensity,
    })
    print(f"  {base_cn} -> {sh_cn}: {len(signer_geoms)} signers, intensity={intensity:.4f}")

res_df = pd.DataFrame(results)

# ── Patch into advanced_geometric_petition_intensity.csv under .SH name ──────
base_int = pd.read_csv(PROTEST_PETITIONS_DIR / "advanced_geometric_petition_intensity.csv")
print(f"\nBefore patch: {(base_int['unofficial_protest_intensity'] > 0).sum()} cases > 0")

for _, row in res_df.iterrows():
    sh = row['panel_case_number']
    mask = base_int['case_number'] == sh
    if mask.any():
        base_int.loc[mask, 'unofficial_protest_intensity'] = row['unofficial_protest_intensity']
        base_int.loc[mask, 'spatial_total_parcels'] = row['signers_resolved']
        base_int.loc[mask, 'signers_within_200ft']  = row['signers_within_200ft']
        base_int.loc[mask, 'signers_outside_200ft'] = row['signers_outside_200ft']
    else:
        new_row = {col: np.nan for col in base_int.columns}
        new_row['case_number'] = sh
        new_row['unofficial_protest_intensity'] = row['unofficial_protest_intensity']
        new_row['spatial_total_parcels'] = row['signers_resolved']
        new_row['signers_within_200ft']  = row['signers_within_200ft']
        new_row['signers_outside_200ft'] = row['signers_outside_200ft']
        base_int = pd.concat([base_int, pd.DataFrame([new_row])], ignore_index=True)

print(f"After patch:  {(base_int['unofficial_protest_intensity'] > 0).sum()} cases > 0")
base_int.to_csv(PROTEST_PETITIONS_DIR / "advanced_geometric_petition_intensity.csv", index=False)
print("Saved.")

# ── Re-run 01c injection ─────────────────────────────────────────────────────
print("\nRe-running 01c injection...")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "eng", ROOT / "Scripts" / "pipeline" / "01c_engineer_advanced_petitions.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.engineer_advanced_petitions()
