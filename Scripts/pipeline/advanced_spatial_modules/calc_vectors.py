"""
calc_vectors.py — Spatial vector builder for petition cases
===========================================================
Primary signer geometry source: tcad_parcels.geojson via geo_id (10-digit zero-padded).
Fallback: property_universe.csv lat/lon → nearest parcel within 300ft.

This replaces the old approach that only used property_universe.csv which:
  - Is purely tabular (no geometry for parcels with NaN lat/lon)
  - Does not have the geo_id key (uses standardized_tcad_id which is numeric)
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import time
import json
import os
from shapely.geometry import Point
from config.paths import DATA_DIR, PANEL_DIR, PROTEST_PETITIONS_DIR, GIS_DIR, ZONING_CASES_DIR


def _build_tcad_geo_index(tcad_geo_crs):
    """Build a spatial index and geo_norm lookup from tcad_parcels GeoDataFrame."""
    tcad_geo_crs = tcad_geo_crs.copy()
    tcad_geo_crs['geo_norm'] = tcad_geo_crs['geo_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    tcad_geo_crs['centroid'] = tcad_geo_crs.geometry.centroid
    return tcad_geo_crs.set_index('geo_norm')


def _nearest_parcel(tcad_indexed, lat, lon, max_dist_ft=300):
    """Return geometry of nearest TCAD parcel to a lat/lon within max_dist_ft."""
    pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=2277).geometry.iloc[0]
    dists = tcad_indexed['centroid'].distance(pt)
    close = dists[dists < max_dist_ft]
    if close.empty:
        return None
    return tcad_indexed.loc[close.idxmin(), 'geometry']


def resolve_signer_geoms(signer_norms, tcad_indexed, props_latlon):
    """
    For a list of 10-digit zero-padded TCAD IDs, return a dict of {norm: geometry_ft}.
    Priority:
      1. Direct geo_id match in tcad_parcels.geojson (exact polygon)
      2. property_universe lat/lon → nearest parcel centroid within 300ft
    """
    resolved = {}
    unmatched = []

    for norm in signer_norms:
        if norm in tcad_indexed.index:
            resolved[norm] = tcad_indexed.loc[norm, 'geometry']
        else:
            unmatched.append(norm)

    # Fallback: lat/lon nearest parcel
    for norm in unmatched:
        pu = props_latlon[props_latlon['norm'] == norm]
        if pu.empty:
            continue
        lat = float(pu['latitude'].iloc[0])
        lon = float(pu['longitude'].iloc[0])
        geom = _nearest_parcel(tcad_indexed, lat, lon, max_dist_ft=300)
        if geom is not None:
            resolved[norm] = geom

    return resolved


def build_spatial_vectors(petitions, tcad_geo, cases_gdf, props=None, out_dir=PROTEST_PETITIONS_DIR):
    """
    tcad_geo: GeoDataFrame of tcad_parcels.geojson already projected to EPSG:2277
    cases_gdf: GeoDataFrame of development polygons projected to EPSG:2277
    """
    # Build lookup index
    tcad_indexed = _build_tcad_geo_index(tcad_geo)

    # Build props lat/lon fallback table
    if props is not None:
        props_latlon = props.dropna(subset=['latitude', 'longitude']).copy()
        props_latlon['norm'] = props_latlon.index.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
        props_latlon['latitude']  = pd.to_numeric(props_latlon['latitude'],  errors='coerce')
        props_latlon['longitude'] = pd.to_numeric(props_latlon['longitude'], errors='coerce')
        props_latlon = props_latlon.reset_index()
    else:
        props_latlon = pd.DataFrame(columns=['norm','latitude','longitude'])

    signed_cases = petitions['case_number'].unique()
    print(f"1. Computing spatial distance vectors for {len(signed_cases)} protested cases...")
    results = []
    t0 = time.time()

    for idx, case in enumerate(signed_cases):
        if case not in cases_gdf.index:
            continue

        case_geom = cases_gdf.loc[case, 'geometry']
        if isinstance(case_geom, pd.Series):
            case_geom = case_geom.iloc[0]

        # Normalize signer TCAD IDs
        raw = petitions[petitions['case_number'] == case]['tcad_normalized'].dropna().astype(str)
        signer_norms = list(raw.str.replace(r'\.0$','',regex=True).str.replace('-','',regex=False).str.zfill(10).unique())

        # Resolve signer geometries (geo_id first, lat/lon fallback)
        geom_map = resolve_signer_geoms(signer_norms, tcad_indexed, props_latlon)
        if not geom_map:
            continue

        signer_geoms_series = gpd.GeoSeries(list(geom_map.values()), crs="EPSG:2277")
        distances = signer_geoms_series.distance(case_geom).values

        dist_vector = np.round(distances, 2).tolist()
        results.append({
            'case_number': case,
            'signer_distance_vector': json.dumps(dist_vector),
            'min_signer_dist':   float(np.min(distances)),
            'max_signer_dist':   float(np.max(distances)),
            'median_signer_dist': float(np.median(distances)),
            'signers_within_200ft':  int(np.sum(distances <= 200)),
            'signers_outside_200ft': int(np.sum(distances > 200)),
            'unofficial_protest_intensity': len(distances),
        })

        if idx % 50 == 0:
            print(f"   {idx}/{len(signed_cases)} — {len(geom_map)}/{len(signer_norms)} signers resolved")

    res_df = pd.DataFrame(results)
    print(f"   Completed in {time.time() - t0:.1f}s | {len(res_df)} cases with resolved signers")

    print("Merging with spatial summary base...")
    geo = pd.read_csv(PROTEST_PETITIONS_DIR / "petition_summary_spatial_true.csv")
    merged = pd.merge(geo, res_df, on='case_number', how='left')

    out_path = PROTEST_PETITIONS_DIR / "advanced_geometric_petition_intensity.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved advanced spatial vectors → {out_path}")
