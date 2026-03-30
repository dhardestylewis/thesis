"""
join_irm_environments.py — Map multi-parcel zoning events to panel parcels
==========================================================================
For each closed multi-parcel zoning event (2018–2025), buffer its lat/lon
by 200 m and assign all panel parcels within that buffer to that environment.
Outputs summary statistics for the IRM premise check.

Author: Daniel Hardesty Lewis
Created: 2026-03-09
"""
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import warnings

warnings.filterwarnings('ignore')

PROJECT_DIR = r"c:\Users\dhl\data\thesis\thesis"
PANEL_PATH = os.path.join(PROJECT_DIR, "Data", "Panel", "Output", "Property_Year_Panel_v3.csv")
EVENTS_PATH = os.path.join(PROJECT_DIR, "Data", "Zoning_Cases", "Processed_Data", "multi_parcel_closed_2018_2025.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "Analysis", "Results")


def main():
    # ── 1. Load events (small: 300 rows) ──────────────────────────────────
    print("Loading multi-parcel events...")
    ev = pd.read_csv(EVENTS_PATH)
    ev = ev.dropna(subset=['LATITUDE', 'LONGITUDE'])
    print(f"  Events with coordinates: {len(ev)}")

    ev_geo = gpd.GeoDataFrame(
        ev,
        geometry=[Point(float(x), float(y)) for x, y in zip(ev['LONGITUDE'], ev['LATITUDE'])],
        crs="EPSG:4326"
    ).to_crs("EPSG:32614")
    ev_geo['geometry'] = ev_geo.geometry.buffer(200)  # 200 m buffer

    # ── 2. Load panel (large: 1.7 GB) — only needed columns ──────────────
    print("Loading panel (only needed columns, this takes ~30s)...")
    usecols = ['standardized_tcad_id', 'year', 'latitude', 'longitude',
               'total_market_value', 'zoning_code', 'council_district',
               'zoning_case_on_parcel', 'zoning_case_nearby',
               'lui_land_use', 'lui_general_land_use']
    panel = pd.read_csv(PANEL_PATH, usecols=usecols, low_memory=False)
    print(f"  Panel shape: {panel.shape}")

    # Keep 2018–2025 only
    panel = panel[panel['year'].between(2018, 2025)].copy()
    print(f"  Panel 2018-2025: {panel.shape}")

    # ── 3. Unique parcels with coordinates ────────────────────────────────
    parcels = panel.drop_duplicates(subset=['standardized_tcad_id']).dropna(subset=['latitude', 'longitude']).copy()
    print(f"  Unique parcels with coordinates: {len(parcels)}")

    parcels_geo = gpd.GeoDataFrame(
        parcels[['standardized_tcad_id', 'latitude', 'longitude']],
        geometry=[Point(float(x), float(y)) for x, y in zip(parcels['longitude'], parcels['latitude'])],
        crs="EPSG:4326"
    ).to_crs("EPSG:32614")

    # ── 4. Spatial join ───────────────────────────────────────────────────
    print("Spatial join (parcels × event buffers)...")
    joined = gpd.sjoin(
        parcels_geo,
        ev_geo[['CASE_NUMBER', 'CASE_NAME', 'SUB_TYPE', 'APPLICATION_START_DATE', 'geometry']],
        how='inner',
        predicate='within'
    )
    print(f"  Raw matches: {len(joined)}")

    # A parcel may fall into multiple buffers; keep earliest event
    joined = joined.sort_values('APPLICATION_START_DATE').drop_duplicates(subset=['standardized_tcad_id'])
    print(f"  Unique parcels assigned to an environment: {len(joined)}")

    # ── 5. Summary stats ─────────────────────────────────────────────────
    env_counts = joined.groupby('CASE_NUMBER').size().sort_values(ascending=False)
    print(f"\n{'='*60}")
    print("IRM ENVIRONMENT PREMISE CHECK")
    print(f"{'='*60}")
    print(f"Total environments (closed multi-parcel cases): {len(env_counts)}")
    print(f"Total treated parcels: {len(joined):,}")
    print(f"Avg parcels/environment: {env_counts.mean():.1f}")
    print(f"Median parcels/environment: {env_counts.median():.1f}")
    print(f"Max parcels/environment: {env_counts.max()}")
    print(f"Environments with ≥5 parcels: {(env_counts >= 5).sum()}")
    print(f"Environments with ≥10 parcels: {(env_counts >= 10).sum()}")
    print(f"Environments with ≥20 parcels: {(env_counts >= 20).sum()}")

    # Top 15 environments
    print(f"\nTop 15 environments by parcel count:")
    top15 = joined.groupby(['CASE_NUMBER', 'CASE_NAME', 'SUB_TYPE']).size().sort_values(ascending=False).head(15)
    for (cn, name, st), cnt in top15.items():
        print(f"  {cnt:4d}  {cn:20s}  {st:35s}  {name}")

    # By sub_type
    print(f"\nBy sub_type:")
    by_type = joined.groupby('SUB_TYPE').agg(
        n_envs=('CASE_NUMBER', 'nunique'),
        n_parcels=('standardized_tcad_id', 'nunique')
    ).sort_values('n_parcels', ascending=False)
    print(by_type.to_string())

    # ── 6. Risk variance check ────────────────────────────────────────────
    # Merge total_market_value back to treated parcels
    env_map = joined[['standardized_tcad_id', 'CASE_NUMBER', 'SUB_TYPE', 'APPLICATION_START_DATE']]
    treated = panel.merge(env_map, on='standardized_tcad_id', how='inner')

    tmv = treated.dropna(subset=['total_market_value'])
    if len(tmv) > 0:
        tmv['log_tmv'] = np.log1p(tmv['total_market_value'].clip(lower=0))
        env_risk = tmv.groupby('CASE_NUMBER')['log_tmv'].agg(['mean', 'std', 'count'])
        env_risk = env_risk[env_risk['count'] >= 5]

        print(f"\nRisk (log total_market_value) across {len(env_risk)} environments (≥5 obs):")
        print(f"  Mean of env means:  {env_risk['mean'].mean():.4f}")
        print(f"  Std of env means:   {env_risk['mean'].std():.4f}  ← variance across environments")
        print(f"  Mean within-env std: {env_risk['std'].mean():.4f}")
        ratio = env_risk['mean'].std() / env_risk['std'].mean() if env_risk['std'].mean() > 0 else float('inf')
        print(f"  Between/within ratio: {ratio:.4f}")
        print(f"  (>1 means environments explain more variance than noise → IRM premise supported)")

    # ── 7. Save environment assignment ────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "irm_environment_assignments.csv")
    env_map.to_csv(out_path, index=False)
    print(f"\nSaved environment assignments to {out_path}")

    # Also save the summary
    summary_path = os.path.join(OUT_DIR, "irm_premise_check.txt")
    with open(summary_path, 'w') as f:
        f.write(f"IRM Environment Premise Check\n")
        f.write(f"Generated: 2026-03-09\n\n")
        f.write(f"Total environments: {len(env_counts)}\n")
        f.write(f"Total treated parcels: {len(joined):,}\n")
        f.write(f"Avg parcels/env: {env_counts.mean():.1f}\n")
        f.write(f"Median parcels/env: {env_counts.median():.1f}\n")
        f.write(f"Envs with >=5 parcels: {(env_counts >= 5).sum()}\n")
        f.write(f"Envs with >=10 parcels: {(env_counts >= 10).sum()}\n")
        f.write(f"Envs with >=20 parcels: {(env_counts >= 20).sum()}\n")
        if len(tmv) > 0 and len(env_risk) > 0:
            f.write(f"\nBetween-env std (log TMV): {env_risk['mean'].std():.4f}\n")
            f.write(f"Within-env std (log TMV): {env_risk['std'].mean():.4f}\n")
            f.write(f"Between/within ratio: {ratio:.4f}\n")
    print(f"Saved summary to {summary_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
