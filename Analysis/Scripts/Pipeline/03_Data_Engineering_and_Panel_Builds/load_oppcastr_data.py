#!/usr/bin/env python3
"""
oppcastr Data Loading Pipeline
==============================
Loads Travis County parcel geometries, Census geographies, and protest scores
into the oppcastr Supabase database for the thesis fork of properlytic.

Usage:
    python load_oppcastr_data.py [--step STEP]

Steps:
    1: Create schema + tables
    2: Load Census geometries (tracts, tabblocks, ZCTA)
    3: Load parcel geometries (from LUI 2024)
    4: Build parcel_ladder mapping
    5: Load protest scores into metrics tables
    6: Aggregate to higher geography levels
    7: Create MVT tile functions
    all: Run all steps (default)

Requires:
    - psycopg2-binary
    - geopandas (for shapefile reading)
    - pandas
"""
import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path

import psycopg2
import psycopg2.extras

# ============================================================================
# CONFIGURATION
# ============================================================================
DB_HOST = os.environ.get("OPPCASTR_DB_HOST", "db.lzwuerruoiqdoiycvntf.supabase.co")
DB_PORT = int(os.environ.get("OPPCASTR_DB_PORT", "5432"))
DB_NAME = os.environ.get("OPPCASTR_DB_NAME", "postgres")
DB_USER = os.environ.get("OPPCASTR_DB_USER", "postgres")
DB_PASS = os.environ.get("OPPCASTR_DB_PASS", "Every1sentence!")

THESIS_DIR = Path(__file__).resolve().parent.parent.parent  # thesis root
DATA_DIR = THESIS_DIR / "Data"
LUI_PATH = DATA_DIR / "CoA_Open_Data" / "Land_Use" / "LUI_2024_7vsm-dvxg.csv"
TRACT_DIR = DATA_DIR / "GIS" / "Census" / "tracts"
TABBLOCK_DIR = DATA_DIR / "GIS" / "Census" / "tabblocks"
ZCTA_DIR = DATA_DIR / "GIS" / "Census" / "zcta"
PANEL_DIR = DATA_DIR / "Panel" / "Reference"

SCHEMA = "oppcastr"
BATCH_SIZE = 500

# ============================================================================
# UTILITIES
# ============================================================================
def get_conn():
    """Get database connection, trying multiple endpoints."""
    endpoints = [
        # Direct
        {"host": DB_HOST, "port": DB_PORT, "user": DB_USER},
        # Session pooler
        {"host": "aws-0-us-east-1.pooler.supabase.com", "port": 5432,
         "user": f"postgres.lzwuerruoiqdoiycvntf"},
        # Transaction pooler
        {"host": "aws-0-us-east-1.pooler.supabase.com", "port": 6543,
         "user": f"postgres.lzwuerruoiqdoiycvntf"},
    ]
    for ep in endpoints:
        try:
            conn = psycopg2.connect(
                host=ep["host"], port=ep["port"], dbname=DB_NAME,
                user=ep["user"], password=DB_PASS,
                sslmode="require", connect_timeout=10
            )
            conn.autocommit = True
            print(f"  Connected via {ep['host']}:{ep['port']}")
            return conn
        except Exception as e:
            print(f"  Failed {ep['host']}:{ep['port']}: {e}")
    raise RuntimeError("Could not connect to oppcastr database")


def flush(msg):
    print(msg, flush=True)


# ============================================================================
# STEP 1: Create schema + tables
# ============================================================================
def step1_create_schema(conn):
    flush("[Step 1] Creating schema and tables...")
    cur = conn.cursor()

    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    # Geometry tables
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.geo_parcel_poly (
            acct TEXT PRIMARY KEY,
            prop_id TEXT,
            geom geometry(MultiPolygon, 4326) NOT NULL
        )
    """)
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS geo_parcel_poly_geom_gix
        ON {SCHEMA}.geo_parcel_poly USING gist (geom)
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.geo_tract20_tx (
            geoid TEXT PRIMARY KEY,
            geom geometry(MultiPolygon, 4326) NOT NULL
        )
    """)
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS geo_tract20_tx_geom_gix
        ON {SCHEMA}.geo_tract20_tx USING gist (geom)
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.geo_tabblock20_tx (
            geoid20 TEXT PRIMARY KEY,
            geom geometry(MultiPolygon, 4326) NOT NULL
        )
    """)
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS geo_tabblock20_tx_geom_gix
        ON {SCHEMA}.geo_tabblock20_tx USING gist (geom)
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.geo_zcta20_us (
            zcta5 TEXT PRIMARY KEY,
            geom geometry(MultiPolygon, 4326) NOT NULL
        )
    """)
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS geo_zcta20_us_geom_gix
        ON {SCHEMA}.geo_zcta20_us USING gist (geom)
    """)

    # Parcel ladder (parcel → tabblock → tract → zcta mapping)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.parcel_ladder (
            acct TEXT PRIMARY KEY,
            prop_id TEXT,
            tabblock_geoid20 TEXT,
            tract_geoid20 TEXT,
            zcta5 TEXT
        )
    """)

    # Metrics tables — parcel level
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.metrics_parcel (
            acct TEXT NOT NULL,
            year INTEGER NOT NULL,
            protest_prob DOUBLE PRECISION,
            protest_actual BOOLEAN,
            n INTEGER DEFAULT 1,
            PRIMARY KEY (acct, year)
        )
    """)

    # Metrics — tabblock aggregate
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.metrics_tabblock (
            tabblock_geoid20 TEXT NOT NULL,
            year INTEGER NOT NULL,
            protest_prob DOUBLE PRECISION,
            n INTEGER,
            PRIMARY KEY (tabblock_geoid20, year)
        )
    """)

    # Metrics — tract aggregate
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.metrics_tract (
            tract_geoid20 TEXT NOT NULL,
            year INTEGER NOT NULL,
            protest_prob DOUBLE PRECISION,
            n INTEGER,
            PRIMARY KEY (tract_geoid20, year)
        )
    """)

    # Metrics — zcta aggregate
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.metrics_zcta (
            zcta5 TEXT NOT NULL,
            year INTEGER NOT NULL,
            protest_prob DOUBLE PRECISION,
            n INTEGER,
            PRIMARY KEY (zcta5, year)
        )
    """)

    cur.close()
    flush("  Schema and tables created.")


# ============================================================================
# STEP 2: Load Census geometries
# ============================================================================
def step2_load_census(conn):
    flush("[Step 2] Loading Census geometries...")
    try:
        import geopandas as gpd
    except ImportError:
        flush("  Installing geopandas...")
        os.system(f"{sys.executable} -m pip install geopandas pyproj shapely fiona -q")
        import geopandas as gpd

    cur = conn.cursor()

    # Tracts — filter to Travis County (FIPS 48453)
    shp = list(TRACT_DIR.glob("*.shp"))[0]
    flush(f"  Reading tracts from {shp.name}...")
    gdf = gpd.read_file(shp)
    travis = gdf[gdf["COUNTYFP"] == "453"].copy()  # Travis County
    flush(f"  Travis County tracts: {len(travis)}")

    for _, row in travis.iterrows():
        wkt = row.geometry.wkt
        cur.execute(f"""
            INSERT INTO {SCHEMA}.geo_tract20_tx (geoid, geom)
            VALUES (%s, ST_GeomFromText(%s, 4326))
            ON CONFLICT (geoid) DO NOTHING
        """, (row["GEOID"], wkt))
    flush(f"  Inserted {len(travis)} tracts.")

    # Tabblocks — filter to Travis County
    shp = list(TABBLOCK_DIR.glob("*.shp"))[0]
    flush(f"  Reading tabblocks from {shp.name}...")
    gdf = gpd.read_file(shp)
    travis = gdf[gdf["COUNTYFP20"] == "453"].copy()
    flush(f"  Travis County tabblocks: {len(travis)}")

    count = 0
    for _, row in travis.iterrows():
        wkt = row.geometry.wkt
        cur.execute(f"""
            INSERT INTO {SCHEMA}.geo_tabblock20_tx (geoid20, geom)
            VALUES (%s, ST_GeomFromText(%s, 4326))
            ON CONFLICT (geoid20) DO NOTHING
        """, (row["GEOID20"], wkt))
        count += 1
        if count % 1000 == 0:
            flush(f"    {count} tabblocks...")
    flush(f"  Inserted {count} tabblocks.")

    # ZCTA — filter to Austin-area ZIPs (78xxx)
    shp = list(ZCTA_DIR.glob("*.shp"))[0]
    flush(f"  Reading ZCTA from {shp.name}...")
    gdf = gpd.read_file(shp)
    austin_zips = gdf[gdf["ZCTA5CE20"].str.startswith("78")].copy()
    flush(f"  Austin-area ZCTAs (78xxx): {len(austin_zips)}")

    for _, row in austin_zips.iterrows():
        wkt = row.geometry.wkt
        cur.execute(f"""
            INSERT INTO {SCHEMA}.geo_zcta20_us (zcta5, geom)
            VALUES (%s, ST_GeomFromText(%s, 4326))
            ON CONFLICT (zcta5) DO NOTHING
        """, (row["ZCTA5CE20"], wkt))
    flush(f"  Inserted {len(austin_zips)} ZCTAs.")

    cur.close()


# ============================================================================
# STEP 3: Load parcel geometries from LUI 2024
# ============================================================================
def step3_load_parcels(conn):
    flush("[Step 3] Loading parcel geometries from LUI 2024...")
    csv.field_size_limit(10**8)
    cur = conn.cursor()

    count = 0
    skipped = 0
    with open(LUI_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            geom_wkt = row.get("the_geom", "").strip()
            prop_id = row.get("PROP_ID", "").strip()
            if not geom_wkt or not prop_id:
                skipped += 1
                continue

            batch.append((prop_id, prop_id, geom_wkt))
            if len(batch) >= BATCH_SIZE:
                psycopg2.extras.execute_values(
                    cur,
                    f"""INSERT INTO {SCHEMA}.geo_parcel_poly (acct, prop_id, geom)
                        VALUES %s ON CONFLICT (acct) DO NOTHING""",
                    batch,
                    template=f"(%s, %s, ST_GeomFromText(%s, 4326))",
                )
                count += len(batch)
                batch = []
                if count % 10000 == 0:
                    flush(f"    {count} parcels loaded...")

        if batch:
            psycopg2.extras.execute_values(
                cur,
                f"""INSERT INTO {SCHEMA}.geo_parcel_poly (acct, prop_id, geom)
                    VALUES %s ON CONFLICT (acct) DO NOTHING""",
                batch,
                template=f"(%s, %s, ST_GeomFromText(%s, 4326))",
            )
            count += len(batch)

    cur.close()
    flush(f"  Loaded {count} parcels (skipped {skipped} without geometry).")


# ============================================================================
# STEP 4: Build parcel_ladder (spatial join parcels → census geos)
# ============================================================================
def step4_build_ladder(conn):
    flush("[Step 4] Building parcel_ladder (spatial join)...")
    cur = conn.cursor()

    # Use ST_Intersects centroid to assign each parcel to its containing
    # tabblock, tract, and ZCTA
    cur.execute(f"""
        INSERT INTO {SCHEMA}.parcel_ladder (acct, prop_id, tabblock_geoid20, tract_geoid20, zcta5)
        SELECT
            p.acct,
            p.prop_id,
            tb.geoid20,
            t.geoid,
            z.zcta5
        FROM {SCHEMA}.geo_parcel_poly p
        LEFT JOIN {SCHEMA}.geo_tabblock20_tx tb
            ON ST_Intersects(ST_Centroid(p.geom), tb.geom)
        LEFT JOIN {SCHEMA}.geo_tract20_tx t
            ON ST_Intersects(ST_Centroid(p.geom), t.geom)
        LEFT JOIN {SCHEMA}.geo_zcta20_us z
            ON ST_Intersects(ST_Centroid(p.geom), z.geom)
        ON CONFLICT (acct) DO UPDATE SET
            tabblock_geoid20 = EXCLUDED.tabblock_geoid20,
            tract_geoid20 = EXCLUDED.tract_geoid20,
            zcta5 = EXCLUDED.zcta5
    """)

    cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.parcel_ladder")
    n = cur.fetchone()[0]
    cur.close()
    flush(f"  Built ladder for {n} parcels.")


# ============================================================================
# STEP 5: Load protest scores
# ============================================================================
def step5_load_protest_scores(conn):
    flush("[Step 5] Loading protest scores...")
    cur = conn.cursor()

    # Find per-parcel scores from experiment results
    # Try multiple possible locations
    score_paths = [
        THESIS_DIR / "Analysis" / "Results" / "Experiments" / "exp02_isotonic" / "per_parcel_scores.csv",
        THESIS_DIR / "Analysis" / "Results" / "Diagnostics" / "per_parcel_scores.csv",
        THESIS_DIR / "Analysis" / "Results" / "per_parcel_protest_scores.csv",
    ]

    score_path = None
    for p in score_paths:
        if p.exists():
            score_path = p
            break

    if score_path is None:
        flush("  WARNING: No per-parcel scores file found. Skipping.")
        flush("  Searched: " + ", ".join(str(p) for p in score_paths))
        cur.close()
        return

    flush(f"  Loading from {score_path}...")
    count = 0
    with open(score_path) as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            acct = row.get("acct", row.get("PROP_ID", "")).strip()
            year = int(row.get("year", row.get("eval_year", "2024")))
            prob = float(row.get("protest_prob", row.get("prob", row.get("calibrated_prob", "0"))))
            actual = row.get("protest", row.get("actual", "")).strip()
            actual_bool = actual == "1" or actual.lower() == "true" if actual else None

            batch.append((acct, year, prob, actual_bool))
            if len(batch) >= BATCH_SIZE:
                psycopg2.extras.execute_values(
                    cur,
                    f"""INSERT INTO {SCHEMA}.metrics_parcel (acct, year, protest_prob, protest_actual)
                        VALUES %s ON CONFLICT (acct, year) DO UPDATE SET
                        protest_prob = EXCLUDED.protest_prob,
                        protest_actual = EXCLUDED.protest_actual""",
                    batch,
                )
                count += len(batch)
                batch = []
                if count % 50000 == 0:
                    flush(f"    {count} scores loaded...")

        if batch:
            psycopg2.extras.execute_values(
                cur,
                f"""INSERT INTO {SCHEMA}.metrics_parcel (acct, year, protest_prob, protest_actual)
                    VALUES %s ON CONFLICT (acct, year) DO UPDATE SET
                    protest_prob = EXCLUDED.protest_prob,
                    protest_actual = EXCLUDED.protest_actual""",
                batch,
            )
            count += len(batch)

    cur.close()
    flush(f"  Loaded {count} protest scores.")


# ============================================================================
# STEP 6: Aggregate to higher geographies
# ============================================================================
def step6_aggregate(conn):
    flush("[Step 6] Aggregating protest scores to higher geographies...")
    cur = conn.cursor()

    # Tabblock
    cur.execute(f"""
        INSERT INTO {SCHEMA}.metrics_tabblock (tabblock_geoid20, year, protest_prob, n)
        SELECT pl.tabblock_geoid20, mp.year,
            AVG(mp.protest_prob)::float8, COUNT(*)::int
        FROM {SCHEMA}.metrics_parcel mp
        JOIN {SCHEMA}.parcel_ladder pl ON pl.acct = mp.acct
        WHERE pl.tabblock_geoid20 IS NOT NULL
        GROUP BY pl.tabblock_geoid20, mp.year
        ON CONFLICT (tabblock_geoid20, year) DO UPDATE SET
            protest_prob = EXCLUDED.protest_prob,
            n = EXCLUDED.n
    """)
    cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.metrics_tabblock")
    flush(f"  Tabblock aggregates: {cur.fetchone()[0]}")

    # Tract
    cur.execute(f"""
        INSERT INTO {SCHEMA}.metrics_tract (tract_geoid20, year, protest_prob, n)
        SELECT pl.tract_geoid20, mp.year,
            AVG(mp.protest_prob)::float8, COUNT(*)::int
        FROM {SCHEMA}.metrics_parcel mp
        JOIN {SCHEMA}.parcel_ladder pl ON pl.acct = mp.acct
        WHERE pl.tract_geoid20 IS NOT NULL
        GROUP BY pl.tract_geoid20, mp.year
        ON CONFLICT (tract_geoid20, year) DO UPDATE SET
            protest_prob = EXCLUDED.protest_prob,
            n = EXCLUDED.n
    """)
    cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.metrics_tract")
    flush(f"  Tract aggregates: {cur.fetchone()[0]}")

    # ZCTA
    cur.execute(f"""
        INSERT INTO {SCHEMA}.metrics_zcta (zcta5, year, protest_prob, n)
        SELECT pl.zcta5, mp.year,
            AVG(mp.protest_prob)::float8, COUNT(*)::int
        FROM {SCHEMA}.metrics_parcel mp
        JOIN {SCHEMA}.parcel_ladder pl ON pl.acct = mp.acct
        WHERE pl.zcta5 IS NOT NULL
        GROUP BY pl.zcta5, mp.year
        ON CONFLICT (zcta5, year) DO UPDATE SET
            protest_prob = EXCLUDED.protest_prob,
            n = EXCLUDED.n
    """)
    cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.metrics_zcta")
    flush(f"  ZCTA aggregates: {cur.fetchone()[0]}")

    cur.close()


# ============================================================================
# STEP 7: Create MVT tile functions
# ============================================================================
def step7_mvt_functions(conn):
    flush("[Step 7] Creating MVT tile functions...")
    cur = conn.cursor()

    # Generic MVT builder for protest data
    cur.execute(f"""
        CREATE OR REPLACE FUNCTION {SCHEMA}.mvt_protest_generic(
            p_layer_name TEXT,
            p_geom_table TEXT,
            p_geom_key TEXT,
            p_metrics_table TEXT,
            p_metrics_key TEXT,
            z INTEGER, x INTEGER, y INTEGER,
            p_year INTEGER DEFAULT 2024,
            p_limit INTEGER DEFAULT NULL
        )
        RETURNS bytea
        LANGUAGE plpgsql STABLE
        AS $$
        DECLARE
            v_sql TEXT;
            v_mvt bytea;
            v_limit TEXT := '';
        BEGIN
            IF p_limit IS NOT NULL AND p_limit > 0 THEN
                v_limit := format(' LIMIT %s', p_limit);
            END IF;

            v_sql := format($fmt$
                WITH bounds AS (
                    SELECT ST_TileEnvelope($1,$2,$3) AS b3857,
                           ST_Transform(ST_TileEnvelope($1,$2,$3),4326) AS b4326
                ),
                src AS (
                    SELECT
                        g.%1$I::text AS id,
                        m.year,
                        m.protest_prob AS value,
                        m.protest_prob AS p50,
                        m.n,
                        ST_AsMVTGeom(ST_Transform(g.geom,3857), bounds.b3857, 4096, 256, true) AS geom
                    FROM %2$s g
                    JOIN %3$s m ON m.%4$I = g.%1$I
                    CROSS JOIN bounds
                    WHERE g.geom && bounds.b4326
                      AND ST_Intersects(g.geom, bounds.b4326)
                      AND m.year = $4
                    %5$s
                )
                SELECT ST_AsMVT(src, %6$L, 4096, 'geom') FROM src
            $fmt$,
                p_geom_key,
                p_geom_table,
                p_metrics_table,
                p_metrics_key,
                v_limit,
                p_layer_name
            );

            EXECUTE v_sql USING z, x, y, p_year INTO v_mvt;
            RETURN COALESCE(v_mvt, ''::bytea);
        END;
        $$
    """)

    # Router function (matches properlytic's zoom-level routing)
    cur.execute(f"""
        CREATE OR REPLACE FUNCTION {SCHEMA}.mvt_choropleth_protest(
            z INTEGER, x INTEGER, y INTEGER,
            p_year INTEGER DEFAULT 2024,
            p_level_override TEXT DEFAULT NULL,
            p_parcel_limit INTEGER DEFAULT 3500
        )
        RETURNS bytea
        LANGUAGE plpgsql STABLE
        AS $$
        BEGIN
            IF p_level_override IS NOT NULL THEN
                CASE lower(p_level_override)
                    WHEN 'zcta' THEN
                        RETURN {SCHEMA}.mvt_protest_generic('zcta', '{SCHEMA}.geo_zcta20_us', 'zcta5', '{SCHEMA}.metrics_zcta', 'zcta5', z, x, y, p_year);
                    WHEN 'tract' THEN
                        RETURN {SCHEMA}.mvt_protest_generic('tract', '{SCHEMA}.geo_tract20_tx', 'geoid', '{SCHEMA}.metrics_tract', 'tract_geoid20', z, x, y, p_year);
                    WHEN 'tabblock' THEN
                        RETURN {SCHEMA}.mvt_protest_generic('tabblock', '{SCHEMA}.geo_tabblock20_tx', 'geoid20', '{SCHEMA}.metrics_tabblock', 'tabblock_geoid20', z, x, y, p_year);
                    WHEN 'parcel' THEN
                        RETURN {SCHEMA}.mvt_protest_generic('parcel', '{SCHEMA}.geo_parcel_poly', 'acct', '{SCHEMA}.metrics_parcel', 'acct', z, x, y, p_year, p_parcel_limit);
                    ELSE
                        RETURN ''::bytea;
                END CASE;
            END IF;

            IF z <= 7 THEN
                RETURN {SCHEMA}.mvt_protest_generic('zcta', '{SCHEMA}.geo_zcta20_us', 'zcta5', '{SCHEMA}.metrics_zcta', 'zcta5', z, x, y, p_year);
            ELSIF z <= 11 THEN
                RETURN {SCHEMA}.mvt_protest_generic('tract', '{SCHEMA}.geo_tract20_tx', 'geoid', '{SCHEMA}.metrics_tract', 'tract_geoid20', z, x, y, p_year);
            ELSIF z <= 16 THEN
                RETURN {SCHEMA}.mvt_protest_generic('tabblock', '{SCHEMA}.geo_tabblock20_tx', 'geoid20', '{SCHEMA}.metrics_tabblock', 'tabblock_geoid20', z, x, y, p_year);
            ELSE
                RETURN {SCHEMA}.mvt_protest_generic('parcel', '{SCHEMA}.geo_parcel_poly', 'acct', '{SCHEMA}.metrics_parcel', 'acct', z, x, y, p_year, p_parcel_limit);
            END IF;
        END;
        $$
    """)

    cur.close()
    flush("  MVT tile functions created.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Load oppcastr data")
    parser.add_argument("--step", default="all", help="Step to run (1-7 or 'all')")
    args = parser.parse_args()

    flush("=" * 60)
    flush("oppcastr Data Loading Pipeline")
    flush("=" * 60)

    conn = get_conn()

    steps = {
        "1": step1_create_schema,
        "2": step2_load_census,
        "3": step3_load_parcels,
        "4": step4_build_ladder,
        "5": step5_load_protest_scores,
        "6": step6_aggregate,
        "7": step7_mvt_functions,
    }

    if args.step == "all":
        for k in sorted(steps):
            steps[k](conn)
    elif args.step in steps:
        steps[args.step](conn)
    else:
        flush(f"Unknown step: {args.step}")
        sys.exit(1)

    conn.close()
    flush("\nDone!")


if __name__ == "__main__":
    main()
