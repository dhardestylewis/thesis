"""
enrich_panel.py — Time-Varying Data Enrichment
================================================
Adds time-varying census data and forward-fills land use / Land Database
covariates across all property-years.

Design Decisions:
  1. Census/ACS: Pulled per vintage year at tract level via Census API.
     Panel year → ACS vintage mapping: use same year when available (2009+),
     else nearest available vintage. ACS 5-year estimates lag ~2 years
     (e.g., 2022 ACS released late 2023, covers 2018-2022).

  2. Land Use: Forward-fill from nearest prior snapshot.
     Available snapshots: 2012, 2016, 2021, 2022, 2024.
     For years before 2012, backward-fill from 2012.

  3. Land Database: Rich covariates (zoning, appraisal, FAR, lot size) from
     2016 and 2021 snapshots. Forward-fill between and beyond.

  4. All forward-fills carry a `_source_year` column so the analyst knows
     which snapshot the data came from.

Outputs:
  - Data/Panel/census_tract_timeseries.csv  (ACS data by tract × year)
  - Data/Panel/Property_Year_Panel_Enriched.csv  (final enriched panel)

Author: Daniel Hardesty Lewis
Created: 2026-02-16
"""

import csv
import os
import sys
import json
import logging
import time
from collections import defaultdict
from datetime import datetime

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "Data")
PANEL_DIR = os.path.join(DATA_DIR, "Panel")
COA_DIR = os.path.join(DATA_DIR, "CoA_Open_Data")
PANEL_PATH = os.path.join(PANEL_DIR, "Property_Year_Panel.csv")

# Census API
CENSUS_API_BASE = "https://api.census.gov/data"
CENSUS_STATE = "48"   # Texas
CENSUS_COUNTY = "453"  # Travis County
# ACS 5-year estimates available from 2009 onward
ACS_VINTAGES = list(range(2009, 2024))  # 2009..2023

# ACS variables to pull (B-table codes → friendly names)
ACS_VARIABLES = {
    "B01003_001E": "total_population",
    "B01002_001E": "median_age",
    "B02001_002E": "race_white",
    "B02001_003E": "race_black",
    "B02001_005E": "race_asian",
    "B03003_003E": "race_hispanic",
    "B19013_001E": "median_household_income",
    "B17001_002E": "poverty_count",
    "B25077_001E": "median_home_value",
    "B25003_002E": "owner_occupied_units",
    "B25003_003E": "renter_occupied_units",
    "B25064_001E": "median_gross_rent",
    "B25001_001E": "total_housing_units",
}

# Land Database files (with PID_10 key)
LDB_FILES = {
    2016: os.path.join(COA_DIR, "LDB_2016_4nsn-uea6.csv"),
    2021: os.path.join(COA_DIR, "LDB_2021_kk8y-6cmt.csv"),
}

# Columns to extract from Land Databases (using 2016 names; 2021 has truncated names)
LDB_COLS_2016 = {
    "PID_10": "pid_10",
    "BASEZONE": "ldb_basezone",
    "EFF_ZONE": "ldb_eff_zone",
    "LOTSIZE": "ldb_lotsize",
    "COUNCIL_DI": "ldb_council_district",
    "FAR": "ldb_far",
    "ILR": "ldb_ilr",
    "UNITS": "ldb_units",
    "YR_BUILT": "ldb_yr_built",
    "LAND_ACRES": "ldb_land_acres",
    "SUM_IMPRV_SQFT": "ldb_imprv_sqft",
    "APPRAISED_VAL": "ldb_appraised_val",
    "MARKET_VAL": "ldb_market_val",
    "LAND_USE": "ldb_land_use",
    "GEN_LAND_USE": "ldb_gen_land_use",
    "LU_DESC": "ldb_lu_desc",
    "GEN_LU_DESC": "ldb_gen_lu_desc",
    "CONSTRAINED_AREA": "ldb_constrained_area",
    "I35SIDE": "ldb_i35side",
    "IMPRV_TYPE_DESC": "ldb_imprv_type_desc",
}

# 2021 Land Database has truncated column names — map them
LDB_COLS_2021 = {
    "PID_10": "pid_10",
    "BASEZONE": "ldb_basezone",
    # 2021 truncates EFF_ZONE, no exact match — skip or check
    "ILR": "ldb_ilr",
    "COUNCIL_DI": "ldb_council_district",
    "FAR": "ldb_far",
    "UNITS": "ldb_units",
    "YR_BUILT": "ldb_yr_built",
    "LAND_ACRES": "ldb_land_acres",
    "SUM_IMPRV_": "ldb_imprv_sqft",
    "APPRAISED_": "ldb_appraised_val",
    "MARKET_VAL": "ldb_market_val",
    "LAND_USE": "ldb_land_use",
    "GENERAL_LA": "ldb_gen_land_use",
    "LU_DESC": "ldb_lu_desc",
    "GEN_LU_DES": "ldb_gen_lu_desc",
    "CONSTRAINE": "ldb_constrained_area",
    "I35SIDE": "ldb_i35side",
    "IMPRV_TYPE": "ldb_imprv_type_desc",
}

# LUI snapshots for forward-fill of land use codes
LUI_FILES = {
    2012: os.path.join(COA_DIR, "LUI_2012_3k7r-w54d.csv"),
    2022: os.path.join(COA_DIR, "LUI_2022_6qkk-xgys.csv"),
    2024: os.path.join(COA_DIR, "LUI_2024_7vsm-dvxg.csv"),
}

# Panel year range
YEAR_MIN = 2007
YEAR_MAX = 2024

# Logging
LOG_PATH = os.path.join(PANEL_DIR, "enrich_panel.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step A: Pull ACS Census Data
# ---------------------------------------------------------------------------
def step_a_census_api():
    """Pull ACS 5-year estimates from Census API for all vintage years."""
    log.info("=" * 60)
    log.info("STEP A: Pulling ACS 5-Year Estimates from Census API")
    log.info("=" * 60)

    if not HAS_REQUESTS:
        log.error("requests library not installed. Run: pip install requests")
        return {}

    # Collect all unique tract GEOIDs from property universe
    universe_path = os.path.join(PANEL_DIR, "property_universe.csv")
    tracts = set()
    with open(universe_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for geoid_col in ['zoning_case_GEOID', 'nearby_GEOID']:
                geoid = row.get(geoid_col, '').strip()
                if geoid and len(geoid) >= 11:
                    # GEOID format: SSCCCTTTTTT (state+county+tract)
                    tracts.add(geoid)
    log.info(f"Unique tract GEOIDs in panel: {len(tracts)}")

    # Extract unique tract codes (6-digit) within Travis County
    tract_codes = set()
    for geoid in tracts:
        if len(geoid) >= 11:
            tract_codes.add(geoid[-6:])  # last 6 digits = tract
    log.info(f"Unique Travis County tracts: {len(tract_codes)}")

    # Pull data for each vintage year
    var_list = ",".join(ACS_VARIABLES.keys())
    census_data = {}  # (geoid_11, vintage) -> {var: val}

    for vintage in ACS_VINTAGES:
        url = (
            f"{CENSUS_API_BASE}/{vintage}/acs/acs5"
            f"?get=NAME,{var_list}"
            f"&for=tract:*"
            f"&in=state:{CENSUS_STATE}&in=county:{CENSUS_COUNTY}"
        )

        log.info(f"  Pulling ACS {vintage}...")
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                headers = data[0]
                rows = data[1:]

                # Find column indices
                state_idx = headers.index("state")
                county_idx = headers.index("county")
                tract_idx = headers.index("tract")

                var_indices = {}
                for api_var in ACS_VARIABLES.keys():
                    if api_var in headers:
                        var_indices[api_var] = headers.index(api_var)

                for row in rows:
                    state = row[state_idx]
                    county = row[county_idx]
                    tract = row[tract_idx]
                    geoid_11 = f"{state}{county}{tract}"

                    record = {"vintage": vintage, "geoid": geoid_11}
                    for api_var, idx in var_indices.items():
                        friendly = ACS_VARIABLES[api_var]
                        val = row[idx]
                        # Census uses negative values for missing/suppressed
                        if val and val not in ('-666666666', '-888888888', '-999999999', 'null', 'None'):
                            record[friendly] = val
                        else:
                            record[friendly] = ''

                    census_data[(geoid_11, vintage)] = record

                log.info(f"    Got {len(rows)} tracts for vintage {vintage}")
            else:
                log.warning(f"    HTTP {resp.status_code} for vintage {vintage}")

            time.sleep(0.5)  # Rate limit courtesy

        except Exception as e:
            log.warning(f"    Error for vintage {vintage}: {e}")
            time.sleep(1)

    # Write census time series
    census_path = os.path.join(PANEL_DIR, "census_tract_timeseries.csv")
    if census_data:
        friendly_names = list(ACS_VARIABLES.values())
        fieldnames = ['geoid', 'vintage'] + friendly_names

        with open(census_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for key in sorted(census_data.keys()):
                writer.writerow(census_data[key])

        log.info(f"Wrote {len(census_data)} tract-year records to {census_path}")
    else:
        log.warning("No census data retrieved!")

    return census_data


# ---------------------------------------------------------------------------
# Step B: Load Land Database Snapshots
# ---------------------------------------------------------------------------
def step_b_load_land_databases():
    """Load Land Database snapshots for forward-filling."""
    log.info("=" * 60)
    log.info("STEP B: Loading Land Database Snapshots")
    log.info("=" * 60)

    ldb_data = {}  # year -> {pid_10 -> {col: val}}

    for year, filepath in sorted(LDB_FILES.items()):
        if not os.path.exists(filepath):
            log.warning(f"Land Database {year} not found: {filepath}")
            continue

        col_map = LDB_COLS_2016 if year == 2016 else LDB_COLS_2021
        log.info(f"Loading Land Database {year}: {filepath}")

        records = {}
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            # Build mapping from actual header to our friendly name
            actual_map = {}
            for orig_col, friendly in col_map.items():
                # Try exact match first, then case-insensitive
                if orig_col in headers:
                    actual_map[orig_col] = friendly
                else:
                    for h in headers:
                        if h.upper().startswith(orig_col.upper()):
                            actual_map[h] = friendly
                            break

            for row in reader:
                pid = None
                # Find PID_10 column
                for h in headers:
                    if h.upper() in ('PID_10', 'PID10'):
                        pid = row.get(h, '').strip()
                        break

                if not pid:
                    continue

                # Pad PID to 10 digits if needed
                try:
                    pid = str(int(float(pid))).zfill(10) if pid else None
                except (ValueError, OverflowError):
                    pid = pid.zfill(10) if len(pid) < 10 else pid

                if not pid:
                    continue

                record = {}
                for orig_col, friendly in actual_map.items():
                    if friendly == 'pid_10':
                        continue
                    record[friendly] = row.get(orig_col, '')

                records[pid] = record

        ldb_data[year] = records
        log.info(f"  Loaded {len(records):,} parcels for {year}")

    return ldb_data


# ---------------------------------------------------------------------------
# Step C: Load LUI Snapshots for Land Use Forward-Fill
# ---------------------------------------------------------------------------
def step_c_load_lui_snapshots():
    """Load LUI snapshots for land use forward-filling."""
    log.info("=" * 60)
    log.info("STEP C: Loading LUI Snapshots for Land Use Forward-Fill")
    log.info("=" * 60)

    lui_data = {}  # year -> {parcel_id_10 -> {land_use, general_land_use}}

    for year, filepath in sorted(LUI_FILES.items()):
        if not os.path.exists(filepath):
            log.warning(f"LUI {year} not found: {filepath}")
            continue

        log.info(f"Loading LUI {year}: {filepath}")
        records = {}

        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            # Find the parcel ID column (varies by file)
            pid_col = None
            for h in headers:
                if h.upper() in ('PARCEL_ID_10', 'PID_10'):
                    pid_col = h
                    break

            lu_col = None
            glu_col = None
            for h in headers:
                hu = h.upper()
                if hu == 'LAND_USE':
                    lu_col = h
                elif hu in ('GENERAL_LAND_USE', 'GENERAL_LA', 'GEN_LAND_USE'):
                    glu_col = h

            if not pid_col:
                log.warning(f"  No parcel ID column found in {filepath}")
                continue

            for row in reader:
                pid = row.get(pid_col, '').strip()
                if not pid:
                    continue

                # Pad to 10 digits
                try:
                    pid = str(int(float(pid))).zfill(10) if pid else None
                except (ValueError, OverflowError):
                    pid = pid.zfill(10) if len(pid) < 10 else pid

                if not pid:
                    continue

                records[pid] = {
                    'lui_land_use': row.get(lu_col, '') if lu_col else '',
                    'lui_general_land_use': row.get(glu_col, '') if glu_col else '',
                }

        lui_data[year] = records
        log.info(f"  Loaded {len(records):,} parcels for {year}")

    return lui_data


# ---------------------------------------------------------------------------
# Step D: Forward-Fill Helper
# ---------------------------------------------------------------------------
def forward_fill_lookup(data_by_year, pid, panel_year, available_years):
    """Get the nearest prior (or equal) snapshot for a property.

    Strategy:
      - If panel_year >= latest snapshot, use latest
      - If panel_year <= earliest snapshot, use earliest (backward-fill)
      - Otherwise, use the most recent snapshot <= panel_year
    Returns (record_dict, source_year) or (None, None).
    """
    # Find the best matching year
    best_year = None
    for snap_year in sorted(available_years):
        if snap_year <= panel_year:
            best_year = snap_year

    # If no prior year, backward-fill from earliest
    if best_year is None and available_years:
        best_year = min(available_years)

    if best_year is not None and best_year in data_by_year:
        record = data_by_year[best_year].get(pid)
        if record:
            return record, best_year

    return None, None


# ---------------------------------------------------------------------------
# Step E: Enrich Panel
# ---------------------------------------------------------------------------
def step_e_enrich_panel(census_data, ldb_data, lui_data):
    """Merge time-varying data onto the panel."""
    log.info("=" * 60)
    log.info("STEP E: Enriching Panel with Time-Varying Data")
    log.info("=" * 60)

    ldb_years = sorted(ldb_data.keys())
    lui_years = sorted(lui_data.keys())
    log.info(f"Land Database snapshot years: {ldb_years}")
    log.info(f"LUI snapshot years: {lui_years}")

    # Census variable names
    census_vars = list(ACS_VARIABLES.values())

    # LDB variable names (excluding pid_10)
    ldb_vars = []
    if ldb_data:
        first_year = list(ldb_data.values())[0]
        if first_year:
            first_record = list(first_year.values())[0]
            ldb_vars = list(first_record.keys())

    # Read existing panel
    out_path = os.path.join(PANEL_DIR, "Property_Year_Panel_Enriched.csv")
    total = 0
    census_matched = 0
    ldb_matched = 0
    lui_matched = 0

    with open(PANEL_PATH, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        existing_fields = reader.fieldnames

        # Remove old snapshot census columns (we're replacing with time-varying)
        old_census_cols = [
            'zoning_case_total_population', 'zoning_case_median_age',
            'zoning_case_race_white', 'zoning_case_race_black',
            'zoning_case_race_asian', 'zoning_case_race_hispanic',
            'zoning_case_median_income', 'zoning_case_poverty_count',
            'zoning_case_median_home_value',
            'zoning_case_owner_occupied', 'zoning_case_renter_occupied',
            'zoning_case_commute_time',
        ]
        keep_fields = [f for f in existing_fields if f not in old_census_cols]

        # New columns
        new_census = [f"acs_{v}" for v in census_vars] + ['acs_vintage']
        new_ldb = ldb_vars + ['ldb_source_year'] if ldb_vars else []
        new_lui = ['lui_land_use_tv', 'lui_general_land_use_tv', 'lui_source_year']

        out_fields = keep_fields + new_census + new_ldb + new_lui

        with open(out_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields)
            writer.writeheader()

            for row in reader:
                total += 1
                panel_year = int(row['year'])
                tcad_id = row['standardized_tcad_id']

                out_row = {k: row.get(k, '') for k in keep_fields}

                # --- Census: match by GEOID + nearest vintage ---
                geoid = row.get('zoning_case_GEOID', '').strip()
                if not geoid:
                    geoid = row.get('nearby_GEOID', '').strip()

                # Panel GEOIDs may be 12-digit (block group); Census uses 11-digit (tract)
                if geoid and len(geoid) > 11:
                    geoid = geoid[:11]

                # Find best ACS vintage for this panel year
                best_vintage = None
                for v in ACS_VINTAGES:
                    if v <= panel_year:
                        best_vintage = v
                if best_vintage is None and ACS_VINTAGES:
                    best_vintage = min(ACS_VINTAGES)

                census_rec = census_data.get((geoid, best_vintage)) if geoid and best_vintage else None
                if census_rec:
                    for var in census_vars:
                        out_row[f"acs_{var}"] = census_rec.get(var, '')
                    out_row['acs_vintage'] = best_vintage
                    census_matched += 1
                else:
                    for var in census_vars:
                        out_row[f"acs_{var}"] = ''
                    out_row['acs_vintage'] = ''

                # --- Land Database: forward-fill ---
                ldb_rec, ldb_year = forward_fill_lookup(
                    ldb_data, tcad_id, panel_year, ldb_years
                )
                if ldb_rec:
                    for var in ldb_vars:
                        out_row[var] = ldb_rec.get(var, '')
                    out_row['ldb_source_year'] = ldb_year
                    ldb_matched += 1
                else:
                    for var in ldb_vars:
                        out_row[var] = ''
                    out_row['ldb_source_year'] = ''

                # --- LUI: forward-fill ---
                lui_rec, lui_year = forward_fill_lookup(
                    lui_data, tcad_id, panel_year, lui_years
                )
                if lui_rec:
                    out_row['lui_land_use_tv'] = lui_rec.get('lui_land_use', '')
                    out_row['lui_general_land_use_tv'] = lui_rec.get('lui_general_land_use', '')
                    out_row['lui_source_year'] = lui_year
                    lui_matched += 1
                else:
                    out_row['lui_land_use_tv'] = ''
                    out_row['lui_general_land_use_tv'] = ''
                    out_row['lui_source_year'] = ''

                writer.writerow(out_row)

    log.info(f"Total panel rows: {total:,}")
    log.info(f"Census matched: {census_matched:,} ({100*census_matched/max(total,1):.1f}%)")
    log.info(f"LDB matched: {ldb_matched:,} ({100*ldb_matched/max(total,1):.1f}%)")
    log.info(f"LUI time-varying matched: {lui_matched:,} ({100*lui_matched/max(total,1):.1f}%)")
    log.info(f"Enriched panel written to: {out_path}")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start = datetime.now()
    log.info(f"Panel enrichment started at {start.isoformat()}")

    # Step A: Census API
    census_data = step_a_census_api()

    # Step B: Land Databases
    ldb_data = step_b_load_land_databases()

    # Step C: LUI snapshots
    lui_data = step_c_load_lui_snapshots()

    # Step D is a helper function

    # Step E: Enrich panel
    out_path = step_e_enrich_panel(census_data, ldb_data, lui_data)

    elapsed = datetime.now() - start
    log.info("=" * 60)
    log.info(f"DONE. Elapsed: {elapsed}")
    log.info(f"Final output: {out_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
