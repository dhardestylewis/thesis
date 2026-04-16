"""
build_panel.py — Panel Dataset Construction Pipeline
=====================================================
Predicting Zoning Opposition: Property-Year Panel for Causal Analysis

Steps:
  1. Extract property universe from GeoJSON
  2. Create balanced property × year skeleton with protest outcome
  3. Parse & clean EARS appraisal rolls (2019–2022)
  4. Build ID crosswalk (parcel_id_10 → EARS account_number)
  5. Merge EARS onto panel
  6. Merge Land Use Inventory
  7. Add census variables (from GeoJSON)

Outputs:
  - Data/Panel/property_universe.csv
  - Data/Panel/property_year_skeleton.csv
  - Data/Panel/Intermediate/ears_YYYY_clean.parquet  (per year)
  - Data/Panel/id_crosswalk.csv
  - Data/Panel/Property_Year_Panel.csv  (final)

Author: Daniel Hardesty Lewis
Created: 2026-02-16
"""

import json
import csv
import os
import sys
import logging
from collections import defaultdict
from datetime import datetime

# Increase CSV field size limit for Land Use Inventory geometry columns
# On Windows, sys.maxsize may exceed C long range, so use 2^31-1
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "Data")
PANEL_DIR = os.path.join(DATA_DIR, "Panel")
EARS_DIR = os.path.join(DATA_DIR, "Appraisal_Rolls")
GEOJSON_PATH = os.path.join(DATA_DIR, "Protest_Petitions", "GeoJSON", "protest_petitions_v1.geojson")
LUI_PATH = os.path.join(DATA_DIR, "Zoning_Cases", "Source_Data", "land_use_inventory_prefetched.csv")
LUI_2024_PATH = os.path.join(DATA_DIR, "CoA_Open_Data", "LUI_2024_7vsm-dvxg.csv")
LAYOUT_PATH = os.path.join(PANEL_DIR, "EARS_Column_Layout.csv")

# Year scope for the panel
YEAR_MIN = 2007  # earliest status_date year in GeoJSON
YEAR_MAX = 2024  # latest year
EARS_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]  # years with CSV EARS data on disk

# Set up logging
os.makedirs(PANEL_DIR, exist_ok=True)
LOG_PATH = os.path.join(PANEL_DIR, "build_panel.log")
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
# Step 1: Extract Property Universe from LUI + GeoJSON
# ---------------------------------------------------------------------------
def step1_property_universe():
    """Build full property universe from LUI parcels, overlay protest history from GeoJSON."""
    log.info("=" * 60)
    log.info("STEP 1: Extracting Property Universe (Full LUI + GeoJSON protests)")
    log.info("=" * 60)

    # --- Load ALL LUI parcels as the property universe ---
    properties = {}  # parcel_id_10 -> property info
    log.info(f"Loading LUI parcels from: {LUI_PATH}")
    with open(LUI_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid10 = row.get('parcel_id_10', '').strip()
            if not pid10:
                continue
            if pid10 not in properties:
                properties[pid10] = {
                    'standardized_tcad_id': pid10,
                    'lui_land_use': row.get('land_use', ''),
                    'lui_general_land_use': row.get('general_land_use', ''),
                    'lui_shape_area': row.get('shape_area', ''),
                    'latitude': '',
                    'longitude': '',
                    'council_district': '',
                    'nearby_GEOID': '',
                    'zoning_case_GEOID': '',
                }
    log.info(f"LUI parcels loaded: {len(properties):,}")

    # --- Overlay GeoJSON protest properties for metadata + protest history ---
    with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    features = data['features']
    log.info(f"Total GeoJSON features: {len(features)}")

    protest_history = defaultdict(set)  # tcad_id -> set of protest years
    geojson_props_found = 0

    for feat in features:
        p = feat['properties']
        tcad_id = p.get('standardized_tcad_id')
        if not tcad_id:
            continue

        # Extract protest year from status_date
        status_date = p.get('status_date', '')
        if status_date and len(status_date) >= 4:
            try:
                protest_year = int(status_date[:4])
                protest_history[tcad_id].add(protest_year)
            except ValueError:
                pass

        # Add metadata from GeoJSON (first occurrence)
        if tcad_id in properties:
            prop = properties[tcad_id]
            if not prop['latitude']:  # only set once
                geom = feat.get('geometry')
                lat, lon = None, None
                if geom and geom.get('coordinates'):
                    coords = geom['coordinates']
                    if geom['type'] == 'Point':
                        lon, lat = coords[0], coords[1]
                    elif geom['type'] == 'Polygon' and coords:
                        ring = coords[0]
                        lon = sum(c[0] for c in ring) / len(ring)
                        lat = sum(c[1] for c in ring) / len(ring)
                prop['latitude'] = lat or p.get('latitude', '')
                prop['longitude'] = lon or p.get('longitude', '')
                prop['council_district'] = p.get('council_district', '')
                prop['nearby_GEOID'] = p.get('nearby_GEOID', '')
                prop['zoning_case_GEOID'] = p.get('zoning_case_GEOID', '')
                geojson_props_found += 1
        else:
            # GeoJSON property not in LUI — add it anyway
            geom = feat.get('geometry')
            lat, lon = None, None
            if geom and geom.get('coordinates'):
                coords = geom['coordinates']
                if geom['type'] == 'Point':
                    lon, lat = coords[0], coords[1]
                elif geom['type'] == 'Polygon' and coords:
                    ring = coords[0]
                    lon = sum(c[0] for c in ring) / len(ring)
                    lat = sum(c[1] for c in ring) / len(ring)
            properties[tcad_id] = {
                'standardized_tcad_id': tcad_id,
                'lui_land_use': '',
                'lui_general_land_use': '',
                'lui_shape_area': '',
                'latitude': lat or p.get('latitude', ''),
                'longitude': lon or p.get('longitude', ''),
                'council_district': p.get('council_district', ''),
                'nearby_GEOID': p.get('nearby_GEOID', ''),
                'zoning_case_GEOID': p.get('zoning_case_GEOID', ''),
            }

    log.info(f"Total properties after merge: {len(properties):,}")
    log.info(f"GeoJSON protest properties matched in LUI: {geojson_props_found:,}")
    log.info(f"Properties with protest history: {len(protest_history):,}")
    ever_protested = sum(1 for pid in properties if pid in protest_history)
    log.info(f"Properties ever protested: {ever_protested:,} ({100*ever_protested/len(properties):.2f}%)")

    # Write property universe
    out_path = os.path.join(PANEL_DIR, "property_universe.csv")
    fieldnames = list(next(iter(properties.values())).keys())
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for prop in sorted(properties.values(), key=lambda x: x['standardized_tcad_id']):
            writer.writerow(prop)

    log.info(f"Wrote {len(properties):,} properties to {out_path}")
    return properties, protest_history


# ---------------------------------------------------------------------------
# Step 2: Create Property x Year Skeleton
# ---------------------------------------------------------------------------
def step2_property_year_skeleton(properties, protest_history):
    """Create balanced panel with protest outcome — streaming write for large universe."""
    log.info("=" * 60)
    log.info("STEP 2: Creating Property x Year Skeleton")
    log.info("=" * 60)

    years = list(range(YEAR_MIN, YEAR_MAX + 1))
    log.info(f"Year range: {YEAR_MIN}-{YEAR_MAX} ({len(years)} years)")
    log.info(f"Expected rows: {len(properties):,} x {len(years)} = {len(properties) * len(years):,}")

    out_path = os.path.join(PANEL_DIR, "property_year_skeleton.csv")
    fieldnames = ['standardized_tcad_id', 'year', 'protest']

    total_protests = 0
    total_rows = 0

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tcad_id in sorted(properties.keys()):
            protest_years = protest_history.get(tcad_id, set())
            for year in years:
                protest = 1 if year in protest_years else 0
                writer.writerow({
                    'standardized_tcad_id': tcad_id,
                    'year': year,
                    'protest': protest,
                })
                total_protests += protest
                total_rows += 1

    protest_rate = 100 * total_protests / total_rows if total_rows else 0
    log.info(f"Wrote {total_rows:,} rows to {out_path}")
    log.info(f"Total protest=1: {total_protests:,} ({protest_rate:.3f}%)")
    log.info(f"Total protest=0: {total_rows - total_protests:,}")

    # Per-year summary
    year_counts = defaultdict(int)
    for tcad_id, years_set in protest_history.items():
        for y in years_set:
            year_counts[y] += 1
    for y in sorted(year_counts.keys()):
        log.info(f"  Year {y}: {year_counts[y]} protests")

    return out_path


# ---------------------------------------------------------------------------
# Step 3: Parse & Clean EARS Data
# ---------------------------------------------------------------------------
def step3_parse_ears():
    """Parse EARS CSV files for available years, output clean per-year files."""
    log.info("=" * 60)
    log.info("STEP 3: Parsing EARS Appraisal Rolls")
    log.info("=" * 60)

    # Load column layout
    layout = []
    with open(LAYOUT_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layout.append(row)

    col_names = [row['field_name'] for row in layout]
    log.info(f"EARS layout: {len(col_names)} columns")

    # Identify SAFE/CAUTION columns to keep
    keep_cols = set()
    exclude_cols = set()
    for row in layout:
        if row['leakage_risk'] == 'EXCLUDE':
            exclude_cols.add(row['field_name'])
        else:
            keep_cols.add(row['field_name'])

    log.info(f"Keeping {len(keep_cols)} columns, excluding {len(exclude_cols)}")
    log.info(f"Excluded columns: {exclude_cols}")

    ears_data = {}  # year -> list of dicts

    for year in EARS_YEARS:
        csv_path = os.path.join(EARS_DIR, str(year), f"EARS_{year}_Master.csv")
        if not os.path.exists(csv_path):
            log.warning(f"EARS {year} Master CSV not found at {csv_path}, skipping")
            continue

        log.info(f"Parsing EARS {year}: {csv_path}")
        records = []
        seen_accounts = set()
        total_rows = 0
        ajr_rows = 0
        dupes = 0

        with open(csv_path, 'r', encoding='latin-1') as f:
            reader = csv.reader(f)
            for row_data in reader:
                total_rows += 1
                # Pad or trim to expected column count
                if len(row_data) < len(col_names):
                    row_data.extend([''] * (len(col_names) - len(row_data)))
                elif len(row_data) > len(col_names):
                    row_data = row_data[:len(col_names)]

                row = dict(zip(col_names, row_data))

                # Filter to AJR records only
                if row.get('record_type') != 'AJR':
                    continue
                ajr_rows += 1

                acct = row.get('account_number', '').strip()
                if not acct:
                    continue

                # 2022+ uses 10-digit account_number = parcel_id_10;
                # 2019-2021 uses 6-digit account_number = property_id.
                # Store a canonical parcel_id_10 when available.
                if len(acct) == 10:
                    # 10-digit IS the parcel_id_10 (direct LUI join)
                    parcel_id_10 = acct
                    property_id = row.get('account_number_formatted', '').strip()
                else:
                    # 6-digit is property_id (needs crosswalk to parcel_id_10)
                    parcel_id_10 = ''  # filled later via crosswalk
                    property_id = acct

                # Deduplicate: keep first occurrence per (parcel_id or property_id, jurisdiction)
                dedup_key = parcel_id_10 or property_id
                if dedup_key in seen_accounts:
                    dupes += 1
                    continue
                seen_accounts.add(dedup_key)

                # Keep only non-excluded columns
                clean_row = {k: v for k, v in row.items() if k in keep_cols}
                clean_row['ears_year'] = year
                clean_row['_parcel_id_10'] = parcel_id_10
                clean_row['_property_id'] = property_id
                records.append(clean_row)

        log.info(f"  Total rows: {total_rows:,}")
        log.info(f"  AJR rows: {ajr_rows:,}")
        log.info(f"  Unique accounts: {len(records):,}")
        log.info(f"  Duplicates skipped: {dupes:,}")

        # Write clean file
        out_path = os.path.join(PANEL_DIR, f"ears_{year}_clean.csv")
        if records:
            fieldnames = list(records[0].keys())
            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
            log.info(f"  Wrote {len(records):,} records to {out_path}")

        ears_data[year] = records

    return ears_data


# ---------------------------------------------------------------------------
# Step 4: Build ID Crosswalk
# ---------------------------------------------------------------------------
def step4_id_crosswalk(ears_data):
    """Build crosswalk between parcel_id_10 (GeoJSON) and EARS account_number."""
    log.info("=" * 60)
    log.info("STEP 4: Building ID Crosswalk")
    log.info("=" * 60)

    # Load Land Use Inventory
    log.info(f"Loading Land Use Inventory: {LUI_PATH}")
    lui_map = {}  # parcel_id_10 -> property_id
    lui_reverse = {}  # property_id -> parcel_id_10
    with open(LUI_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid10 = row.get('parcel_id_10', '').strip()
            prop_id = row.get('property_id', '').strip()
            if pid10 and prop_id:
                lui_map[pid10] = prop_id
                lui_reverse[prop_id] = pid10
    log.info(f"Land Use Inventory: {len(lui_map):,} parcel_id_10 → property_id mappings")

    # Collect all EARS account numbers
    ears_accounts = set()
    for year, records in ears_data.items():
        for rec in records:
            acct = rec.get('account_number', '').strip()
            if acct:
                ears_accounts.add(acct)
    log.info(f"Total unique EARS account numbers: {len(ears_accounts):,}")

    # Test: does property_id match EARS account_number?
    property_ids = set(lui_map.values())
    overlap = property_ids.intersection(ears_accounts)
    log.info(f"LUI property_id ∩ EARS account_number overlap: {len(overlap):,}")

    if len(overlap) > 100:
        log.info("✓ property_id matches EARS account_number — using as crosswalk")
        # Build crosswalk: parcel_id_10 -> EARS account_number via property_id
        crosswalk = {}
        for pid10, prop_id in lui_map.items():
            if prop_id in ears_accounts:
                crosswalk[pid10] = prop_id
        log.info(f"Crosswalk entries (parcel_id_10 → account_number): {len(crosswalk):,}")
    else:
        log.warning("✗ property_id does NOT match EARS account_number")
        log.warning("Falling back to address-based matching (not yet implemented)")
        crosswalk = {}

    # Write crosswalk
    out_path = os.path.join(PANEL_DIR, "id_crosswalk.csv")
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['parcel_id_10', 'ears_account_number', 'source'])
        for pid10, acct in sorted(crosswalk.items()):
            writer.writerow([pid10, acct, 'LUI_property_id'])
    log.info(f"Wrote {len(crosswalk):,} crosswalk entries to {out_path}")

    return crosswalk


# ---------------------------------------------------------------------------
# Step 5: Merge EARS onto Panel
# ---------------------------------------------------------------------------
def step5_merge_ears(skeleton_path, ears_data, crosswalk):
    """Merge EARS covariates onto the property-year skeleton."""
    log.info("=" * 60)
    log.info("STEP 5: Merging EARS onto Panel")
    log.info("=" * 60)

    # Build EARS lookup with dual-key strategy:
    # - 2022+: (parcel_id_10, year) -> record  [direct 10-digit join]
    # - 2019-2021: (property_id, year) -> record  [6-digit, needs crosswalk]
    ears_by_pid10 = {}   # (parcel_id_10, year) -> record
    ears_by_propid = {}  # (property_id, year) -> record
    for year, records in ears_data.items():
        for rec in records:
            pid10 = rec.get('_parcel_id_10', '')
            propid = rec.get('_property_id', '')
            if pid10:
                ears_by_pid10[(pid10, year)] = rec
            if propid:
                ears_by_propid[(propid, year)] = rec
    log.info(f"EARS lookup: {len(ears_by_pid10):,} by parcel_id_10, {len(ears_by_propid):,} by property_id")

    # Determine EARS columns to add (from first available record)
    ears_cols = []
    for year, records in ears_data.items():
        if records:
            ears_cols = [k for k in records[0].keys()
                        if k not in ('account_number', 'ears_year', 'record_type',
                                     'sequence_number', 'county_district_code', 'year',
                                     '_parcel_id_10', '_property_id')]
            break
    log.info(f"EARS columns to merge: {len(ears_cols)}")

    # Read skeleton and merge
    out_path = os.path.join(PANEL_DIR, "panel_with_ears.csv")
    matched = 0
    unmatched = 0
    total = 0

    with open(skeleton_path, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        skeleton_fields = reader.fieldnames

        out_fields = skeleton_fields + ears_cols + ['ears_matched', 'ears_source']

        with open(out_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields)
            writer.writeheader()

            for row in reader:
                total += 1
                tcad_id = row['standardized_tcad_id']
                year = int(row['year'])

                # Dual EARS lookup:
                # 1. Try year-matched EARS (direct pid10 join, then crosswalk)
                # 2. For pre-2019 years, backfill with EARS 2019 baseline
                ears_rec = None
                ears_source = ''
                lookup_year = year if year in EARS_YEARS else (2019 if year < 2019 else None)

                if lookup_year:
                    # Try direct parcel_id_10 join (works for 2022+ data)
                    ears_rec = ears_by_pid10.get((tcad_id, lookup_year))
                    if not ears_rec:
                        # Fall back to crosswalk property_id join
                        ears_acct = crosswalk.get(tcad_id)
                        if ears_acct:
                            ears_rec = ears_by_propid.get((ears_acct, lookup_year))
                    if ears_rec:
                        ears_source = str(lookup_year) if lookup_year == year else f'{lookup_year}_backfill'

                out_row = dict(row)
                if ears_rec:
                    for col in ears_cols:
                        out_row[col] = ears_rec.get(col, '')
                    out_row['ears_matched'] = 1
                    out_row['ears_source'] = ears_source
                    matched += 1
                else:
                    for col in ears_cols:
                        out_row[col] = ''
                    out_row['ears_matched'] = 0
                    out_row['ears_source'] = ''
                    unmatched += 1

                writer.writerow(out_row)

    # All rows are EARS-eligible (year-matched or 2019 backfill)
    ears_eligible = total

    log.info(f"Total panel rows: {total:,}")
    log.info(f"EARS-eligible rows (years {EARS_YEARS}): {ears_eligible:,}")
    log.info(f"EARS matched: {matched:,} ({100*matched/max(ears_eligible,1):.1f}% of eligible)")
    log.info(f"EARS unmatched: {unmatched:,}")

    return out_path


# ---------------------------------------------------------------------------
# Step 6: Merge Land Use
# ---------------------------------------------------------------------------
def step6_merge_land_use(panel_path):
    """Merge Land Use Inventory onto panel (direct parcel_id_10 join)."""
    log.info("=" * 60)
    log.info("STEP 6: Merging Land Use Inventory")
    log.info("=" * 60)

    # Load Land Use Inventory
    lui_data = {}  # parcel_id_10 -> {land_use, general_land_use, shape_area}
    with open(LUI_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid10 = row.get('parcel_id_10', '').strip()
            if pid10:
                lui_data[pid10] = {
                    'lui_land_use': row.get('land_use', ''),
                    'lui_general_land_use': row.get('general_land_use', ''),
                    'lui_shape_area': row.get('shape_area', ''),
                }
    log.info(f"Land Use Inventory: {len(lui_data):,} parcels loaded")

    lui_cols = ['lui_land_use', 'lui_general_land_use', 'lui_shape_area']

    out_path = os.path.join(PANEL_DIR, "panel_with_landuse.csv")
    matched = 0
    total = 0

    with open(panel_path, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        out_fields = reader.fieldnames + lui_cols + ['lui_matched']

        with open(out_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields)
            writer.writeheader()

            for row in reader:
                total += 1
                tcad_id = row['standardized_tcad_id']

                lui_rec = lui_data.get(tcad_id)
                out_row = dict(row)
                if lui_rec:
                    out_row.update(lui_rec)
                    out_row['lui_matched'] = 1
                    matched += 1
                else:
                    for col in lui_cols:
                        out_row[col] = ''
                    out_row['lui_matched'] = 0

                writer.writerow(out_row)

    log.info(f"Total rows: {total:,}")
    log.info(f"Land Use matched: {matched:,} ({100*matched/max(total,1):.1f}%)")
    log.info(f"Land Use unmatched: {total - matched:,}")

    return out_path


# ---------------------------------------------------------------------------
# Step 7: Add Census Variables (from GeoJSON)
# ---------------------------------------------------------------------------
def step7_add_census(panel_path, properties):
    """Add census variables from the property universe (extracted from GeoJSON)."""
    log.info("=" * 60)
    log.info("STEP 7: Adding Census Variables from GeoJSON")
    log.info("=" * 60)

    census_cols = [
        'council_district',
        'nearby_GEOID', 'zoning_case_GEOID',
        'zoning_case_total_population', 'zoning_case_median_age',
        'zoning_case_race_white', 'zoning_case_race_black',
        'zoning_case_race_asian', 'zoning_case_race_hispanic',
        'zoning_case_median_income', 'zoning_case_poverty_count',
        'zoning_case_median_home_value',
        'zoning_case_owner_occupied', 'zoning_case_renter_occupied',
        'zoning_case_commute_time',
        'latitude', 'longitude',
    ]

    out_path = os.path.join(PANEL_DIR, "Property_Year_Panel.csv")
    added = 0
    total = 0

    with open(panel_path, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        # Only add columns not already present
        new_cols = [c for c in census_cols if c not in reader.fieldnames]
        out_fields = reader.fieldnames + new_cols

        with open(out_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields)
            writer.writeheader()

            for row in reader:
                total += 1
                tcad_id = row['standardized_tcad_id']
                prop = properties.get(tcad_id, {})

                out_row = dict(row)
                for col in new_cols:
                    out_row[col] = prop.get(col, '')

                if any(out_row.get(c) for c in new_cols):
                    added += 1

                writer.writerow(out_row)

    log.info(f"Total rows: {total:,}")
    log.info(f"Rows with census data: {added:,}")
    log.info(f"Final panel written to: {out_path}")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start = datetime.now()
    log.info(f"Panel construction started at {start.isoformat()}")
    log.info(f"Project dir: {PROJECT_DIR}")
    log.info(f"Data dir: {DATA_DIR}")
    log.info(f"Panel dir: {PANEL_DIR}")

    # Step 1
    properties, protest_history = step1_property_universe()

    # Step 2
    skeleton_path = step2_property_year_skeleton(properties, protest_history)

    # Step 3
    ears_data = step3_parse_ears()

    # Step 4
    crosswalk = step4_id_crosswalk(ears_data)

    # Step 5
    panel_path = step5_merge_ears(skeleton_path, ears_data, crosswalk)

    # Step 6: Merge property-level attributes (from LUI + GeoJSON, embedded in Step 1)
    log.info("=" * 60)
    log.info("STEP 6: Merging Property Attributes onto Panel")
    log.info("=" * 60)

    prop_cols = ['lui_land_use', 'lui_general_land_use', 'lui_shape_area',
                 'latitude', 'longitude', 'council_district',
                 'nearby_GEOID', 'zoning_case_GEOID']

    final_path = os.path.join(PANEL_DIR, "Property_Year_Panel.csv")
    total = 0
    with open(panel_path, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        out_fields = reader.fieldnames + prop_cols

        with open(final_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields)
            writer.writeheader()

            for row in reader:
                total += 1
                tcad_id = row['standardized_tcad_id']
                prop = properties.get(tcad_id, {})
                out_row = dict(row)
                for col in prop_cols:
                    out_row[col] = prop.get(col, '')
                writer.writerow(out_row)

    log.info(f"Merged property attributes for {total:,} rows")

    elapsed = datetime.now() - start
    log.info("=" * 60)
    log.info(f"DONE. Elapsed: {elapsed}")
    log.info(f"Final output: {final_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
