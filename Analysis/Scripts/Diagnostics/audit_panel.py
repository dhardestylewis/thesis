"""
audit_panel.py — Comprehensive Panel Audit
============================================
1. Column-by-column coverage (% non-empty) across all property-years
2. Property universe comparison: panel vs full TCAD/LDB/LUI
3. Temporal leakage re-audit for every variable in the enriched panel

Author: Daniel Hardesty Lewis
Created: 2026-02-16
"""

import csv
import os
import sys
from collections import defaultdict, Counter

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "Data")
PANEL_DIR = os.path.join(DATA_DIR, "Panel")
COA_DIR = os.path.join(DATA_DIR, "CoA_Open_Data")
ENRICHED = os.path.join(PANEL_DIR, "Property_Year_Panel_Enriched.csv")

# Temporal leakage classifications for ALL variables
# Each var: (risk, justification)
LEAKAGE_AUDIT = {
    # --- Panel identifiers ---
    'standardized_tcad_id': ('N/A', 'Property identifier'),
    'year': ('N/A', 'Panel year dimension'),
    'protest': ('N/A', 'OUTCOME variable — not a predictor'),

    # --- EARS: SAFE (assessed Jan 1, before protest period) ---
    'land_market_value': ('SAFE', 'Appraised Jan 1, before protest window'),
    'improvement_market_value': ('SAFE', 'Appraised Jan 1'),
    'total_market_value': ('SAFE', 'Appraised Jan 1'),
    'appraised_value': ('SAFE', 'Appraised Jan 1'),
    'assessed_value': ('SAFE', 'Appraised Jan 1'),
    'taxable_value': ('SAFE', 'Taxable value set Jan 1'),
    'prior_year_taxable_value': ('SAFE', 'Prior year, strictly pre-treatment'),
    'property_category_code': ('SAFE', 'Physical attribute, time-invariant or slow-changing'),
    'subcategory_code': ('SAFE', 'Physical attribute'),
    'year_built': ('SAFE', 'Historical, cannot change post-protest'),
    'deed_acreage': ('SAFE', 'Physical attribute'),
    'land_acres': ('SAFE', 'Physical attribute'),
    'homesite_flag': ('SAFE', 'Set in prior assessment'),
    'improvement_sq_ft': ('SAFE', 'Physical attribute'),
    'exemption_flag_hs': ('SAFE', 'Exemption filed before Jan 1'),
    'exemption_flag_ov65': ('SAFE', 'Exemption filed before Jan 1'),
    'exemption_flag_dp': ('SAFE', 'Exemption filed before Jan 1'),
    'exemption_flag_dv': ('SAFE', 'Exemption filed before Jan 1'),
    'exemption_amount_hs': ('SAFE', 'Exemption amount set Jan 1'),
    'exemption_amount_ov65': ('SAFE', 'Exemption amount set Jan 1'),
    'total_exemption_amount': ('SAFE', 'Total exemptions set Jan 1'),
    'freeze_flag': ('SAFE', 'Tax ceiling set prior year'),
    'ears_matched': ('N/A', 'Merge indicator'),
    'ears_year': ('N/A', 'Merge metadata'),

    # --- EARS: CAUTION ---
    'owner_name': ('CAUTION', 'Ownership may change post-protest; use for owner-type classification only'),
    'most_recent_sale_date': ('CAUTION', 'Sale may post-date protest; use lagged t-1 only'),
    'new_construction_value': ('CAUTION', 'Construction may be related to zoning; use lagged'),
    'zoning_code': ('CAUTION', 'Zoning changes are the SUBJECT of protests; use lagged t-1 only'),

    # --- EARS: EXCLUDE (should NOT be in panel) ---
    'arb_protest_flag': ('EXCLUDE', 'Directly encodes whether appraisal was protested'),
    'arb_protest_result': ('EXCLUDE', 'Encodes protest hearing outcome'),

    # --- GeoJSON metadata ---
    'site_address': ('N/A', 'Property identifier/metadata'),
    'latitude': ('N/A', 'Spatial coordinate'),
    'longitude': ('N/A', 'Spatial coordinate'),
    'council_district': ('SAFE', 'Administrative boundary, slow-changing'),
    'nearby_parcel_id_10': ('N/A', 'Join key'),
    'nearby_property_id': ('N/A', 'Join key'),
    'zoning_case_parcel_id_10': ('N/A', 'Join key'),
    'zoning_case_property_id': ('N/A', 'Join key'),
    'nearby_GEOID': ('N/A', 'Census join key'),
    'zoning_case_GEOID': ('N/A', 'Census join key'),

    # --- Land Use Inventory (static snapshot from build_panel) ---
    'lui_land_use': ('SAFE', 'Land use classification from static snapshot'),
    'lui_general_land_use': ('SAFE', 'General land use from static snapshot'),
    'lui_shape_area': ('SAFE', 'Physical parcel area, time-invariant'),
    'lui_matched': ('N/A', 'Merge indicator'),

    # --- Time-varying LUI (forward-filled) ---
    'lui_land_use_tv': ('SAFE', 'Land use from nearest prior snapshot; forward-filled with source_year tracking'),
    'lui_general_land_use_tv': ('SAFE', 'General land use from nearest prior snapshot'),
    'lui_source_year': ('N/A', 'Forward-fill provenance indicator'),

    # --- Time-varying Census/ACS ---
    'acs_total_population': ('SAFE', 'ACS released ~2yr after survey period; always pre-dates panel year'),
    'acs_median_age': ('SAFE', 'Tract-level demographic, ACS vintage <= panel year'),
    'acs_race_white': ('SAFE', 'Tract-level demographic'),
    'acs_race_black': ('SAFE', 'Tract-level demographic'),
    'acs_race_asian': ('SAFE', 'Tract-level demographic'),
    'acs_race_hispanic': ('SAFE', 'Tract-level demographic'),
    'acs_median_household_income': ('SAFE', 'Tract-level economic, ACS vintage <= panel year'),
    'acs_poverty_count': ('SAFE', 'Tract-level economic'),
    'acs_median_home_value': ('SAFE', 'Tract-level housing market'),
    'acs_owner_occupied_units': ('SAFE', 'Tract-level housing tenure'),
    'acs_renter_occupied_units': ('SAFE', 'Tract-level housing tenure'),
    'acs_median_gross_rent': ('SAFE', 'Tract-level housing cost'),
    'acs_total_housing_units': ('SAFE', 'Tract-level housing stock'),
    'acs_vintage': ('N/A', 'ACS vintage year for provenance'),

    # --- Land Database (forward-filled) ---
    'ldb_basezone': ('CAUTION', 'Zoning may change as RESULT of protest; forward-filled from prior snapshot mitigates but does not eliminate risk'),
    'ldb_eff_zone': ('CAUTION', 'Same risk as basezone'),
    'ldb_lotsize': ('SAFE', 'Physical attribute, slow-changing'),
    'ldb_council_district': ('SAFE', 'Administrative boundary'),
    'ldb_far': ('SAFE', 'Physical attribute derived from building/lot'),
    'ldb_ilr': ('SAFE', 'Improvement-to-land ratio, assessed pre-protest'),
    'ldb_units': ('CAUTION', 'Unit count could change with redevelopment tied to zoning; forward-fill mitigates'),
    'ldb_yr_built': ('SAFE', 'Historical, cannot change post-protest'),
    'ldb_land_acres': ('SAFE', 'Physical attribute'),
    'ldb_imprv_sqft': ('SAFE', 'Physical attribute'),
    'ldb_appraised_val': ('CAUTION', 'Appraised value from snapshot year, not panel year; forward-fill may mix eras'),
    'ldb_market_val': ('CAUTION', 'Same risk as appraised_val; snapshot value, not contemporaneous'),
    'ldb_land_use': ('SAFE', 'Land use code from prior snapshot'),
    'ldb_gen_land_use': ('SAFE', 'General land use category'),
    'ldb_lu_desc': ('SAFE', 'Land use description'),
    'ldb_gen_lu_desc': ('SAFE', 'General land use description'),
    'ldb_constrained_area': ('SAFE', 'Environmental/regulatory constraint, slow-changing'),
    'ldb_i35side': ('SAFE', 'Geographic, time-invariant'),
    'ldb_imprv_type_desc': ('SAFE', 'Building type, slow-changing'),
    'ldb_source_year': ('N/A', 'Forward-fill provenance indicator'),
}

def audit_coverage():
    """Audit every column for coverage across all property-years."""
    print("=" * 80)
    print("AUDIT 1: Variable Coverage in Enriched Panel")
    print("=" * 80)

    col_stats = defaultdict(lambda: {'total': 0, 'filled': 0, 'empty': 0})
    year_stats = defaultdict(lambda: defaultdict(int))  # col -> year -> filled_count
    total_rows = 0
    n_properties = set()
    n_years = set()

    with open(ENRICHED, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        print(f"\nTotal columns: {len(cols)}\n")

        for row in reader:
            total_rows += 1
            yr = int(row['year'])
            n_properties.add(row['standardized_tcad_id'])
            n_years.add(yr)

            for col in cols:
                val = row.get(col, '').strip()
                col_stats[col]['total'] += 1
                if val and val != '':
                    col_stats[col]['filled'] += 1
                    year_stats[col][yr] += 1
                else:
                    col_stats[col]['empty'] += 1

    print(f"Panel dimensions: {len(n_properties)} properties × {len(n_years)} years = {total_rows} rows")
    print(f"Year range: {min(n_years)}–{max(n_years)}\n")

    # Sort by coverage (worst first)
    print(f"{'Column':<40} {'Filled':>8} {'Empty':>8} {'Coverage':>10} {'Leakage':>10} {'Years w/ Data'}")
    print("-" * 110)

    for col in sorted(cols, key=lambda c: col_stats[c]['filled'] / max(col_stats[c]['total'], 1)):
        s = col_stats[col]
        pct = 100 * s['filled'] / max(s['total'], 1)
        leakage = LEAKAGE_AUDIT.get(col, ('UNKNOWN', ''))[0]
        years_with_data = sorted(year_stats[col].keys())
        if years_with_data:
            yr_range = f"{min(years_with_data)}-{max(years_with_data)} ({len(years_with_data)} yrs)"
        else:
            yr_range = "NONE"
        print(f"{col:<40} {s['filled']:>8,} {s['empty']:>8,} {pct:>9.1f}% {leakage:>10} {yr_range}")

    # Flag problems
    print("\n" + "=" * 80)
    print("FLAGGED ISSUES")
    print("=" * 80)

    for col in cols:
        s = col_stats[col]
        pct = 100 * s['filled'] / max(s['total'], 1)
        leakage = LEAKAGE_AUDIT.get(col, ('UNKNOWN', ''))[0]

        if leakage == 'EXCLUDE':
            print(f"[EXCLUDE]: '{col}' should NOT be in the panel! ({s['filled']:,} filled)")
        elif leakage == 'UNKNOWN':
            print(f"[UNKNOWN] leakage status: '{col}' ({pct:.1f}% coverage)")
        elif leakage == 'CAUTION' and pct > 0:
            justification = LEAKAGE_AUDIT[col][1]
            print(f"[CAUTION]: '{col}' ({pct:.1f}% coverage) -- {justification}")

    return col_stats


def audit_universe():
    """Compare panel property universe against full TCAD/LDB/LUI universes."""
    print("\n" + "=" * 80)
    print("AUDIT 2: Property Universe Coverage")
    print("=" * 80)

    # Panel properties
    panel_props = set()
    with open(os.path.join(PANEL_DIR, "property_universe.csv"), 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            panel_props.add(row['standardized_tcad_id'])
    print(f"\nPanel properties: {len(panel_props):,}")

    # LUI 2024 (most comprehensive)
    lui_path = os.path.join(COA_DIR, "LUI_2024_7vsm-dvxg.csv")
    if os.path.exists(lui_path):
        lui_props = set()
        with open(lui_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            pid_col = None
            for h in reader.fieldnames:
                if h.upper() in ('PARCEL_ID_10', 'PID_10'):
                    pid_col = h
                    break
            if pid_col:
                for row in reader:
                    pid = row.get(pid_col, '').strip()
                    if pid:
                        try:
                            pid = str(int(float(pid))).zfill(10)
                        except (ValueError, OverflowError):
                            pid = pid.zfill(10)
                        lui_props.add(pid)

        overlap = panel_props.intersection(lui_props)
        print(f"LUI 2024 parcels: {len(lui_props):,}")
        print(f"Panel ∩ LUI 2024: {len(overlap):,} ({100*len(overlap)/max(len(panel_props),1):.1f}% of panel)")
        print(f"LUI parcels NOT in panel: {len(lui_props) - len(overlap):,}")
        print(f"Panel parcels NOT in LUI: {len(panel_props) - len(overlap):,}")

    # LDB 2016
    ldb16_path = os.path.join(COA_DIR, "LDB_2016_4nsn-uea6.csv")
    if os.path.exists(ldb16_path):
        ldb_props = set()
        with open(ldb16_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get('PID_10', '').strip()
                if pid:
                    try:
                        pid = str(int(float(pid))).zfill(10)
                    except (ValueError, OverflowError):
                        pass
                    ldb_props.add(pid)
        overlap = panel_props.intersection(ldb_props)
        print(f"\nLDB 2016 parcels: {len(ldb_props):,}")
        print(f"Panel ∩ LDB 2016: {len(overlap):,} ({100*len(overlap)/max(len(panel_props),1):.1f}% of panel)")

    # LDB 2021
    ldb21_path = os.path.join(COA_DIR, "LDB_2021_kk8y-6cmt.csv")
    if os.path.exists(ldb21_path):
        ldb21_props = set()
        with open(ldb21_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            pid_col = None
            for h in reader.fieldnames:
                if h.upper() in ('PID_10',):
                    pid_col = h
                    break
            if pid_col:
                for row in reader:
                    pid = row.get(pid_col, '').strip()
                    if pid:
                        try:
                            pid = str(int(float(pid))).zfill(10)
                        except (ValueError, OverflowError):
                            pass
                        ldb21_props.add(pid)
            overlap = panel_props.intersection(ldb21_props)
            print(f"\nLDB 2021 parcels: {len(ldb21_props):,}")
            print(f"Panel ∩ LDB 2021: {len(overlap):,} ({100*len(overlap)/max(len(panel_props),1):.1f}% of panel)")

    # EARS crosswalk
    xwalk_path = os.path.join(PANEL_DIR, "id_crosswalk.csv")
    if os.path.exists(xwalk_path):
        xwalk_props = set()
        with open(xwalk_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                xwalk_props.add(row['parcel_id_10'])
        overlap = panel_props.intersection(xwalk_props)
        print(f"\nEARS crosswalk parcels: {len(xwalk_props):,}")
        print(f"Panel ∩ crosswalk: {len(overlap):,} ({100*len(overlap)/max(len(panel_props),1):.1f}% of panel)")

    # Key insight
    print("\n" + "-" * 80)
    print("NOTE: The panel currently includes ONLY properties that appear in the")
    print("protest petitions GeoJSON (2,902 parcels). The full Austin parcel universe")
    print("is ~250K-285K parcels. If the analysis requires a control group of")
    print("never-protested properties, the panel must be expanded.")
    print("-" * 80)


def audit_leakage():
    """Re-audit all variables for temporal leakage."""
    print("\n" + "=" * 80)
    print("AUDIT 3: Temporal Leakage Classification")
    print("=" * 80)

    with open(ENRICHED, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames

    categories = defaultdict(list)
    unknown = []

    for col in cols:
        if col in LEAKAGE_AUDIT:
            risk, justification = LEAKAGE_AUDIT[col]
            categories[risk].append((col, justification))
        else:
            unknown.append(col)

    for risk in ['EXCLUDE', 'CAUTION', 'SAFE', 'N/A']:
        items = categories.get(risk, [])
        if items:
            print(f"\n{risk} ({len(items)} variables):")
            for col, just in items:
                print(f"  {col:<40} {just}")

    if unknown:
        print(f"\n🚨 UNKNOWN ({len(unknown)} variables — NEED CLASSIFICATION):")
        for col in unknown:
            print(f"  {col}")


if __name__ == "__main__":
    audit_coverage()
    audit_universe()
    audit_leakage()
