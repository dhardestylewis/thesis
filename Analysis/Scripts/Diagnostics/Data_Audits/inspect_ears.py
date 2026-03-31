"""
Inspect EARS data across years and generate column layout.
Outputs: Data/Panel/Reference/EARS_Column_Layout.csv

References:
  - TX Comptroller EARS Record Layout and Instructions Manual (June 2019)
  - TCAD Electronic Records Submission Manual
  - Data inspection of EARS_YYYY_Master.csv / EARS_YYYY_Jurisdiction_Tax_Values.{csv,txt}
"""

import csv
import os
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Data")

# ============================================================
# EARS AJR Column Layout (2019+ CSV format, 84 columns)
# Source: TX Comptroller EARS Record Layout and Instructions Manual June 2019
#         Cross-referenced with TCAD data inspection
#
# Format: (position, field_id, field_name, data_type, description,
#          temporal_leakage_risk, leakage_justification)
# ============================================================

EARS_AJR_COLUMNS = [
    # Col | Field ID  | Field Name                        | Type    | Description
    (0,  "AJR001", "record_type",                    "text",    "Record type identifier (always 'AJR')",
         "N/A",     "Metadata field, not a variable"),
    (1,  "AJR002", "tax_year",                       "numeric", "SDPVS tax year for which data is submitted",
         "SAFE",    "Panel identifier, not a predictor"),
    (2,  "AJR003", "appraisal_district_id",          "text",    "Appraisal district identification number (e.g., 227 = Travis)",
         "SAFE",    "Static administrative identifier"),
    (3,  "AJR004", "county_id",                      "text",    "County identification number (FIPS-like)",
         "SAFE",    "Static geographic identifier"),
    (4,  "AJR005", "taxing_unit_id",                 "text",    "Taxing unit identification number",
         "SAFE",    "Static administrative identifier"),
    (5,  "AJR006", "taxing_unit_name",               "text",    "Name of taxing unit",
         "SAFE",    "Static administrative name"),
    (6,  "AJR007", "account_number",                 "text",    "Property account number in the CAD system",
         "SAFE",    "Property identifier, links to TCAD ID"),
    (7,  "AJR008", "account_number_formatted",       "text",    "Formatted version of account number",
         "SAFE",    "Property identifier"),
    (8,  "AJR009", "property_type_code",             "text",    "Property type classification code",
         "SAFE",    "Determined by property characteristics, not protests"),
    (9,  "AJR010", "situs_street_number",            "text",    "Property street number (situs address)",
         "SAFE",    "Static location identifier"),
    (10, "AJR011", "situs_street_prefix",            "text",    "Street direction prefix (N/S/E/W)",
         "SAFE",    "Static location identifier"),
    (11, "AJR012", "situs_street_name",              "text",    "Property street name",
         "SAFE",    "Static location identifier"),
    (12, "AJR013", "situs_street_suffix",            "text",    "Street type suffix (St/Ave/Blvd)",
         "SAFE",    "Static location identifier"),
    (13, "AJR014", "situs_city_state_zip",           "text",    "City, state, and ZIP code of property",
         "SAFE",    "Static location, may change very rarely"),
    (14, "AJR015", "special_use_flag",               "text",    "Flag for special use designation (Y/N)",
         "SAFE",    "Determined by land use, not protests"),
    (15, "AJR016", "legal_description",              "text",    "Legal description of the property",
         "SAFE",    "Recorded at platting, rarely changes"),
    (16, "AJR017", "owner_name",                     "text",    "Property owner name",
         "CAUTION", "May change due to sales; post-protest sales could leak"),
    (17, "AJR018", "owner_id",                       "text",    "Owner identification code",
         "CAUTION", "Changes with ownership transfers"),
    (18, "AJR019", "owner_address_line1",            "text",    "Owner mailing address line 1",
         "CAUTION", "Changes with ownership transfers"),
    (19, "AJR020", "owner_address_line2",            "text",    "Owner mailing address line 2",
         "CAUTION", "Changes with ownership transfers"),
    (20, "AJR021", "owner_city",                     "text",    "Owner mailing city",
         "CAUTION", "Owner demographics, may change"),
    (21, "AJR022", "owner_state",                    "text",    "Owner mailing state",
         "CAUTION", "Owner demographics, investor vs local"),
    (22, "AJR023", "year_built",                     "numeric", "Year the improvement was built",
         "SAFE",    "Physical characteristic, pre-determined"),
    (23, "AJR024", "homesite_flag",                  "text",    "Homesite designation flag (Y/N)",
         "SAFE",    "Reflects homestead status at assessment time"),
    (24, "AJR025", "improvement_sq_ft",              "text",    "Improvement square footage or descriptor",
         "SAFE",    "Physical characteristic, rarely changes"),
    (25, "AJR026", "deed_acreage",                   "numeric", "Deeded acreage of the property",
         "SAFE",    "Physical characteristic, fixed at platting"),
    (26, "AJR027", "most_recent_sale_date",          "text",    "Date of most recent property sale (YYYY or MMDDYYYY)",
         "CAUTION", "Sales may be triggered by protests or vice versa; use lagged"),
    (27, "AJR028", "second_most_recent_sale_date",   "text",    "Date of second most recent sale",
         "CAUTION", "Historical transaction, generally safe if pre-protest"),
    (28, "AJR029", "land_acres",                     "numeric", "Acreage for land valuation",
         "SAFE",    "Physical characteristic"),
    (29, "AJR030", "land_market_value",              "numeric", "Market value of land only",
         "SAFE",    "Assessed Jan 1 of tax year, before protest decisions"),
    (30, "AJR031", "property_category_code",         "text",    "Category code (A=SFR, B=MFR, C=Vacant, D1=Ag, E=Industrial, F=Commercial, G=Oil/Gas, J=Utility, L=Personal Prop, M=Mobile Home, O=Other, S=Special)",
         "SAFE",    "Reflects property use, rarely changes"),
    (31, "AJR032", "subcategory_code",               "text",    "Detailed subcategory code",
         "SAFE",    "Reflects property characteristics"),
    (32, "AJR033", "improvement_market_value",       "numeric", "Market value of improvements",
         "SAFE",    "Assessed Jan 1 of tax year, before protests"),
    (33, "AJR034", "total_market_value",             "numeric", "Total market value (land + improvements)",
         "SAFE",    "Assessed Jan 1 of tax year, before protest filing"),
    (34, "AJR035", "appraised_value",                "numeric", "Appraised value after cap/limitation adjustments",
         "SAFE",    "Set before protest period"),
    (35, "AJR036", "assessed_value",                 "numeric", "Assessed value for tax base",
         "SAFE",    "Set before protest period"),
    (36, "AJR037", "exemption_amount_hs",            "numeric", "Homestead exemption amount",
         "SAFE",    "Exemption applied before protests"),
    (37, "AJR038", "exemption_amount_ov65",          "numeric", "Over-65 exemption amount",
         "SAFE",    "Age-based exemption, predetermined"),
    (38, "AJR039", "exemption_flag_hs",              "text",    "Homestead exemption flag (Y/N)",
         "SAFE",    "Indicates homestead, pre-determined"),
    (39, "AJR040", "exemption_flag_ov65",            "text",    "Over-65 exemption flag (Y/N)",
         "SAFE",    "Age-based, pre-determined"),
    (40, "AJR041", "exemption_flag_dp",              "text",    "Disabled person exemption flag (Y/N)",
         "SAFE",    "Disability-based, pre-determined"),
    (41, "AJR042", "exemption_flag_dv",              "text",    "Disabled veteran exemption flag (Y/N)",
         "SAFE",    "Veteran status, pre-determined"),
    (42, "AJR043", "exemption_amount_dv",            "numeric", "Disabled veteran exemption amount",
         "SAFE",    "Pre-determined exemption"),
    (43, "AJR044", "exemption_amount_dp",            "numeric", "Disabled person exemption amount",
         "SAFE",    "Pre-determined exemption"),
    (44, "AJR045", "exemption_amount_ex366",         "numeric", "Section 11.13(f) exemption amount",
         "SAFE",    "Statutory exemption, pre-determined"),
    (45, "AJR046", "taxable_value",                  "numeric", "Taxable value after all exemptions",
         "SAFE",    "Derived from assessed value minus exemptions, before protests"),
    (46, "AJR047", "productivity_value",             "numeric", "Agricultural productivity value (1-d-1 or 1-d)",
         "SAFE",    "Ag valuation, pre-determined"),
    (47, "AJR048", "ag_market_value",                "numeric", "Ag land market value if not under ag use",
         "SAFE",    "Valuation, pre-determined"),
    (48, "AJR049", "exemption_amount_pc",            "numeric", "Pollution control exemption",
         "SAFE",    "Industrial exemption, pre-determined"),
    (49, "AJR050", "exemption_amount_fr",            "numeric", "Freeport exemption amount",
         "SAFE",    "Business inventory exemption"),
    (50, "AJR051", "exemption_amount_ht",            "numeric", "Historic/cultural exemption amount",
         "SAFE",    "Historic designation, pre-determined"),
    (51, "AJR052", "exemption_amount_solar_wind",    "numeric", "Solar/wind energy exemption",
         "SAFE",    "Energy incentive exemption"),
    (52, "AJR053", "exemption_amount_ch313",         "numeric", "Chapter 313 (TEDA) agreement exemption",
         "SAFE",    "Tax abatement, pre-determined"),
    (53, "AJR054", "exemption_amount_other1",        "numeric", "Other exemption amount 1",
         "SAFE",    "Miscellaneous exemption"),
    (54, "AJR055", "exemption_amount_other2",        "numeric", "Other exemption amount 2",
         "SAFE",    "Miscellaneous exemption"),
    (55, "AJR056", "exemption_amount_other3",        "numeric", "Other exemption amount 3",
         "SAFE",    "Miscellaneous exemption"),
    (56, "AJR057", "exemption_amount_other4",        "numeric", "Other exemption amount 4",
         "SAFE",    "Miscellaneous exemption"),
    (57, "AJR058", "exemption_amount_other5",        "numeric", "Other exemption amount 5",
         "SAFE",    "Miscellaneous exemption"),
    (58, "AJR059", "exemption_amount_other6",        "numeric", "Other exemption amount 6",
         "SAFE",    "Miscellaneous exemption"),
    (59, "AJR060", "exemption_amount_other7",        "numeric", "Other exemption amount 7",
         "SAFE",    "Miscellaneous exemption"),
    (60, "AJR061", "exemption_amount_other8",        "numeric", "Other exemption amount 8",
         "SAFE",    "Miscellaneous exemption"),
    (61, "AJR062", "exemption_amount_other9",        "numeric", "Other exemption amount 9",
         "SAFE",    "Miscellaneous exemption"),
    (62, "AJR063", "exemption_amount_other10",       "numeric", "Other exemption amount 10",
         "SAFE",    "Miscellaneous exemption"),
    (63, "AJR064", "exemption_amount_other11",       "numeric", "Other exemption amount 11",
         "SAFE",    "Miscellaneous exemption"),
    (64, "AJR065", "exemption_amount_other12",       "numeric", "Other exemption amount 12",
         "SAFE",    "Miscellaneous exemption"),
    (65, "AJR066", "total_exemption_amount",         "numeric", "Total of all exemption amounts",
         "SAFE",    "Sum of exemptions, pre-determined"),
    (66, "AJR067", "net_taxable_value",              "numeric", "Net taxable value (appraised - exemptions)",
         "SAFE",    "Derived value, before protests"),
    (67, "AJR068", "freeze_flag",                    "text",    "Tax ceiling/freeze flag (Y/N)",
         "SAFE",    "Over-65/disabled freeze, pre-determined"),
    (68, "AJR069", "freeze_flag_2",                  "text",    "Second freeze flag",
         "SAFE",    "Pre-determined"),
    (69, "AJR070", "ch313_flag",                     "text",    "Chapter 313 agreement flag (Y/N) (new in 2019)",
         "SAFE",    "Tax abatement flag, pre-determined"),
    (70, "AJR071", "abatement_flag",                 "text",    "Tax abatement agreement flag",
         "SAFE",    "Pre-determined designation"),
    (71, "AJR072", "partial_exemption_flag",         "text",    "Partial exemption flag (Y/N)",
         "SAFE",    "Pre-determined exemption status"),
    (72, "AJR073", "partial_exemption_flag_2",       "text",    "Second partial exemption flag",
         "SAFE",    "Pre-determined"),
    (73, "AJR074", "land_use_code",                  "text",    "Land use code",
         "SAFE",    "Current land use, rarely changes organically"),
    (74, "AJR075", "zoning_code",                    "text",    "Zoning designation code",
         "CAUTION", "Zoning changes may be related to protests; use lagged"),
    (75, "AJR076", "school_district_flag",           "text",    "School district association flag (Y/N)",
         "SAFE",    "Geographic, pre-determined"),
    (76, "AJR077", "ceiling_value",                  "numeric", "Tax ceiling value (for frozen accounts)",
         "SAFE",    "Pre-determined for eligible accounts"),
    (77, "AJR078", "arb_protest_flag",               "text",    "ARB (Appraisal Review Board) protest flag (Y/N)",
         "EXCLUDE", "DIRECT OUTCOME LEAKAGE: indicates whether property protested its appraisal"),
    (78, "AJR079", "arb_protest_result",             "text",    "ARB protest hearing result/status",
         "EXCLUDE", "DIRECT OUTCOME LEAKAGE: encodes protest outcome"),
    (79, "AJR080", "appraisal_district_id_2",        "text",    "Secondary appraisal district identifier",
         "SAFE",    "Administrative identifier"),
    (80, "AJR081", "prior_year_taxable_value",       "numeric", "Prior year taxable value",
         "SAFE",    "Lagged value, safe as prior-year covariate"),
    (81, "AJR082", "new_construction_value",         "numeric", "Value of new construction / improvements added",
         "CAUTION", "New construction may trigger or be triggered by zoning changes"),
    (82, "AJR083", "supplemental_record_count",      "numeric", "Number of supplemental records for account",
         "SAFE",    "Administrative count"),
    (83, "AJR084", "record_sequence_number",         "numeric", "Sequence number within file",
         "N/A",     "Metadata, not a variable"),
]


def write_column_layout():
    """Write EARS column layout to CSV for the codebook."""
    output_path = os.path.join(DATA_DIR, "Panel", "EARS_Column_Layout.csv")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "position", "field_id", "field_name", "data_type",
            "description", "leakage_risk", "leakage_justification"
        ])
        for col in EARS_AJR_COLUMNS:
            writer.writerow(col)
    print(f"Wrote {len(EARS_AJR_COLUMNS)} columns to {output_path}")
    return output_path


def inspect_ears_data():
    """Inspect EARS data files for each year and validate column counts."""
    print("\n=== EARS Data Inspection ===")
    for year in range(2018, 2026):
        year_dir = os.path.join(DATA_DIR, "Appraisal_Rolls", str(year))
        if not os.path.exists(year_dir):
            print(f"\n{year}: Directory not found")
            continue

        # Find the main data file
        candidates = [
            f"EARS_{year}_Master.csv",
            f"EARS_{year}_Jurisdiction_Tax_Values.csv",
            f"EARS_{year}_Master.txt",
            f"EARS_{year}_Jurisdiction_Tax_Values.txt",
        ]
        main_file = None
        for cand in candidates:
            path = os.path.join(year_dir, cand)
            if os.path.exists(path):
                main_file = path
                break

        if main_file is None:
            print(f"\n{year}: No main data file found")
            continue

        # Inspect
        is_csv = main_file.endswith('.csv')
        basename = os.path.basename(main_file)
        size_mb = os.path.getsize(main_file) / (1024*1024)

        with open(main_file, 'r', encoding='latin-1') as f:
            first_line = f.readline()

        if is_csv:
            cols = len(next(csv.reader([first_line])))
            rec_type = next(csv.reader([first_line]))[0]
        else:
            # Fixed-width: first 3 chars are record type
            rec_type = first_line[:3]
            cols = "fixed-width"

        print(f"\n{year}: {basename} ({size_mb:.0f} MB)")
        print(f"  Format: {'CSV' if is_csv else 'Fixed-width'}, Columns: {cols}, Record Type: {rec_type}")

        # Count records
        with open(main_file, 'r', encoding='latin-1') as f:
            line_count = sum(1 for _ in f)
        print(f"  Records: {line_count:,}")


def inspect_geojson_summary():
    """Quick summary of GeoJSON protest petitions data."""
    print("\n=== Protest Petitions GeoJSON Inspection ===")
    geojson_path = os.path.join(DATA_DIR, "Protest_Petitions", "GeoJSON", "protest_petitions_v1.geojson")
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    feats = data['features']
    print(f"Total features: {len(feats)}")

    # Unique TCAD IDs
    tcad_ids = set()
    years = []
    for feat in feats:
        props = feat['properties']
        tcad = props.get('standardized_tcad_id') or props.get('TCAD ID', '')
        if tcad:
            tcad_ids.add(tcad)
        fd = props.get('final_date', '')
        if fd and len(fd) >= 4:
            years.append(fd[:4])

    from collections import Counter
    yr_counts = Counter(years)

    print(f"Unique TCAD IDs: {len(tcad_ids)}")
    print(f"Year distribution (final_date):")
    for y in sorted(yr_counts.keys()):
        print(f"  {y}: {yr_counts[y]}")


if __name__ == "__main__":
    # Step 1: Write column layout
    layout_path = write_column_layout()

    # Step 2: Inspect EARS data across years
    inspect_ears_data()

    # Step 3: Inspect GeoJSON
    inspect_geojson_summary()

    print("\n\nDone! Column layout written to:", layout_path)
