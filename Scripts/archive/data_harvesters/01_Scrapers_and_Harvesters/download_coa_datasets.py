"""
download_coa_datasets.py — Download City of Austin Open Data Portal datasets
=============================================================================
Downloads all CoA datasets referenced in the panel construction pipeline.
Each file is named with the dataset ID and labeled year for provenance traceability.

Naming convention: {Category}_{LabeledYear}_{DatasetID}.csv

Datasets:
  Land Use / Land Database:
    - LUI_2024_7vsm-dvxg.csv  — Land Use Inventory Detailed (current/2024, 284K rows)
    - LUI_2022_6qkk-xgys.csv  — Land Use Inventory Detailed (2022 snapshot, 281K rows)
    - LDB_2021_kk8y-6cmt.csv  — Land Database 2021 (272K rows)
    - LDB_2016_4nsn-uea6.csv  — Land Database Data Only 2016 (265K rows)
    - LUI_2012_3k7r-w54d.csv  — Land Use Inventory (2012 original, 285K rows, already local)

  Zoning:
    - ZC_current_edir-dcnf.csv  — Zoning Cases (living dataset)
    - ZBA_current_nbzi-qabm.csv — Zoning By Address (living dataset)

  Boundaries:
    - JURISDICTIONS_current_3pzb-6mbr.csv — City jurisdiction boundaries

  Permits:
    - PERMITS_current_3syk-w9eu.csv — Building Permits (living dataset, 2.3M rows)
      Columns: permit_number, permittype, permit_type_desc, permit_class,
               permit_class_mapped, work_class, tcad_id, legal_description,
               applieddate, issue_date, status_current, expiresdate,
               original_address1, council_district, jurisdiction,
               latitude, longitude

Author: Daniel Hardesty Lewis
Created: 2026-02-16
Updated: 2026-03-09 — added Building Permits
"""

import os
import sys
import shutil
import requests
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "Data")
COA_DIR = os.path.join(DATA_DIR, "CoA_Open_Data")
PERMITS_DIR = os.path.join(COA_DIR, "Permits")

# Socrata SODA2 CSV export endpoint pattern
# https://data.austintexas.gov/api/views/{DATASET_ID}/rows.csv?accessType=DOWNLOAD
SODA2_BASE = "https://data.austintexas.gov/api/views/{dataset_id}/rows.csv?accessType=DOWNLOAD"

# SODA JSON endpoint for filtered/selected downloads (used for large datasets)
SODA_JSON_BASE = "https://data.austintexas.gov/resource/{dataset_id}.csv"

DATASETS = [
    # (filename, dataset_id, description, labeled_year, expected_rows)
    ("LUI_2024_7vsm-dvxg.csv", "7vsm-dvxg",
     "Land Use Inventory Detailed (current/2024)", "2024", 284958),
    ("LUI_2022_6qkk-xgys.csv", "6qkk-xgys",
     "Land Use Inventory Detailed (2022 snapshot)", "2022", 280889),
    ("LDB_2021_kk8y-6cmt.csv", "kk8y-6cmt",
     "Land Database 2021", "2021", 271568),
    ("LDB_2016_4nsn-uea6.csv", "4nsn-uea6",
     "Land Database Data Only 2016", "2016", 265422),
    ("ZC_current_edir-dcnf.csv", "edir-dcnf",
     "Zoning Cases", "current", None),
    ("ZBA_current_nbzi-qabm.csv", "nbzi-qabm",
     "Zoning By Address", "current", None),
    ("JURISDICTIONS_current_3pzb-6mbr.csv", "3pzb-6mbr",
     "City jurisdiction boundaries", "current", None),
]

# Building Permits: too large for bulk CSV; use paginated SODA JSON endpoint
# Filter to Building Permits with a TCAD ID (panel-joinable rows only)
# Drop contractor PII and Socrata computed region columns
PERMITS_DATASET = {
    "filename": "PERMITS_current_3syk-w9eu.csv",
    "dataset_id": "3syk-w9eu",
    "description": "Building Permits (all types, panel-joinable rows only)",
    "year": "current",
    # Only rows with a TCAD ID (joinable to panel)
    "where": "tcad_id IS NOT NULL",
    # Drop contractor PII, description, computed regions
    "select": (
        "permit_number,permittype,permit_type_desc,permit_class,"
        "permit_class_mapped,work_class,tcad_id,legal_description,"
        "applieddate,issue_date,day_issued,calendar_year_issued,"
        "fiscal_year_issued,status_current,statusdate,expiresdate,"
        "original_address1,original_city,original_state,original_zip,"
        "council_district,jurisdiction,latitude,longitude"
    ),
    "page_size": 50000,
    "expected_rows": None,
}

# Also copy existing local land use file with provenance name
EXISTING_LUI = {
    "source": os.path.join(DATA_DIR, "Zoning_Cases", "Source_Data", "land_use_inventory_prefetched.csv"),
    "dest_name": "LUI_2012_3k7r-w54d.csv",
    "description": "Land Use Inventory (2012 original, already local)",
}


def download_dataset(filename, dataset_id, description, expected_rows=None):
    """Download a single dataset from the CoA Open Data Portal."""
    dest = os.path.join(COA_DIR, filename)

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  SKIP {filename} — already exists ({size_mb:.1f} MB)")
        return True

    url = SODA2_BASE.format(dataset_id=dataset_id)
    print(f"  Downloading {filename} ({description})")
    print(f"    URL: {url}")

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_bytes = 0
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                total_bytes += len(chunk)

        size_mb = total_bytes / (1024 * 1024)
        print(f"    Saved: {size_mb:.1f} MB")

        # Verify row count if expected
        if expected_rows:
            with open(dest, 'r', encoding='utf-8', errors='replace') as f:
                actual = sum(1 for _ in f) - 1  # subtract header
            print(f"    Rows: {actual:,} (expected ~{expected_rows:,})")
            if actual < expected_rows * 0.9:
                print(f"    WARNING: Row count significantly below expected!")

        return True

    except requests.RequestException as e:
        print(f"    FAILED: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def download_permits_paginated(cfg):
    """Download Building Permits via paginated SODA JSON/CSV endpoint."""
    os.makedirs(PERMITS_DIR, exist_ok=True)
    dest = os.path.join(PERMITS_DIR, cfg["filename"])

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  SKIP {cfg['filename']} — already exists ({size_mb:.1f} MB)")
        return True

    base = SODA_JSON_BASE.format(dataset_id=cfg["dataset_id"])
    page_size = cfg["page_size"]
    total_written = 0
    offset = 0
    header_written = False

    print(f"  Downloading {cfg['filename']} ({cfg['description']}) — paginated")

    try:
        with open(dest, 'wb') as f:
            while True:
                params = {
                    "$select": cfg["select"],
                    "$where": cfg["where"],
                    "$limit": page_size,
                    "$offset": offset,
                    "$order": ":id",
                }
                r = requests.get(base, params=params, timeout=120)
                r.raise_for_status()

                lines = r.content.splitlines(keepends=True)
                if not lines or (len(lines) == 1 and offset > 0):
                    break  # only header returned — done

                if not header_written:
                    f.writelines(lines)
                    header_written = True
                    rows_this_page = len(lines) - 1
                else:
                    f.writelines(lines[1:])  # skip repeated header
                    rows_this_page = len(lines) - 1

                total_written += rows_this_page
                print(f"    offset={offset:,}  rows_this_page={rows_this_page:,}  total={total_written:,}")

                if rows_this_page < page_size:
                    break  # last page
                offset += page_size
                time.sleep(0.25)  # be polite

        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"    Done: {total_written:,} rows saved ({size_mb:.1f} MB)")
        return True

    except requests.RequestException as e:
        print(f"    FAILED: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False

def copy_existing():
    """Copy existing local land use file with provenance naming."""
    src = EXISTING_LUI["source"]
    dest = os.path.join(COA_DIR, EXISTING_LUI["dest_name"])

    if os.path.exists(dest):
        print(f"  SKIP {EXISTING_LUI['dest_name']} — already exists")
        return

    if os.path.exists(src):
        shutil.copy2(src, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  Copied {EXISTING_LUI['dest_name']} ({size_mb:.1f} MB)")
    else:
        print(f"  WARNING: Source not found: {src}")


def write_manifest():
    """Write a manifest file documenting all downloaded datasets."""
    manifest_path = os.path.join(COA_DIR, "MANIFEST.md")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write("# City of Austin Open Data — Downloaded Datasets\n\n")
        f.write(f"Downloaded: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| File | Dataset ID | Description | Year | Portal URL |\n")
        f.write("|------|-----------|-------------|------|------------|\n")

        for filename, dataset_id, desc, year, _ in DATASETS:
            url = f"https://data.austintexas.gov/d/{dataset_id}"
            f.write(f"| `{filename}` | `{dataset_id}` | {desc} | {year} | [{dataset_id}]({url}) |\n")

        # Add the existing local file
        f.write(f"| `{EXISTING_LUI['dest_name']}` | `3k7r-w54d` | {EXISTING_LUI['description']} | 2012 | [3k7r-w54d](https://data.austintexas.gov/d/3k7r-w54d) |\n")

        # Add Building Permits
        p = PERMITS_DATASET
        url = f"https://data.austintexas.gov/d/{p['dataset_id']}"
        permits_dest = os.path.join(PERMITS_DIR, p['filename'])
        note = ""
        if os.path.exists(permits_dest):
            size_mb = os.path.getsize(permits_dest) / (1024 * 1024)
            note = f" ({size_mb:.0f} MB, tcad_id-filtered)"
        f.write(f"| `Permits/{p['filename']}` | `{p['dataset_id']}` | {p['description']}{note} | {p['year']} | [{p['dataset_id']}]({url}) |\n")

        f.write("\n## Naming Convention\n\n")
        f.write("Files are named `{Category}_{LabeledYear}_{DatasetID}.csv`:\n")
        f.write("- **LUI** = Land Use Inventory\n")
        f.write("- **LDB** = Land Database (richer: includes zoning, appraisal, structure data)\n")
        f.write("- **ZC** = Zoning Cases\n")
        f.write("- **ZBA** = Zoning By Address\n")
        f.write("- **JURISDICTIONS** = City boundary polygons\n")
        f.write("- **PERMITS** = Building Permits (tcad_id-filtered, contractor PII stripped)\n")

    print(f"  Wrote manifest to {manifest_path}")


def main():
    print("=" * 60)
    print("City of Austin Open Data — Dataset Download")
    print("=" * 60)

    os.makedirs(COA_DIR, exist_ok=True)
    os.makedirs(PERMITS_DIR, exist_ok=True)
    print(f"Output directory: {COA_DIR}\n")

    # Copy existing local file
    print("Copying existing local data:")
    copy_existing()

    # Download all datasets
    print("\nDownloading datasets from CoA Open Data Portal:")
    successes = 0
    failures = 0
    for filename, dataset_id, desc, year, expected in DATASETS:
        ok = download_dataset(filename, dataset_id, desc, expected)
        if ok:
            successes += 1
        else:
            failures += 1

    # Download Building Permits (paginated)
    print("\nDownloading Building Permits (paginated, tcad_id-filtered):")
    ok = download_permits_paginated(PERMITS_DATASET)
    if ok:
        successes += 1
    else:
        failures += 1

    # Write manifest
    print("\nWriting manifest:")
    write_manifest()

    print(f"\nDone: {successes} succeeded, {failures} failed")


if __name__ == "__main__":
    main()
