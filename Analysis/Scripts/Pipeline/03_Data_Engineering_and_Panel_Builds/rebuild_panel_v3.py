"""
Rebuild Panel v3 — Complete Zoning Case + 200ft Buffer + Petition Signer Merge
===============================================================================
Integrates three data sources into the panel:

1. ALL zoning cases (ZC CSV) — marks parcels that had zoning cases filed on them
2. Pre-computed 200ft buffer (combined GeoJSON) — marks parcels within 200ft
   of any zoning case, using polygon-overlap (not centroids)
3. PDF petition signers — ground truth protest signatures

New columns:
  - zoning_case_on_parcel:  1 if a zoning case was filed on this parcel
  - zoning_case_nearby:     1 if any zoning case within 200ft (polygon overlap)
  - protest_signed:         1 if parcel owner signed a protest petition
  - protest_nearby_area_pct: parcel area as % of total nearby area in 200ft buffer
  - zoning_case_numbers:    pipe-delimited case numbers linked to this parcel

Outputs: Data/Panel/Output/Property_Year_Panel_Enriched.csv
"""
import csv, json, sys, os, re, time
from collections import defaultdict

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# ========== Paths ==========
ZC_PATH = "Data/CoA_Open_Data/Zoning/ZC_current_edir-dcnf.csv"
COMBINED_GJ_PATH = "Data/Zoning_Cases/Processed_Data/combined_cases_with_nearby.geojson"
PDF_SIGNERS_PATH = "Data/Protest_Petitions/petition_signers_from_pdf.csv"
PANEL_PATH = "Data/Panel/Output/Property_Year_Panel.csv"
OUT_PATH = "Data/Panel/Output/Property_Year_Panel_Enriched.csv"


def normalize_tcad(tid):
    """Normalize TCAD IDs: remove dashes, spaces, leading zeros."""
    if not tid:
        return ""
    return tid.replace("-", "").replace(" ", "").lstrip("0")


def extract_year_from_case(case_number):
    """Extract year from case number like C14-2007-0144 or NPA-2024-0130."""
    if not case_number:
        return None
    m = re.search(r"((?:19|20)\d\d)", case_number)
    return int(m.group(1)) if m else None


# ========== Step 1: Load ALL zoning cases ==========
print("=" * 60)
print("Step 1: Loading all zoning cases from ZC CSV...")
print("=" * 60)

# Maps (normalized_tcad, year) -> set of case numbers
zc_on_parcel = defaultdict(set)
zc_stats = {"total": 0, "with_tcad": 0, "with_year": 0, "mapped": 0}

with open(ZC_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        zc_stats["total"] += 1
        tcad_raw = (row.get("TCAD_ID") or "").strip()
        if not tcad_raw:
            continue
        zc_stats["with_tcad"] += 1

        # Get year: prefer CALENDAR_YEAR_FOLDER_CREATED, fallback to case number
        year = None
        yr_str = (row.get("CALENDAR_YEAR_FOLDER_CREATED") or "").strip()
        if yr_str:
            try:
                year = int(yr_str)
            except ValueError:
                pass
        if not year:
            start = (row.get("APPLICATION_START_DATE") or "").strip()
            m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", start)
            if m:
                year = int(m.group(3))
        if not year:
            cn = (row.get("CASE_NUMBER") or "").strip()
            year = extract_year_from_case(cn)
        if not year:
            continue
        zc_stats["with_year"] += 1

        # A parcel can have multiple TCADs (semicolon-separated)
        for tid in tcad_raw.split(";"):
            tid = tid.strip()
            if tid:
                norm = normalize_tcad(tid)
                cn = (row.get("CASE_NUMBER") or "").strip()
                zc_on_parcel[(norm, year)].add(cn)
                zc_stats["mapped"] += 1

print(f"  Total ZC rows: {zc_stats['total']}")
print(f"  With TCAD: {zc_stats['with_tcad']}")
print(f"  With year: {zc_stats['with_year']}")
print(f"  Mapped parcel-year pairs: {len(zc_on_parcel)}")

# ========== Step 2: Load 200ft buffer from combined GeoJSON ==========
print()
print("=" * 60)
print("Step 2: Loading pre-computed 200ft buffer (combined GeoJSON)...")
print("=" * 60)

with open(COMBINED_GJ_PATH, "r") as f:
    gj = json.load(f)

# Maps (normalized_tcad, year) -> {case_numbers, area}
buffer_nearby = defaultdict(lambda: {"cases": set(), "area": 0.0})
buffer_stats = {"total": 0, "with_tcad": 0, "with_year": 0, "mapped": 0}

for feat in gj["features"]:
    buffer_stats["total"] += 1
    props = feat["properties"]

    # Get TCAD ID (try both field naming conventions)
    tcad = (props.get("TCAD ID") or props.get("parcel_id_10") or "").strip()
    if not tcad:
        continue
    buffer_stats["with_tcad"] += 1

    # Get case number
    cn = (props.get("Case Number") or props.get("case_number") or "").strip()

    # Get year from case number or date fields
    year = extract_year_from_case(cn)
    if not year:
        start = props.get("application_start_date") or ""
        m = re.search(r"((?:19|20)\d\d)", start)
        if m:
            year = int(m.group(1))
    if not year:
        continue
    buffer_stats["with_year"] += 1

    norm = normalize_tcad(tcad)
    sig = (props.get("Signature") or "").lower().strip()

    # Get area for percentage calculation
    area = 0.0
    area_val = props.get("nearby_shape_area") or props.get("Shape_Area")
    if area_val:
        try:
            area = float(area_val)
        except (ValueError, TypeError):
            pass

    key = (norm, year)
    buffer_nearby[key]["cases"].add(cn)
    buffer_nearby[key]["area"] = max(buffer_nearby[key]["area"], area)
    buffer_stats["mapped"] += 1

print(f"  Total GeoJSON features: {buffer_stats['total']}")
print(f"  With TCAD: {buffer_stats['with_tcad']}")
print(f"  With year: {buffer_stats['with_year']}")
print(f"  Mapped nearby parcel-years: {len(buffer_nearby)}")

# Calculate area percentage per case
# Group by case+year to get total area, then compute each parcel's share
case_total_area = defaultdict(float)
case_parcels = defaultdict(list)  # case_year -> [(norm_tcad, area)]

for (norm, yr), info in buffer_nearby.items():
    for cn in info["cases"]:
        case_key = (cn, yr)
        case_total_area[case_key] += info["area"]
        case_parcels[case_key].append((norm, info["area"]))

# Build area_pct lookup
area_pct = {}  # (norm_tcad, year) -> max pct across all cases
for case_key, parcels in case_parcels.items():
    total = case_total_area[case_key]
    if total <= 0:
        continue
    for (norm, area) in parcels:
        pct = (area / total) * 100.0
        key = (norm, case_key[1])  # (norm_tcad, year)
        area_pct[key] = max(area_pct.get(key, 0.0), pct)

print(f"  Parcels with area_pct: {len(area_pct)}")

# ========== Step 3: Load PDF petition signers ==========
print()
print("=" * 60)
print("Step 3: Loading PDF petition signers...")
print("=" * 60)

pdf_signed = defaultdict(set)  # (norm_tcad, year) -> case numbers
pdf_stats = {"signed": 0, "not_signed": 0, "no_data": 0}

with open(PDF_SIGNERS_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        tcad_norm = row.get("tcad_normalized", "")
        year_str = row.get("year", "")
        signed = row.get("signed", "0")
        cn = row.get("case_number", "")

        if not tcad_norm or not year_str:
            pdf_stats["no_data"] += 1
            continue

        try:
            year = int(year_str)
        except ValueError:
            pdf_stats["no_data"] += 1
            continue

        if signed == "1":
            pdf_signed[(tcad_norm, year)].add(cn)
            pdf_stats["signed"] += 1
        else:
            pdf_stats["not_signed"] += 1

print(f"  PDF signed parcels: {len(pdf_signed)} unique (tcad, year)")
print(f"  Stats: {pdf_stats}")

# Also check GeoJSON for Signature=yes (from the same combined file)
gj_signed = defaultdict(set)
for feat in gj["features"]:
    props = feat["properties"]
    sig = (props.get("Signature") or "").lower().strip()
    if sig != "yes":
        continue
    tcad = (props.get("TCAD ID") or props.get("parcel_id_10") or "").strip()
    cn = (props.get("Case Number") or props.get("case_number") or "").strip()
    if not tcad:
        continue
    year = extract_year_from_case(cn)
    if not year:
        continue
    norm = normalize_tcad(tcad)
    gj_signed[(norm, year)].add(cn)

# Merge PDF + GeoJSON signers
all_signed = defaultdict(set)
for key, cases in pdf_signed.items():
    all_signed[key].update(cases)
for key, cases in gj_signed.items():
    all_signed[key].update(cases)

print(f"  GeoJSON signed: {len(gj_signed)} unique (tcad, year)")
print(f"  Combined signed: {len(all_signed)} unique (tcad, year)")

# ========== Step 4: Discover panel years and build TCAD index ==========
print()
print("=" * 60)
print("Step 4: Building panel TCAD index...")
print("=" * 60)

panel_norm_to_id = {}
panel_years = set()

with open(PANEL_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pid = row.get("standardized_tcad_id", "")
        if pid:
            norm = normalize_tcad(pid)
            panel_norm_to_id[norm] = pid
        yr = row.get("year", "")
        if yr:
            try:
                panel_years.add(int(yr))
            except ValueError:
                pass

panel_years = sorted(panel_years)
print(f"  Panel TCADs: {len(panel_norm_to_id)}")
print(f"  Panel years: {panel_years}")


def closest_panel_year(case_year):
    if case_year in panel_years:
        return case_year
    return min(panel_years, key=lambda y: abs(y - case_year))


# ========== Step 5: Build final lookups with panel IDs ==========
print()
print("=" * 60)
print("Step 5: Mapping to panel IDs...")
print("=" * 60)

# Remap all lookups to panel years and panel IDs
final_on_parcel = {}  # (panel_pid, year) -> case numbers
final_nearby = {}     # (panel_pid, year) -> case numbers
final_signed = {}     # (panel_pid, year) -> case numbers
final_area_pct = {}   # (panel_pid, year) -> pct

match_stats = {"zc_matched": 0, "buffer_matched": 0, "signed_matched": 0}

# ZC on-parcel
for (norm, yr), cases in zc_on_parcel.items():
    if norm in panel_norm_to_id:
        mapped_yr = closest_panel_year(yr)
        key = (panel_norm_to_id[norm], mapped_yr)
        if key not in final_on_parcel:
            final_on_parcel[key] = set()
        final_on_parcel[key].update(cases)
        match_stats["zc_matched"] += 1

# Buffer nearby
for (norm, yr), info in buffer_nearby.items():
    if norm in panel_norm_to_id:
        mapped_yr = closest_panel_year(yr)
        key = (panel_norm_to_id[norm], mapped_yr)
        if key not in final_nearby:
            final_nearby[key] = set()
        final_nearby[key].update(info["cases"])
        match_stats["buffer_matched"] += 1

# Signed
for (norm, yr), cases in all_signed.items():
    if norm in panel_norm_to_id:
        mapped_yr = closest_panel_year(yr)
        key = (panel_norm_to_id[norm], mapped_yr)
        if key not in final_signed:
            final_signed[key] = set()
        final_signed[key].update(cases)
        match_stats["signed_matched"] += 1

# Area pct
for (norm, yr), pct in area_pct.items():
    if norm in panel_norm_to_id:
        mapped_yr = closest_panel_year(yr)
        key = (panel_norm_to_id[norm], mapped_yr)
        final_area_pct[key] = max(final_area_pct.get(key, 0.0), pct)

print(f"  ZC on-parcel matched: {match_stats['zc_matched']} -> {len(final_on_parcel)} panel keys")
print(f"  Buffer nearby matched: {match_stats['buffer_matched']} -> {len(final_nearby)} panel keys")
print(f"  Signed matched: {match_stats['signed_matched']} -> {len(final_signed)} panel keys")
print(f"  Area pct entries: {len(final_area_pct)}")

# Per-year breakdown
print("\n=== Per-year breakdown ===")
print("Year |  ZC on | Nearby | Signed | Total Affected")
print("-----|--------|--------|--------|---------------")
yr_stats = defaultdict(lambda: [0, 0, 0])
for (pid, yr) in final_on_parcel:
    yr_stats[yr][0] += 1
for (pid, yr) in final_nearby:
    yr_stats[yr][1] += 1
for (pid, yr) in final_signed:
    yr_stats[yr][2] += 1
for yr in sorted(yr_stats):
    s = yr_stats[yr]
    print(f"{yr} | {s[0]:6d} | {s[1]:6d} | {s[2]:6d} | {s[0]+s[1]+s[2]:14d}")

# ========== Step 6: Write v3 panel ==========
print()
print("=" * 60)
print("Step 6: Writing Property_Year_Panel_Enriched.csv...")
print("=" * 60)
t0 = time.time()

with open(PANEL_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames)

# Remove old protest columns if present
old_cols = ["protest_zoning", "protest_nearby", "protest_case_numbers"]
fieldnames = [c for c in fieldnames if c not in old_cols]

# Add new columns
new_cols = [
    "zoning_case_on_parcel",
    "zoning_case_nearby",
    "protest_signed",
    "protest_nearby_area_pct",
    "zoning_case_numbers",
]
for col in new_cols:
    if col not in fieldnames:
        fieldnames.append(col)

n_written = 0
counts = {"on_parcel": 0, "nearby": 0, "signed": 0}

with (
    open(PANEL_PATH, "r", encoding="utf-8") as fin,
    open(OUT_PATH, "w", newline="", encoding="utf-8") as fout,
):
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        pid = row.get("standardized_tcad_id", "")
        try:
            yr = int(row.get("year", 0))
        except ValueError:
            yr = 0

        key = (pid, yr)

        # Zoning case on this parcel
        if key in final_on_parcel:
            row["zoning_case_on_parcel"] = "1"
            counts["on_parcel"] += 1
        else:
            row["zoning_case_on_parcel"] = "0"

        # Nearby (200ft buffer)
        if key in final_nearby:
            row["zoning_case_nearby"] = "1"
            counts["nearby"] += 1
        else:
            row["zoning_case_nearby"] = "0"

        # Signed petition
        if key in final_signed:
            row["protest_signed"] = "1"
            counts["signed"] += 1
        else:
            row["protest_signed"] = "0"

        # Area pct
        if key in final_area_pct:
            row["protest_nearby_area_pct"] = f"{final_area_pct[key]:.2f}"
        else:
            row["protest_nearby_area_pct"] = ""

        # Build combined case numbers
        all_cases = set()
        if key in final_on_parcel:
            all_cases.update(final_on_parcel[key])
        if key in final_nearby:
            all_cases.update(final_nearby[key])
        if key in final_signed:
            all_cases.update(final_signed[key])
        row["zoning_case_numbers"] = "|".join(sorted(all_cases)) if all_cases else ""

        # Remove old columns from row dict
        for old in old_cols:
            row.pop(old, None)

        writer.writerow(row)
        n_written += 1

elapsed = time.time() - t0
print(f"  Written {n_written} rows in {elapsed:.1f}s")
print(f"  Counts: {counts}")
print(f"\nOutput: {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)")

# Final summary
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  zoning_case_on_parcel = 1:  {counts['on_parcel']:,} rows")
print(f"  zoning_case_nearby = 1:     {counts['nearby']:,} rows")
print(f"  protest_signed = 1:         {counts['signed']:,} rows")
print(f"  Total panel rows:           {n_written:,}")
