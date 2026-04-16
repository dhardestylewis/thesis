"""
Rebuild Protest Panel — Combined PDF Signers + GeoJSON
======================================================
Uses two data sources to identify protest parcels:
1. PDF petition signers (petition_signers_from_pdf.csv) — ground truth signed parcels
2. GeoJSON features with ALL nearby parcels (those within 200' buffer)

Creates Property_Year_Panel_v2.csv with:
- protest_zoning: 1 if parcel signed a petition (from PDF or GeoJSON Signature='yes')
- protest_nearby: 1 if parcel is in a petition area but didn't sign (GeoJSON + PDF non-signers)
- protest_case_numbers: case numbers linked to this parcel
"""
import csv, json, sys, os, re, time
from collections import defaultdict

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

PDF_SIGNERS_PATH = "Data/Protest_Petitions/petition_signers_from_pdf.csv"
GEOJSON_PATH = "Data/Protest_Petitions/GeoJSON/protest_petitions_v1.geojson"
PANEL_PATH = "Data/Panel/Output/Property_Year_Panel.csv"
OUT_PATH = "Data/Panel/Output/Property_Year_Panel_v2.csv"


def normalize_tcad(tid):
    if not tid:
        return ""
    return tid.replace("-", "").replace(" ", "").lstrip("0")


# ---- Step 1: Load PDF signers ----
print("Loading PDF signers...")
pdf_signed = defaultdict(set)      # (norm_tcad, year) -> set of case numbers
pdf_nearby = defaultdict(set)      # (norm_tcad, year) -> set of case numbers (non-signers)
pdf_stats = {"signed": 0, "not_signed": 0, "no_year": 0}

with open(PDF_SIGNERS_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        tcad_norm = row.get("tcad_normalized", "")
        year_str = row.get("year", "")
        signed = row.get("signed", "0")
        cn = row.get("case_number", "")

        if not tcad_norm or not year_str:
            pdf_stats["no_year"] += 1
            continue

        try:
            year = int(year_str)
        except ValueError:
            pdf_stats["no_year"] += 1
            continue

        if signed == "1":
            pdf_signed[(tcad_norm, year)].add(cn)
            pdf_stats["signed"] += 1
        else:
            pdf_nearby[(tcad_norm, year)].add(cn)
            pdf_stats["not_signed"] += 1

print("  PDF signed: %d unique (tcad, year) pairs" % len(pdf_signed))
print("  PDF nearby (not signed): %d" % len(pdf_nearby))
print("  Stats: %s" % pdf_stats)

# ---- Step 2: Load GeoJSON signers ----
print("\nLoading GeoJSON...")
with open(GEOJSON_PATH, "r") as f:
    gj = json.load(f)

gj_signed = defaultdict(set)
gj_nearby = defaultdict(set)
gj_stats = {"signed": 0, "nearby": 0, "no_tcad": 0, "no_year": 0}

for feat in gj["features"]:
    props = feat["properties"]
    tcad = (props.get("TCAD ID") or "").strip()
    cn = (props.get("Case Number") or "").strip()
    sig = (props.get("Signature") or "").lower()

    if not tcad:
        gj_stats["no_tcad"] += 1
        continue

    # Get year
    year = None
    m = re.search(r"((?:19|20)\d\d)", cn)
    if m:
        year = int(m.group(1))
    if not year:
        d = props.get("application_start_date") or ""
        m2 = re.search(r"((?:19|20)\d\d)", d)
        if m2:
            year = int(m2.group(1))
    if not year:
        gj_stats["no_year"] += 1
        continue

    tcad_norm = normalize_tcad(tcad)

    if sig == "yes":
        gj_signed[(tcad_norm, year)].add(cn)
        gj_stats["signed"] += 1
    else:
        gj_nearby[(tcad_norm, year)].add(cn)
        gj_stats["nearby"] += 1

print("  GeoJSON signed: %d unique (tcad, year) pairs" % len(gj_signed))
print("  GeoJSON nearby: %d" % len(gj_nearby))
print("  Stats: %s" % gj_stats)

# ---- Step 3: Merge both sources ----
print("\nMerging PDF + GeoJSON signers...")
all_signed = defaultdict(set)
all_nearby = defaultdict(set)

# PDF signers take priority
for key, cases in pdf_signed.items():
    all_signed[key].update(cases)
for key, cases in gj_signed.items():
    all_signed[key].update(cases)

# Nearby parcels (non-signers from both sources)
for key, cases in pdf_nearby.items():
    if key not in all_signed:  # Don't mark as nearby if they're a signer
        all_nearby[key].update(cases)
for key, cases in gj_nearby.items():
    if key not in all_signed:
        all_nearby[key].update(cases)

print("  Combined signed: %d unique (tcad, year) pairs" % len(all_signed))
print("  Combined nearby: %d" % len(all_nearby))

# Per-year summary
year_signed = defaultdict(int)
year_nearby = defaultdict(int)
for (_, yr) in all_signed:
    year_signed[yr] += 1
for (_, yr) in all_nearby:
    year_nearby[yr] += 1

print("\n=== Combined protest data per year ===")
print("Year | Signers | Nearby | Total")
print("-----|---------|--------|------")
for yr in sorted(set(year_signed.keys()) | set(year_nearby.keys())):
    print("%d | %7d | %6d | %5d" % (yr, year_signed[yr], year_nearby[yr],
                                      year_signed[yr] + year_nearby[yr]))

# ---- Step 4: Discover panel years for closest-year mapping ----
print("\nDiscovering panel years...")
panel_years = set()
with open(PANEL_PATH, "r", encoding="utf-8") as f:
    for i, row in enumerate(csv.DictReader(f)):
        panel_years.add(int(row.get("year", 0)))
        if i > 500_000:
            break
panel_years = sorted(panel_years)
print("  Panel years: %s" % panel_years)


def closest_panel_year(case_year):
    if case_year in panel_years:
        return case_year
    return min(panel_years, key=lambda y: abs(y - case_year))


# Remap to panel years
remapped_signed = defaultdict(set)
remapped_nearby = defaultdict(set)
for (tcad, yr), cases in all_signed.items():
    mapped = closest_panel_year(yr)
    remapped_signed[(tcad, mapped)].update(cases)
for (tcad, yr), cases in all_nearby.items():
    mapped = closest_panel_year(yr)
    if (tcad, mapped) not in remapped_signed:
        remapped_nearby[(tcad, mapped)].update(cases)

print("  After year mapping: %d signed, %d nearby" % (len(remapped_signed), len(remapped_nearby)))

# ---- Step 5: Build panel TCAD lookup ----
print("\nBuilding panel TCAD index...")
panel_norm_to_id = {}
with open(PANEL_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pid = row.get("standardized_tcad_id", "")
        if pid:
            norm = normalize_tcad(pid)
            panel_norm_to_id[norm] = pid
print("  Panel TCADs: %d" % len(panel_norm_to_id))

# Check match rates
signed_matched = sum(1 for (t, y) in remapped_signed if t in panel_norm_to_id)
nearby_matched = sum(1 for (t, y) in remapped_nearby if t in panel_norm_to_id)
print("  Signed matching panel: %d/%d" % (signed_matched, len(remapped_signed)))
print("  Nearby matching panel: %d/%d" % (nearby_matched, len(remapped_nearby)))

# Build final lookups with panel IDs
protest_signed = {}  # (panel_pid, year) -> case numbers
protest_nearby = {}
for (norm, yr), cases in remapped_signed.items():
    if norm in panel_norm_to_id:
        protest_signed[(panel_norm_to_id[norm], yr)] = cases
for (norm, yr), cases in remapped_nearby.items():
    if norm in panel_norm_to_id:
        protest_nearby[(panel_norm_to_id[norm], yr)] = cases

print("  Final signed parcel-years: %d" % len(protest_signed))
print("  Final nearby parcel-years: %d" % len(protest_nearby))

# ---- Step 6: Write v2 panel ----
print("\nRebuilding panel CSV...")
t0 = time.time()

with open(PANEL_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames)

for col in ["protest_zoning", "protest_nearby", "protest_case_numbers"]:
    if col not in fieldnames:
        fieldnames.append(col)

n_written = 0
counts = {"old_protest": 0, "new_signed": 0, "new_nearby": 0}

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

        if row.get("protest", "0") == "1":
            counts["old_protest"] += 1

        key = (pid, yr)

        if key in protest_signed:
            row["protest_zoning"] = "1"
            row["protest_nearby"] = "0"
            row["protest_case_numbers"] = "|".join(sorted(protest_signed[key]))
            counts["new_signed"] += 1
        elif key in protest_nearby:
            row["protest_zoning"] = "0"
            row["protest_nearby"] = "1"
            row["protest_case_numbers"] = "|".join(sorted(protest_nearby[key]))
            counts["new_nearby"] += 1
        else:
            row["protest_zoning"] = "0"
            row["protest_nearby"] = "0"
            row["protest_case_numbers"] = ""

        writer.writerow(row)
        n_written += 1

elapsed = time.time() - t0
print("  Written %d rows in %.1fs" % (n_written, elapsed))
print("  Counts: %s" % counts)
print("\nOutput: %s (%.1f MB)" % (OUT_PATH, os.path.getsize(OUT_PATH) / 1e6))
