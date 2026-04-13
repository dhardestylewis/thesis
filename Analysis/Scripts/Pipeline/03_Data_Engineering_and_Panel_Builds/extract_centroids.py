"""
Extract parcel centroids from LUI 2024 geometry and merge into panel.
====================================================================
Uses the LUI 2024 CSV which contains MULTIPOLYGON WKT in `the_geom`.
Computes centroids for all ~285K parcels and writes a lookup CSV.

Justification: parcels are stable across vintages (~284K in 2012 & 2024),
so we use 2024 as the single reference geometry for all panel years.
"""
import csv, sys, time, re, os, shutil

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

LUI_PATH = "Data/CoA_Open_Data/Land_Use/LUI_2024_7vsm-dvxg.csv"
OUT_CENTROIDS = "Data/Panel/Reference/parcel_centroids.csv"
PANEL_IN = "Data/Panel/Output/Property_Year_Panel_Enriched.csv"
PANEL_OUT = "Data/Panel/Output/Property_Year_Panel_Enriched.csv"

# ========== Step 1: Extract centroids from WKT ==========
print("=" * 60)
print("Step 1: Extracting centroids from LUI 2024 geometry...")
print("=" * 60)
t0 = time.time()

def parse_wkt_centroid(wkt):
    """Extract centroid from WKT MULTIPOLYGON by averaging all vertex coords."""
    if not wkt or wkt.strip() == "":
        return None, None
    nums = re.findall(r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', wkt)
    if not nums:
        return None, None
    lons = [float(n[0]) for n in nums]
    lats = [float(n[1]) for n in nums]
    return sum(lats) / len(lats), sum(lons) / len(lons)

centroids = {}
n_total = 0
n_geo = 0

with open(LUI_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        n_total += 1
        pid = (row.get("PARCEL_ID_10") or "").strip()
        if not pid:
            continue
        wkt = row.get("the_geom") or ""
        lat, lon = parse_wkt_centroid(wkt)
        if lat is not None:
            centroids[pid] = (lat, lon)
            n_geo += 1

elapsed = time.time() - t0
print(f"  Total LUI rows: {n_total:,}")
print(f"  With geometry: {n_geo:,}")
print(f"  Unique parcels: {len(centroids):,}")
print(f"  Time: {elapsed:.1f}s")

# ========== Step 2: Write centroid reference file ==========
print()
print("=" * 60)
print("Step 2: Writing centroid reference CSV...")
print("=" * 60)

os.makedirs(os.path.dirname(OUT_CENTROIDS), exist_ok=True)

with open(OUT_CENTROIDS, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["parcel_id_10", "latitude", "longitude"])
    for pid in sorted(centroids):
        lat, lon = centroids[pid]
        writer.writerow([pid, f"{lat:.8f}", f"{lon:.8f}"])

print(f"  Written {len(centroids):,} centroids to {OUT_CENTROIDS}")

# ========== Step 3: Build TCAD -> centroid lookup ==========
print()
print("=" * 60)
print("Step 3: Mapping panel TCADs to centroids...")
print("=" * 60)

def normalize_tcad(tid):
    if not tid:
        return ""
    return tid.replace("-", "").replace(" ", "").lstrip("0")

norm_centroids = {}
for pid, (lat, lon) in centroids.items():
    norm = normalize_tcad(pid)
    norm_centroids[norm] = (lat, lon)

print(f"  Normalized centroid lookup: {len(norm_centroids):,} entries")

# ========== Step 4: Update panel with centroids ==========
print()
print("=" * 60)
print("Step 4: Updating panel lat/lon from LUI centroids...")
print("=" * 60)
t0 = time.time()

TEMP_OUT = "Data/Panel/Output/Property_Year_Panel_Enriched_tmp.csv"

with open(PANEL_IN, "r", encoding="utf-8") as fin:
    reader = csv.DictReader(fin)
    fieldnames = list(reader.fieldnames)

    if "latitude" not in fieldnames:
        fieldnames.append("latitude")
    if "longitude" not in fieldnames:
        fieldnames.append("longitude")

    n_written = 0
    n_updated = 0

    with open(TEMP_OUT, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            pid = row.get("standardized_tcad_id", "")
            norm = normalize_tcad(pid)

            if norm in norm_centroids:
                lat, lon = norm_centroids[norm]
                row["latitude"] = f"{lat:.8f}"
                row["longitude"] = f"{lon:.8f}"
                n_updated += 1

            writer.writerow(row)
            n_written += 1

elapsed = time.time() - t0
print(f"  Written: {n_written:,} rows in {elapsed:.1f}s")
print(f"  Updated with centroid: {n_updated:,} ({n_updated/n_written*100:.1f}%)")

shutil.move(TEMP_OUT, PANEL_OUT)
print(f"  Replaced {PANEL_OUT}")

# ========== Summary ==========
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  LUI parcels with geometry: {n_geo:,}")
print(f"  Panel rows with centroids: {n_updated:,} / {n_written:,}")
print(f"  Coverage: {n_updated/n_written*100:.1f}%")
print(f"  Reference file: {OUT_CENTROIDS}")
