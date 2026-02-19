"""Extract lat/lon centroids from LUI geometry for all parcels."""
import csv, json, sys, os
csv.field_size_limit(min(sys.maxsize, 2**31-1))

LUI_PREFETCHED = "Data/Zoning_Cases/Source_Data/land_use_inventory_prefetched.csv"
LUI_2024 = "Data/CoA_Open_Data/LUI_2024_7vsm-dvxg.csv"
OUT_PATH = "Data/Panel/Reference/parcel_centroids.csv"

def centroid_from_geojson(geom_str):
    """Compute centroid from GeoJSON geometry string."""
    try:
        geom = json.loads(geom_str.replace("'", '"'))
    except (json.JSONDecodeError, ValueError):
        return None, None

    coords_flat = []
    gtype = geom.get("type", "")
    raw = geom.get("coordinates", [])

    if gtype == "Point":
        return raw[1], raw[0]  # lat, lon
    elif gtype == "Polygon":
        for ring in raw:
            coords_flat.extend(ring)
    elif gtype == "MultiPolygon":
        for polygon in raw:
            for ring in polygon:
                coords_flat.extend(ring)
    else:
        return None, None

    if not coords_flat:
        return None, None

    lons = [c[0] for c in coords_flat]
    lats = [c[1] for c in coords_flat]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def centroid_from_wkt(wkt_str):
    """Compute centroid from WKT POLYGON/MULTIPOLYGON string."""
    import re
    # Extract all coordinate pairs
    nums = re.findall(r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', wkt_str)
    if not nums:
        return None, None
    lons = [float(n[0]) for n in nums]
    lats = [float(n[1]) for n in nums]
    return sum(lats) / len(lats), sum(lons) / len(lons)


print("=== Extracting Parcel Centroids ===")

centroids = {}  # parcel_id_10 -> (lat, lon)

# Source 1: prefetched LUI (GeoJSON in the_geom)
print("Reading prefetched LUI...")
with open(LUI_PREFETCHED, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row.get("parcel_id_10", "").strip()
        if not pid or pid in centroids:
            continue
        geom_str = row.get("the_geom", "")
        if geom_str:
            lat, lon = centroid_from_geojson(geom_str)
            if lat and lon and -98.5 < lon < -97.0 and 29.5 < lat < 31.0:
                centroids[pid] = (lat, lon)

print("After prefetched LUI: %d parcels with coords" % len(centroids))

# Source 2: LUI 2024 (WKT in the_geom) — fill gaps
print("Reading LUI 2024...")
with open(LUI_2024, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row.get("PARCEL_ID_10", "").strip()
        if not pid or pid in centroids:
            continue
        geom_str = row.get("the_geom", "")
        if geom_str:
            if geom_str.startswith("MULTI") or geom_str.startswith("POLY"):
                lat, lon = centroid_from_wkt(geom_str)
            else:
                lat, lon = centroid_from_geojson(geom_str)
            if lat and lon and -98.5 < lon < -97.0 and 29.5 < lat < 31.0:
                centroids[pid] = (lat, lon)

print("After LUI 2024: %d parcels with coords" % len(centroids))

# Write centroids
with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["parcel_id_10", "latitude", "longitude"])
    for pid in sorted(centroids.keys()):
        lat, lon = centroids[pid]
        writer.writerow([pid, "%.8f" % lat, "%.8f" % lon])

print("Wrote %d centroids to %s" % (len(centroids), OUT_PATH))
