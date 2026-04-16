"""
Download TCAD parcel boundaries from Travis County ArcGIS REST API.
Paginates through ~382K parcels at 2000/batch, writing GeoJSON.
Only downloads parcel ID + geometry (minimal fields for map rendering).
"""
import urllib.request, json, os, time, sys

OUT_DIR = "Data/GIS/TCAD"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_URL = (
    "https://gis.traviscountytx.gov/server1/rest/services/"
    "Boundaries_and_Jurisdictions/TCAD_public/MapServer/0/query"
)
BATCH_SIZE = 2000
FIELDS = "PROP_ID,geo_id"  # minimal — we just need ID + geometry

all_features = []
offset = 0
batch_num = 0

while True:
    batch_num += 1
    params = (
        f"?where=1%3D1"
        f"&outFields={FIELDS}"
        f"&outSR=4326"
        f"&f=geojson"
        f"&resultOffset={offset}"
        f"&resultRecordCount={BATCH_SIZE}"
    )
    url = BASE_URL + params

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=60)
            data = json.loads(resp.read())
            break
        except Exception as e:
            print(f"  Retry {attempt+1}/3: {e}")
            time.sleep(2)
    else:
        print(f"FAILED at offset {offset}. Saving partial results.")
        break

    features = data.get("features", [])
    n = len(features)
    all_features.extend(features)

    if batch_num % 10 == 0 or n < BATCH_SIZE:
        print(f"  Batch {batch_num}: {n} features (total: {len(all_features)})")

    if n < BATCH_SIZE:
        print(f"Done! Total features: {len(all_features)}")
        break

    offset += BATCH_SIZE
    time.sleep(0.1)  # be polite to the server

# Write combined GeoJSON
out_path = os.path.join(OUT_DIR, "tcad_parcels.geojson")
geojson = {"type": "FeatureCollection", "features": all_features}
print(f"Writing {len(all_features)} features to {out_path}...")
with open(out_path, "w") as f:
    json.dump(geojson, f)

size_mb = os.path.getsize(out_path) / 1e6
print(f"Saved: {size_mb:.1f} MB")
