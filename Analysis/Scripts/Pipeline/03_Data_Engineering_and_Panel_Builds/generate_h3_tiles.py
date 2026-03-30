"""
Generate H3 hexagonal GeoJSON from protest scores.
Aggregates parcel-level protest probabilities into H3 hexagons at
resolutions 6-9 to match the properlytic upstream tile pipeline.

Output: One GeoJSON per resolution with hex polygon geometry and
aggregated protest_prob, ensemble_prob, n_parcels, etc.
"""
import csv, json, os
import numpy as np
from collections import defaultdict
import h3

SCORES = "Analysis/Results/Experiments/exp02_isotonic/per_parcel_scores.csv"
RAW_SCORES = "Analysis/Results/Diffusion_v3/per_parcel_scores.csv"
OUT_DIR = "v0-properlytic-8v/public/data/h3"
os.makedirs(OUT_DIR, exist_ok=True)

# Load lat/lon
print("Loading lat/lon...")
latlon = {}
with open(RAW_SCORES) as f:
    for row in csv.DictReader(f):
        lat = float(row.get("lat", 0))
        lon = float(row.get("lon", 0))
        if lat > 0 and lon < 0:
            latlon[row["parcel_id"]] = (lat, lon)

# Load calibrated scores — use latest year per parcel
print("Loading calibrated scores...")
parcel_scores = {}
with open(SCORES) as f:
    for row in csv.DictReader(f):
        pid = row["pid"]
        year = int(row["year"])
        if pid not in parcel_scores or year > parcel_scores[pid]["year"]:
            parcel_scores[pid] = {
                "year": year,
                "protest_prob": float(row["diff_calibrated"]),
                "ensemble_prob": float(row["ens_calibrated"]),
                "lr_prob": float(row["lr"]),
                "actual": float(row["actual"]),
            }

print(f"  {len(parcel_scores)} parcels with scores, {len(latlon)} with coords")

# For each resolution, aggregate into hexagons
for res in [6, 7, 8, 9]:
    print(f"\nResolution {res}:")
    hex_data = defaultdict(list)

    for pid, scores in parcel_scores.items():
        if pid not in latlon:
            continue
        lat, lon = latlon[pid]
        h3_id = h3.latlng_to_cell(lat, lon, res)
        hex_data[h3_id].append(scores)

    print(f"  {len(hex_data)} hexagons")

    features = []
    for h3_id, parcels in hex_data.items():
        n = len(parcels)
        protest_prob = np.mean([p["protest_prob"] for p in parcels])
        ensemble_prob = np.mean([p["ensemble_prob"] for p in parcels])
        lr_prob = np.mean([p["lr_prob"] for p in parcels])
        actual_rate = np.mean([p["actual"] for p in parcels])
        n_protested = sum(1 for p in parcels if p["actual"] > 0.5)

        # Get hex boundary as polygon
        boundary = h3.cell_to_boundary(h3_id)
        # h3 returns (lat, lon) tuples; GeoJSON needs [lon, lat]
        coords = [[lon, lat] for lat, lon in boundary]
        coords.append(coords[0])  # close polygon

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {
                "id": h3_id,
                "O": round(float(protest_prob * 100), 3),  # "Opportunity" → protest prob %
                "R": round(float(1.0 - float(np.std([p["protest_prob"] for p in parcels]) / max(protest_prob, 0.001))), 3),  # reliability
                "protest_prob": round(float(protest_prob), 5),
                "ensemble_prob": round(float(ensemble_prob), 5),
                "lr_prob": round(float(lr_prob), 5),
                "actual_rate": round(float(actual_rate), 5),
                "n_accts": n,
                "n_protested": n_protested,
                "has_data": True,
            }
        })

    geojson = {"type": "FeatureCollection", "features": features}
    out_path = os.path.join(OUT_DIR, f"h3_res{res}.json")
    with open(out_path, "w") as f:
        json.dump(geojson, f)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Written {len(features)} hexagons to {out_path} ({size_mb:.1f} MB)")

    # Stats
    probs = [f["properties"]["protest_prob"] for f in features]
    print(f"  Prob range: {min(probs):.5f} - {max(probs):.5f}")
    print(f"  Mean: {np.mean(probs)*100:.3f}%")
    print(f"  Parcels/hex: min={min(len(v) for v in hex_data.values())}, "
          f"median={int(np.median([len(v) for v in hex_data.values()]))}, "
          f"max={max(len(v) for v in hex_data.values())}")

print("\nDone! H3 hexagonal GeoJSON files ready.")
