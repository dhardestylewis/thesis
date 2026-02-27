"""Generate GeoJSON from calibrated protest scores for the properlytic map."""
import csv, json

SCORES = "Analysis/Results/Experiments/exp02_isotonic/per_parcel_scores.csv"
RAW_SCORES = "Analysis/Results/Diffusion_v3/per_parcel_scores.csv"
OUT = "v0-properlytic-8v/public/data/protest_scores.json"

# Load lat/lon
latlon = {}
with open(RAW_SCORES) as f:
    for row in csv.DictReader(f):
        lat = float(row.get("lat", 0))
        lon = float(row.get("lon", 0))
        if lat > 0 and lon < 0:
            latlon[row["parcel_id"]] = (lat, lon)

# Load calibrated scores, group by parcel
from collections import defaultdict
parcel_data = defaultdict(dict)
with open(SCORES) as f:
    for row in csv.DictReader(f):
        pid = row["pid"]
        year = int(row["year"])
        parcel_data[pid][year] = {
            "lr": float(row["lr"]),
            "diff": float(row["diff"]),
            "cal": float(row["diff_calibrated"]),
            "ens": float(row["ens_calibrated"]),
            "actual": float(row["actual"]),
        }

# Build GeoJSON
features = []
for pid, years in parcel_data.items():
    if pid not in latlon:
        continue
    lat, lon = latlon[pid]
    
    # Latest year data for map color
    latest_year = max(years.keys())
    latest = years[latest_year]
    
    # Time series for fan chart
    series = []
    for y in sorted(years.keys()):
        d = years[y]
        series.append({
            "yr": y,
            "cal": round(d["cal"], 5),
            "lr": round(d["lr"], 5),
            "ens": round(d["ens"], 5),
            "actual": int(d["actual"]),
        })
    
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "id": pid,
            "protest_prob": round(latest["cal"], 5),
            "ensemble_prob": round(latest["ens"], 5),
            "lr_prob": round(latest["lr"], 5),
            "actual": int(latest["actual"]),
            "year": latest_year,
            "series": series,
        }
    })

geojson = {
    "type": "FeatureCollection",
    "features": features,
}

with open(OUT, "w") as f:
    json.dump(geojson, f)

print(f"Written {len(features)} features to {OUT}")
print(f"File size: {len(json.dumps(geojson)) / 1e6:.1f} MB")

# Quick stats
probs = [f["properties"]["protest_prob"] for f in features]
print(f"Prob range: {min(probs):.5f} - {max(probs):.5f}")
print(f"Mean: {sum(probs)/len(probs)*100:.3f}%")
