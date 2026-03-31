"""Build interactive protest risk heatmap."""
import csv, json, sys, os
csv.field_size_limit(min(sys.maxsize, 2**31-1))

FORECAST_PATH = None
for f in os.listdir("Analysis/Results"):
    if f.startswith("forecast_scores_"):
        FORECAST_PATH = os.path.join("Analysis/Results", f)
print("Forecast:", FORECAST_PATH)

# Load forecast scores
scores = {}
with open(FORECAST_PATH, "r") as f:
    for row in csv.DictReader(f):
        scores[row["standardized_tcad_id"]] = float(row["protest_probability"])

# Load coords from parcel_centroids.csv (extracted from LUI geometry)
coords = {}
with open("Data/Panel/Reference/parcel_centroids.csv", "r") as f:
    for row in csv.DictReader(f):
        pid = row["parcel_id_10"]
        lat = row.get("latitude", "")
        lon = row.get("longitude", "")
        if lat and lon:
            try:
                coords[pid] = (float(lat), float(lon))
            except ValueError:
                pass

print("Parcels with scores: %d" % len(scores))
print("Parcels with coords: %d" % len(coords))

# Join: only parcels with both scores AND coords
joined = []
for pid, score in scores.items():
    if pid in coords:
        lat, lon = coords[pid]
        if -98.5 < lon < -97.0 and 29.5 < lat < 31.0:  # Austin bounding box
            joined.append({"lat": lat, "lon": lon, "score": score, "id": pid})

print("Joined parcels with valid Austin coords: %d" % len(joined))

# For heatmap efficiency, bin into tiers
high_risk = [p for p in joined if p["score"] >= 0.5]
medium_risk = [p for p in joined if 0.1 <= p["score"] < 0.5]
low_risk_sample = [p for p in joined if p["score"] < 0.1]

# Sample low-risk for performance (too many for browser)
import random
random.seed(42)
if len(low_risk_sample) > 10000:
    low_risk_sample = random.sample(low_risk_sample, 10000)

print("High risk (>=0.5): %d" % len(high_risk))
print("Medium risk (0.1-0.5): %d" % len(medium_risk))
print("Low risk sample (<0.1): %d" % len(low_risk_sample))

# Also load actual protest locations for overlay
actual_protests = []
with open("Data/Panel/Output/Property_Year_Panel_v3.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    seen = set()
    for row in reader:
        if row["year"] == "2024" and row["protest"] == "1":
            pid = row["standardized_tcad_id"]
            if pid not in seen and pid in coords:
                lat, lon = coords[pid]
                actual_protests.append({"lat": lat, "lon": lon, "id": pid})
                seen.add(pid)
print("Actual 2024 protest locations with coords: %d" % len(actual_protests))

# Generate HTML
all_points = high_risk + medium_risk + low_risk_sample
map_data = {
    "high_risk": [{"lat": p["lat"], "lon": p["lon"], "score": round(p["score"], 4), "id": p["id"]} for p in high_risk],
    "medium_risk": [{"lat": p["lat"], "lon": p["lon"], "score": round(p["score"], 4), "id": p["id"]} for p in medium_risk],
    "low_risk": [{"lat": p["lat"], "lon": p["lon"], "score": round(p["score"], 4), "id": p["id"]} for p in low_risk_sample[:5000]],
    "actual_protests": actual_protests,
}

html = """<!DOCTYPE html>
<html>
<head>
<title>Austin Protest Risk Heatmap</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>
  body { margin: 0; font-family: 'Inter', system-ui, sans-serif; background: #0a0a0a; }
  #map { height: 100vh; width: 100vw; }
  .info-panel {
    position: absolute; top: 16px; right: 16px; z-index: 1000;
    background: rgba(10,10,10,0.92); backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1); border-radius: 12px;
    padding: 20px; color: #e0e0e0; max-width: 320px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
  }
  .info-panel h2 { margin: 0 0 8px 0; color: #fff; font-size: 16px; }
  .info-panel p { margin: 4px 0; font-size: 13px; line-height: 1.5; color: #aaa; }
  .legend { display: flex; flex-direction: column; gap: 6px; margin-top: 12px; }
  .legend-item { display: flex; align-items: center; gap: 8px; font-size: 12px; }
  .legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
  .controls { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
  .controls label { font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 6px; }
  .controls input[type=checkbox] { accent-color: #6366f1; }
  .stat { color: #7c3aed; font-weight: 600; }
</style>
</head>
<body>
<div id="map"></div>
<div class="info-panel">
  <h2>🗺️ Austin Zoning Opposition Risk</h2>
  <p>Logistic regression forecast scores for <span class="stat">TOTAL_SCORED</span> parcels.</p>
  <p>Model: expanding window (2019-2023 → 2024)</p>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div> High risk ≥50% (<span class="stat">HIGH_COUNT</span>)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div> Medium 10-50% (<span class="stat">MED_COUNT</span>)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div> Low risk <10% (sampled)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#3b82f6; border: 2px solid #93c5fd;"></div> Actual 2024 protests (<span class="stat">ACTUAL_COUNT</span>)</div>
  </div>
  <div class="controls">
    <label><input type="checkbox" id="toggleHeat" checked> Heatmap layer</label>
    <label><input type="checkbox" id="toggleMarkers"> Individual markers</label>
    <label><input type="checkbox" id="toggleActual" checked> Actual protests</label>
  </div>
</div>
<script>
const DATA = MAP_DATA_JSON;

const map = L.map('map', {
  center: [30.27, -97.74],
  zoom: 11,
  zoomControl: true,
  preferCanvas: true
});

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
  maxZoom: 19
}).addTo(map);

// Heatmap layer
const heatData = [];
DATA.high_risk.forEach(p => heatData.push([p.lat, p.lon, p.score * 2]));
DATA.medium_risk.forEach(p => heatData.push([p.lat, p.lon, p.score]));
DATA.low_risk.forEach(p => heatData.push([p.lat, p.lon, p.score * 0.3]));

const heatLayer = L.heatLayer(heatData, {
  radius: 18,
  blur: 25,
  maxZoom: 15,
  max: 1.0,
  gradient: {0.1: '#1a1a2e', 0.3: '#16213e', 0.5: '#e94560', 0.7: '#ff6b6b', 0.9: '#ffd93d', 1.0: '#fff'}
}).addTo(map);

// Marker layers
const markerLayer = L.layerGroup();
function makeMarkers() {
  DATA.high_risk.forEach(p => {
    L.circleMarker([p.lat, p.lon], {
      radius: 5, color: '#ef4444', fillColor: '#ef4444', fillOpacity: 0.7, weight: 1
    }).bindPopup('<b>' + p.id + '</b><br>Risk: ' + (p.score*100).toFixed(1) + '%').addTo(markerLayer);
  });
  DATA.medium_risk.forEach(p => {
    L.circleMarker([p.lat, p.lon], {
      radius: 3, color: '#f59e0b', fillColor: '#f59e0b', fillOpacity: 0.5, weight: 1
    }).bindPopup('<b>' + p.id + '</b><br>Risk: ' + (p.score*100).toFixed(1) + '%').addTo(markerLayer);
  });
}
makeMarkers();

// Actual protest overlay
const actualLayer = L.layerGroup();
DATA.actual_protests.forEach(p => {
  L.circleMarker([p.lat, p.lon], {
    radius: 6, color: '#93c5fd', fillColor: '#3b82f6', fillOpacity: 0.9, weight: 2
  }).bindPopup('<b>' + p.id + '</b><br>Actual 2024 protest').addTo(actualLayer);
});
actualLayer.addTo(map);

// Controls
document.getElementById('toggleHeat').addEventListener('change', e => {
  e.target.checked ? map.addLayer(heatLayer) : map.removeLayer(heatLayer);
});
document.getElementById('toggleMarkers').addEventListener('change', e => {
  e.target.checked ? map.addLayer(markerLayer) : map.removeLayer(markerLayer);
});
document.getElementById('toggleActual').addEventListener('change', e => {
  e.target.checked ? map.addLayer(actualLayer) : map.removeLayer(actualLayer);
});
</script>
</body>
</html>"""

# Inject data and counts
html = html.replace("MAP_DATA_JSON", json.dumps(map_data))
html = html.replace("TOTAL_SCORED", "{:,}".format(len(scores)))
html = html.replace("HIGH_COUNT", "{:,}".format(len(high_risk)))
html = html.replace("MED_COUNT", "{:,}".format(len(medium_risk)))
html = html.replace("ACTUAL_COUNT", "{:,}".format(len(actual_protests)))

out_path = "Analysis/Results/Visualizations/protest_risk_heatmap.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("Heatmap saved to %s" % out_path)
print("File size: %.1f KB" % (os.path.getsize(out_path) / 1024))
