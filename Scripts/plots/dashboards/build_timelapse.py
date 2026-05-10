"""Generate per-year backtest scores + forecast + interactive timelapse heatmap."""
import csv, json, sys, os, re
import numpy as np
from collections import defaultdict

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

PANEL_PATH = (PANEL_DIR / "Output/Property_Year_Panel_Enriched.csv")
CENTROIDS_PATH = (PANEL_DIR / "Reference/parcel_centroids.csv")
OUT_DIR = "Analysis/Results"
TRAIN_START = 2019
EVAL_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
FORECAST_YEARS = [2026, 2027, 2028]  # OOT: match backtested horizons h=1,2,3

NUMERIC_FEATURES = [
    "market_value", "assessed_value", "land_value", "improvement_value",
    "living_area", "deed_acreage", "year_built", "land_acres", "improvement_count",
]
CATEGORICAL_FEATURES = ["property_category_code", "lui_general_land_use", "council_district"]
TARGET = "protest"

def safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (ValueError, TypeError):
        return default

# ---- Load centroids ----
print("Loading centroids...")
centroids = {}
with open(CENTROIDS_PATH, "r") as f:
    for row in csv.DictReader(f):
        centroids[row["parcel_id_10"]] = (float(row["latitude"]), float(row["longitude"]))
print("Centroids: %d" % len(centroids))

# ---- Load panel (v3 — all rows for train/eval years) ----
print("Loading panel...")
rows_by_year = defaultdict(list)
with open(PANEL_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        year = int(row["year"])
        if year < TRAIN_START:
            continue
        rows_by_year[year].append(row)

for y in sorted(rows_by_year):
    n_pos = sum(1 for r in rows_by_year[y] if r[TARGET] == "1")
    print("  Year %d: %d rows, %d protests" % (y, len(rows_by_year[y]), n_pos))

# ---- Load 2025 EARS data directly (not yet in panel) ----
EARS_2025 = (PANEL_DIR / "Intermediate/ears_2025_clean.csv")
if os.path.exists(EARS_2025) and 2025 not in rows_by_year:
    print("Loading 2025 EARS data...")
    with open(EARS_2025, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("_parcel_id_10", "").strip()
            if not pid:
                continue
            # Detect protest: check if record_type indicates protest
            protest_val = row.get("protest", "0")
            panel_row = {
                "standardized_tcad_id": pid,
                "year": "2025",
                "ears_matched": "1",
                "ears_source": "ears_2025",
                "protest": protest_val,
                "market_value": row.get("total_market_value", ""),
                "assessed_value": row.get("assessed_value", ""),
                "land_value": row.get("land_market_value", ""),
                "improvement_value": row.get("improvement_market_value", ""),
                "living_area": "",
                "deed_acreage": row.get("deed_acreage", ""),
                "year_built": row.get("year_built", ""),
                "land_acres": row.get("land_acres", ""),
                "improvement_count": "",
                "property_category_code": row.get("property_category_code", ""),
                "lui_general_land_use": "",
                "council_district": "",
            }
            rows_by_year[2025].append(panel_row)
    n_pos = sum(1 for r in rows_by_year[2025] if r[TARGET] == "1")
    print("  Year 2025: %d rows, %d protests" % (len(rows_by_year[2025]), n_pos))

    # Carry forward LUI + council_district from 2024 panel data
    print("  Carrying forward LUI/council from 2024...")
    lookup_2024 = {}
    for r in rows_by_year.get(2024, []):
        pid = r["standardized_tcad_id"]
        lookup_2024[pid] = {
            "lui_general_land_use": r.get("lui_general_land_use", ""),
            "council_district": r.get("council_district", ""),
        }
    filled = 0
    for r in rows_by_year[2025]:
        pid = r["standardized_tcad_id"]
        if pid in lookup_2024:
            r["lui_general_land_use"] = lookup_2024[pid]["lui_general_land_use"]
            r["council_district"] = lookup_2024[pid]["council_district"]
            filled += 1
    rows_by_year[2025] = [
        r for r in rows_by_year[2025]
        if r["lui_general_land_use"]  # Council is 99% missing in both years, so don't filter on it
    ]
    print("  Filtered to %d high-quality 2025 rows (valid LUI only)" % len(rows_by_year[2025]))

# ---- Collect unique categories across all years ----
print("Building feature maps...")
cat_values = {feat: set() for feat in CATEGORICAL_FEATURES}
for year_rows in rows_by_year.values():
    for row in year_rows:
        for feat in CATEGORICAL_FEATURES:
            val = row.get(feat, "").strip()
            if val:
                cat_values[feat].add(val)

cat_maps = {}
for feat in CATEGORICAL_FEATURES:
    vals = sorted(cat_values[feat])
    cat_maps[feat] = {v: i for i, v in enumerate(vals)}

n_numeric = len(NUMERIC_FEATURES)
n_cat = sum(len(m) for m in cat_maps.values())
n_features = n_numeric + n_cat
print("Features: %d" % n_features)

def featurize_rows(rows):
    X = np.zeros((len(rows), n_features), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int32)
    ids = []
    for i, row in enumerate(rows):
        for j, feat in enumerate(NUMERIC_FEATURES):
            X[i, j] = safe_float(row.get(feat, ""))
        offset = n_numeric
        for feat in CATEGORICAL_FEATURES:
            val = row.get(feat, "").strip()
            if val and val in cat_maps[feat]:
                X[i, offset + cat_maps[feat][val]] = 1.0
            offset += len(cat_maps[feat])
        y[i] = int(row[TARGET])
        ids.append(row.get("standardized_tcad_id", ""))
    return X, y, ids

# ---- Run expanding window, save per-year scores ----
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

year_data = {}  # year -> {pid: {score, actual, lat, lon}}

for eval_year in EVAL_YEARS:
    print("\n--- Eval year %d ---" % eval_year)
    # Train: all years < eval_year
    train_rows = []
    for y in range(TRAIN_START, eval_year):
        train_rows.extend(rows_by_year[y])

    test_rows = rows_by_year[eval_year]

    X_train, y_train, _ = featurize_rows(train_rows)
    X_test, y_test, test_ids = featurize_rows(test_rows)

    scaler = StandardScaler()
    X_train_s = np.nan_to_num(scaler.fit_transform(X_train), nan=0, posinf=0, neginf=0)
    X_test_s = np.nan_to_num(scaler.transform(X_test), nan=0, posinf=0, neginf=0)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
    model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_test_s)[:, 1]

    parcels = {}
    for i, pid in enumerate(test_ids):
        if pid in centroids:
            lat, lon = centroids[pid]
            parcels[pid] = {"score": float(probs[i]), "actual": int(y_test[i]), "lat": lat, "lon": lon}

    year_data[eval_year] = parcels
    n_actual = sum(1 for p in parcels.values() if p["actual"] == 1)
    n_high = sum(1 for p in parcels.values() if p["score"] >= 0.5)
    print("  Scored %d parcels, %d actual protests, %d high-risk" % (len(parcels), n_actual, n_high))

# Also add actual-only data for 2019 (no predictions, just actuals)
parcels_2019 = {}
for row in rows_by_year[2019]:
    pid = row.get("standardized_tcad_id", "")
    if pid in centroids:
        lat, lon = centroids[pid]
        parcels_2019[pid] = {"score": 0, "actual": int(row[TARGET]), "lat": lat, "lon": lon}
year_data[2019] = parcels_2019

# OOT Forecast: train on ALL data (2019-2024), forecast h=1,2,3 (2025,2026,2027)
print("\n--- OOT Forecast (train all 2019-2024, forecast 2025-2027) ---")
all_train = []
for y in range(TRAIN_START, EVAL_YEARS[-1] + 1):
    all_train.extend(rows_by_year[y])
# Use latest year rows as the scoring universe (features frozen at 2024)
latest_rows = rows_by_year[EVAL_YEARS[-1]]
X_all_train, y_all_train, _ = featurize_rows(all_train)
X_latest, _, latest_ids = featurize_rows(latest_rows)

scaler = StandardScaler()
X_all_s = np.nan_to_num(scaler.fit_transform(X_all_train), nan=0, posinf=0, neginf=0)
X_latest_s = np.nan_to_num(scaler.transform(X_latest), nan=0, posinf=0, neginf=0)

model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
model.fit(X_all_s, y_all_train)
forecast_probs = model.predict_proba(X_latest_s)[:, 1]

# Same scores for all forecast years (features frozen, model is static)
for fc_year in FORECAST_YEARS:
    parcels_forecast = {}
    for i, pid in enumerate(latest_ids):
        if pid in centroids:
            lat, lon = centroids[pid]
            parcels_forecast[pid] = {"score": float(forecast_probs[i]), "actual": -1, "lat": lat, "lon": lon}
    year_data[fc_year] = parcels_forecast
    n_high_fc = sum(1 for p in parcels_forecast.values() if p["score"] >= 0.5)
    print("  Forecast %d (h=%d): %d parcels, %d high-risk" % (fc_year, fc_year - EVAL_YEARS[-1], len(parcels_forecast), n_high_fc))

# ---- Build per-year JSON for the map ----
print("\nBuilding map data...")

import random
import sys, os
ROOT_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR_PATH not in sys.path: sys.path.append(ROOT_DIR_PATH)
from pipeline.config.paths import DATA_DIR, PANEL_DIR, PROTEST_PETITIONS_DIR, GIS_DIR, ZONING_CASES_DIR

random.seed(42)

map_years = {}
for year, parcels in sorted(year_data.items()):
    high = [{"la": round(p["lat"],5), "lo": round(p["lon"],5), "s": round(p["score"],3), "a": p["actual"]}
            for p in parcels.values() if p["score"] >= 0.3]
    medium = [{"la": round(p["lat"],5), "lo": round(p["lon"],5), "s": round(p["score"],3), "a": p["actual"]}
              for p in parcels.values() if 0.05 <= p["score"] < 0.3]
    # sample medium for size
    if len(medium) > 15000:
        medium = random.sample(medium, 15000)
    # actual protests (regardless of score)
    actuals = [{"la": round(p["lat"],5), "lo": round(p["lon"],5), "s": round(p["score"],3)}
               for p in parcels.values() if p["actual"] == 1]

    map_years[str(year)] = {"high": high, "med": medium, "act": actuals}
    print("  Year %d: %d high, %d med, %d actual" % (year, len(high), len(medium), len(actuals)))

# ---- Generate HTML ----
html = """<!DOCTYPE html>
<html>
<head>
<title>Austin Zoning Opposition — Backtest Timelapse</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: 'Inter', system-ui, sans-serif; background: #0a0a0a; overflow: hidden; }
  #map { height: 100vh; width: 100vw; }
  .panel {
    position: absolute; top: 16px; right: 16px; z-index: 1000;
    background: rgba(10,10,10,0.94); backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 14px;
    padding: 22px; color: #e0e0e0; width: 340px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.7);
  }
  .panel h2 { margin: 0 0 4px; color: #fff; font-size: 17px; font-weight: 700; }
  .panel .sub { font-size: 12px; color: #888; margin-bottom: 14px; }
  .year-display { text-align: center; margin: 10px 0 4px; }
  .year-num { font-size: 48px; font-weight: 700; color: #fff; letter-spacing: -2px; }
  .year-label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 2px; }
  .slider-row { display: flex; align-items: center; gap: 8px; margin: 8px 0 16px; }
  .slider-row input[type=range] { flex: 1; accent-color: #7c3aed; height: 6px; }
  .slider-row button {
    background: rgba(124,58,237,0.2); border: 1px solid rgba(124,58,237,0.4);
    color: #a78bfa; border-radius: 6px; padding: 4px 10px; cursor: pointer;
    font-size: 12px; font-family: inherit;
  }
  .slider-row button:hover { background: rgba(124,58,237,0.4); }
  .slider-row button.active { background: #7c3aed; color: #fff; }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 12px 0; }
  .stat-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px; padding: 10px; text-align: center;
  }
  .stat-val { font-size: 22px; font-weight: 700; }
  .stat-label { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }
  .controls { display: flex; flex-direction: column; gap: 6px; margin-top: 12px; }
  .controls label {
    font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 8px;
    padding: 4px 0;
  }
  .controls input[type=checkbox] { accent-color: #7c3aed; }
  .legend { display: flex; gap: 12px; margin-top: 12px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 4px; font-size: 11px; color: #aaa; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
  .forecast-badge {
    display: inline-block; background: linear-gradient(135deg, #7c3aed, #ec4899);
    color: #fff; font-size: 10px; font-weight: 600; padding: 2px 8px;
    border-radius: 10px; margin-left: 6px; text-transform: uppercase; letter-spacing: 1px;
  }
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <h2>🗺️ Austin Zoning Opposition Risk</h2>
  <div class="sub">Expanding window backtest + OOT forecast</div>
  <div class="year-display">
    <div class="year-label" id="yearLabel">Backtest</div>
    <div class="year-num" id="yearNum">2020</div>
  </div>
  <div class="slider-row">
    <button id="playBtn" onclick="togglePlay()">▶</button>
    <input type="range" id="yearSlider" min="2019" max="2028" value="2020" step="1">
    <span id="speedLabel" style="font-size:11px;color:#888;width:30px">1x</span>
  </div>
  <div class="stats">
    <div class="stat-card">
      <div class="stat-val" style="color:#ef4444" id="nHigh">—</div>
      <div class="stat-label">High Risk</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color:#3b82f6" id="nActual">—</div>
      <div class="stat-label">Actual Protests</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color:#22c55e" id="nTP">—</div>
      <div class="stat-label">True Positives</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color:#f59e0b" id="nPrecision">—</div>
      <div class="stat-label">Precision@30%</div>
    </div>
  </div>
  <div class="controls">
    <label><input type="checkbox" id="toggleHeat" checked onchange="updateMap()"> Heatmap (predicted risk)</label>
    <label><input type="checkbox" id="toggleActual" checked onchange="updateMap()"> Actual protests (blue dots)</label>
    <label><input type="checkbox" id="toggleMarkers" onchange="updateMap()"> Individual risk markers</label>
  </div>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>High ≥30%</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>Med 5-30%</div>
    <div class="legend-item"><div class="legend-dot" style="background:#3b82f6;border:2px solid #93c5fd"></div>Actual</div>
  </div>
</div>
<script>
const DATA = __MAP_DATA__;
const YEARS = Object.keys(DATA).map(Number).sort();

const map = L.map('map', { center: [30.30, -97.74], zoom: 11, zoomControl: true, preferCanvas: true });
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: '© OSM © CARTO', maxZoom: 19
}).addTo(map);

let heatLayer = null;
let markerLayer = L.layerGroup();
let actualLayer = L.layerGroup();
let playing = false;
let playInterval = null;

function updateMap() {
  const year = parseInt(document.getElementById('yearSlider').value);
  const d = DATA[year];
  if (!d) return;

  document.getElementById('yearNum').textContent = year;
  document.getElementById('yearLabel').innerHTML = year >= 2026
    ? 'Forecast <span class="forecast-badge">OOT h' + (year - 2025) + '</span>'
    : year === 2019 ? 'Actuals Only' : year === 2025 ? 'Backtest (no outcome)' : 'Backtest';

  // Stats
  const nHigh = d.high.length;
  const nActual = d.act.length;
  const tp = d.high.filter(p => p.a === 1).length;
  const prec = nHigh > 0 ? (tp / nHigh * 100).toFixed(1) + '%' : '—';
  document.getElementById('nHigh').textContent = nHigh.toLocaleString();
  document.getElementById('nActual').textContent = year >= 2026 ? '?' : nActual.toLocaleString();
  document.getElementById('nTP').textContent = year >= 2026 ? '?' : tp.toLocaleString();
  document.getElementById('nPrecision').textContent = year >= 2025 || year === 2019 ? '?' : prec;

  // Heatmap
  if (heatLayer) map.removeLayer(heatLayer);
  if (document.getElementById('toggleHeat').checked) {
    const heatData = [];
    d.high.forEach(p => heatData.push([p.la, p.lo, Math.min(p.s * 2, 1)]));
    d.med.forEach(p => heatData.push([p.la, p.lo, p.s * 0.5]));
    heatLayer = L.heatLayer(heatData, {
      radius: 16, blur: 22, maxZoom: 15, max: 1.0,
      gradient: {0.1:'#16213e', 0.3:'#1a1a4e', 0.5:'#e94560', 0.7:'#ff6b6b', 0.9:'#ffd93d', 1.0:'#fff'}
    }).addTo(map);
  }

  // Individual markers
  markerLayer.clearLayers();
  if (document.getElementById('toggleMarkers').checked) {
    d.high.forEach(p => {
      const c = p.a === 1 ? '#22c55e' : '#ef4444';
      L.circleMarker([p.la, p.lo], {radius:4, color:c, fillColor:c, fillOpacity:0.7, weight:1})
        .bindPopup('Risk: ' + (p.s*100).toFixed(1) + '%<br>Actual: ' + (p.a === 1 ? 'YES' : p.a === -1 ? 'TBD' : 'no'))
        .addTo(markerLayer);
    });
    markerLayer.addTo(map);
  }

  // Actual protests
  actualLayer.clearLayers();
  if (document.getElementById('toggleActual').checked && year < 2026) {
    d.act.forEach(p => {
      L.circleMarker([p.la, p.lo], {radius:5, color:'#93c5fd', fillColor:'#3b82f6', fillOpacity:0.9, weight:2})
        .bindPopup('Actual protest<br>Predicted risk: ' + (p.s*100).toFixed(1) + '%')
        .addTo(actualLayer);
    });
    actualLayer.addTo(map);
  }
}

document.getElementById('yearSlider').addEventListener('input', updateMap);

function togglePlay() {
  playing = !playing;
  const btn = document.getElementById('playBtn');
  if (playing) {
    btn.textContent = '⏸';
    btn.classList.add('active');
    let slider = document.getElementById('yearSlider');
    playInterval = setInterval(() => {
      let v = parseInt(slider.value) + 1;
      if (v > 2028) v = 2019;
      slider.value = v;
      updateMap();
    }, 1500);
  } else {
    btn.textContent = '▶';
    btn.classList.remove('active');
    clearInterval(playInterval);
  }
}

// Init
updateMap();
</script>
</body>
</html>"""

html = html.replace("__MAP_DATA__", json.dumps(map_years))

out_path = os.path.join(OUT_DIR, "protest_timelapse.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print("\nTimelapse saved to %s (%.1f KB)" % (out_path, os.path.getsize(out_path) / 1024))
