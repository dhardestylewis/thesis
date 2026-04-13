"""
Build Generative Model Timelapse — Per-Parcel Predictions
==========================================================
Extends build_timelapse.py to include LogReg + Diffusion (augmented)
per-parcel predictions on a Leaflet map with model toggle + year slider.

Includes 2025 out-of-sample forecast (covariates only, no protest labels).
"""
import csv, json, sys, os, time, random
import numpy as np
from collections import defaultdict

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
random.seed(42)

PANEL_PATH = "Data/Panel/Output/Property_Year_Panel_Enriched.csv"
CENTROIDS_PATH = "Data/Panel/Reference/parcel_centroids.csv"
OUT_DIR = "Analysis/Results"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_START = 2019
EVAL_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
FORECAST_YEARS = [2026, 2027, 2028]

NUMERIC_FEATURES = [
    "market_value", "assessed_value", "land_value", "improvement_value",
    "living_area", "deed_acreage", "year_built", "land_acres", "improvement_count",
]
CATEGORICAL_FEATURES = ["property_category_code", "lui_general_land_use", "council_district"]
TARGET = "protest"

# Diffusion hyperparams
DIFF_TIMESTEPS = 100
DIFF_HIDDEN = 128
DIFF_EPOCHS = 20
DIFF_LR = 1e-3


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

# ---- Load panel ----
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

# ---- Load 2025 EARS if available ----
EARS_2025 = "Data/Panel/Intermediate/ears_2025_clean.csv"
if os.path.exists(EARS_2025) and 2025 not in rows_by_year:
    print("Loading 2025 EARS data...")
    with open(EARS_2025, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("_parcel_id_10", "").strip()
            if not pid:
                continue
            panel_row = {
                "standardized_tcad_id": pid, "year": "2025",
                "ears_matched": "1", "ears_source": "ears_2025",
                "protest": row.get("protest", "0"),
                "market_value": row.get("total_market_value", ""),
                "assessed_value": row.get("assessed_value", ""),
                "land_value": row.get("land_market_value", ""),
                "improvement_value": row.get("improvement_market_value", ""),
                "living_area": "", "deed_acreage": row.get("deed_acreage", ""),
                "year_built": row.get("year_built", ""), "land_acres": row.get("land_acres", ""),
                "improvement_count": "",
                "property_category_code": row.get("property_category_code", ""),
                "lui_general_land_use": "", "council_district": "",
            }
            rows_by_year[2025].append(panel_row)
    # Carry forward categoricals from 2024
    lookup_2024 = {r["standardized_tcad_id"]: r for r in rows_by_year.get(2024, [])}
    for r in rows_by_year[2025]:
        pid = r["standardized_tcad_id"]
        if pid in lookup_2024:
            r["lui_general_land_use"] = lookup_2024[pid].get("lui_general_land_use", "")
            r["council_district"] = lookup_2024[pid].get("council_district", "")
    rows_by_year[2025] = [r for r in rows_by_year[2025] if r["lui_general_land_use"]]
    n_pos = sum(1 for r in rows_by_year[2025] if r[TARGET] == "1")
    print("  Year 2025: %d rows (filtered), %d protests" % (len(rows_by_year[2025]), n_pos))

# ---- Build feature maps ----
print("Building feature maps...")
cat_values = {f: set() for f in CATEGORICAL_FEATURES}
for year_rows in rows_by_year.values():
    for row in year_rows:
        for f in CATEGORICAL_FEATURES:
            val = row.get(f, "").strip()
            if val:
                cat_values[f].add(val)

cat_maps = {}
for f in CATEGORICAL_FEATURES:
    vals = sorted(cat_values[f])
    cat_maps[f] = {v: i for i, v in enumerate(vals)}

n_numeric = len(NUMERIC_FEATURES)
n_cat = sum(len(m) for m in cat_maps.values())
n_features = n_numeric + n_cat
print("Features: %d (%d numeric + %d cat)" % (n_features, n_numeric, n_cat))


def featurize_rows(rows):
    X = np.zeros((len(rows), n_features), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int32)
    ids = []
    for i, row in enumerate(rows):
        for j, f in enumerate(NUMERIC_FEATURES):
            X[i, j] = safe_float(row.get(f, ""))
        offset = n_numeric
        for f in CATEGORICAL_FEATURES:
            val = row.get(f, "").strip()
            if val and val in cat_maps[f]:
                X[i, offset + cat_maps[f][val]] = 1.0
            offset += len(cat_maps[f])
        y[i] = int(row[TARGET])
        ids.append(row.get("standardized_tcad_id", ""))
    return X, y, ids


def train_diffusion_augmented_logreg(X_train_scaled, y_train, X_test_scaled):
    """Train diffusion on minority, generate synthetic samples, augment LogReg."""
    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression

    device = torch.device("cpu")
    minority_mask = y_train == 1
    if minority_mask.sum() < 10:
        # Fallback: plain LogReg
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
        clf.fit(X_train_scaled, y_train)
        return clf.predict_proba(X_test_scaled)[:, 1]

    X_min = X_train_scaled[minority_mask].astype(np.float32)
    input_dim = X_train_scaled.shape[1]

    class DiffusionMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_emb = nn.Sequential(nn.Linear(1, DIFF_HIDDEN), nn.SiLU())
            self.net = nn.Sequential(
                nn.Linear(input_dim + DIFF_HIDDEN, DIFF_HIDDEN), nn.SiLU(),
                nn.Linear(DIFF_HIDDEN, DIFF_HIDDEN), nn.SiLU(),
                nn.Linear(DIFF_HIDDEN, input_dim),
            )
        def forward(self, x, t):
            return self.net(torch.cat([x, self.time_emb(t)], dim=1))

    model = DiffusionMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=DIFF_LR)
    betas = torch.linspace(1e-4, 0.02, DIFF_TIMESTEPS, device=device)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, 0)

    X_min_t = torch.tensor(X_min, device=device)
    model.train()
    for epoch in range(DIFF_EPOCHS):
        idx = torch.randint(0, len(X_min_t), (min(512, len(X_min_t)),))
        x0 = X_min_t[idx]
        t = torch.randint(0, DIFF_TIMESTEPS, (len(x0),), device=device)
        noise = torch.randn_like(x0)
        ab = alpha_bar[t].unsqueeze(1)
        x_noisy = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
        pred = model(x_noisy, t.float().unsqueeze(1) / DIFF_TIMESTEPS)
        loss = nn.functional.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Generate synthetic minority
    n_gen = int(minority_mask.sum())
    model.eval()
    with torch.no_grad():
        x = torch.randn(n_gen, input_dim, device=device)
        for step in reversed(range(DIFF_TIMESTEPS)):
            pred_noise = model(x, torch.full((n_gen, 1), step / DIFF_TIMESTEPS, device=device))
            beta = betas[step]
            alpha = alphas[step]
            ab = alpha_bar[step]
            x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - ab)) * pred_noise)
            if step > 0:
                x += torch.sqrt(beta) * torch.randn_like(x)
    synthetic = x.cpu().numpy()

    # Augment training set and train LogReg
    X_aug = np.vstack([X_train_scaled, synthetic])
    y_aug = np.concatenate([y_train, np.ones(n_gen, dtype=np.int32)])
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X_aug, y_aug)
    return clf.predict_proba(X_test_scaled)[:, 1]


# ---- Run expanding window + save per-parcel ----
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# year -> model -> {pid: {score, actual, lat, lon}}
year_data = {"logreg": {}, "diffusion": {}}
per_parcel_csv_rows = []

for eval_year in EVAL_YEARS:
    print("\n--- Eval year %d ---" % eval_year)
    t0 = time.time()

    train_rows = []
    for y in range(TRAIN_START, eval_year):
        train_rows.extend(rows_by_year[y])
    test_rows = rows_by_year[eval_year]

    X_train, y_train, _ = featurize_rows(train_rows)
    X_test, y_test, test_ids = featurize_rows(test_rows)

    scaler = StandardScaler()
    X_train_s = np.nan_to_num(scaler.fit_transform(X_train), nan=0, posinf=0, neginf=0)
    X_test_s = np.nan_to_num(scaler.transform(X_test), nan=0, posinf=0, neginf=0)

    # LogReg
    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
    lr_model.fit(X_train_s, y_train)
    lr_probs = lr_model.predict_proba(X_test_s)[:, 1]
    print("  LogReg scored %d parcels (%.1fs)" % (len(lr_probs), time.time() - t0))

    # Diffusion-augmented
    t1 = time.time()
    diff_probs = train_diffusion_augmented_logreg(X_train_s, y_train, X_test_s)
    print("  Diffusion scored %d parcels (%.1fs)" % (len(diff_probs), time.time() - t1))

    # Store per-parcel
    lr_parcels, diff_parcels = {}, {}
    for i, pid in enumerate(test_ids):
        if pid in centroids:
            lat, lon = centroids[pid]
            actual = int(y_test[i])
            lr_parcels[pid] = {"score": float(lr_probs[i]), "actual": actual, "lat": lat, "lon": lon}
            diff_parcels[pid] = {"score": float(diff_probs[i]), "actual": actual, "lat": lat, "lon": lon}
            per_parcel_csv_rows.append({
                "parcel_id": pid, "year": eval_year, "y_true": actual,
                "y_prob_logreg": round(float(lr_probs[i]), 5),
                "y_prob_diffusion": round(float(diff_probs[i]), 5),
                "lat": lat, "lon": lon,
            })

    year_data["logreg"][eval_year] = lr_parcels
    year_data["diffusion"][eval_year] = diff_parcels
    print("  Geocoded %d / %d parcels" % (len(lr_parcels), len(test_ids)))

# ---- 2019 actuals only ----
parcels_2019 = {}
for row in rows_by_year[2019]:
    pid = row.get("standardized_tcad_id", "")
    if pid in centroids:
        lat, lon = centroids[pid]
        parcels_2019[pid] = {"score": 0, "actual": int(row[TARGET]), "lat": lat, "lon": lon}
year_data["logreg"][2019] = parcels_2019
year_data["diffusion"][2019] = parcels_2019

# ---- OOT Forecast ----
print("\n--- OOT Forecast (train 2019-2024/2025, forecast 2026-2028) ---")
all_train = []
last_year = max(y for y in rows_by_year if y <= 2025)
for y in range(TRAIN_START, last_year + 1):
    all_train.extend(rows_by_year[y])

latest_rows = rows_by_year[last_year]
X_all, y_all, _ = featurize_rows(all_train)
X_latest, _, latest_ids = featurize_rows(latest_rows)

scaler = StandardScaler()
X_all_s = np.nan_to_num(scaler.fit_transform(X_all), nan=0, posinf=0, neginf=0)
X_latest_s = np.nan_to_num(scaler.transform(X_latest), nan=0, posinf=0, neginf=0)

# LogReg forecast
lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
lr_model.fit(X_all_s, y_all)
lr_forecast = lr_model.predict_proba(X_latest_s)[:, 1]

# Diffusion forecast
diff_forecast = train_diffusion_augmented_logreg(X_all_s, y_all, X_latest_s)

for fc_year in FORECAST_YEARS:
    lr_fc, diff_fc = {}, {}
    for i, pid in enumerate(latest_ids):
        if pid in centroids:
            lat, lon = centroids[pid]
            lr_fc[pid] = {"score": float(lr_forecast[i]), "actual": -1, "lat": lat, "lon": lon}
            diff_fc[pid] = {"score": float(diff_forecast[i]), "actual": -1, "lat": lat, "lon": lon}
            per_parcel_csv_rows.append({
                "parcel_id": pid, "year": fc_year, "y_true": -1,
                "y_prob_logreg": round(float(lr_forecast[i]), 5),
                "y_prob_diffusion": round(float(diff_forecast[i]), 5),
                "lat": lat, "lon": lon,
            })
    year_data["logreg"][fc_year] = lr_fc
    year_data["diffusion"][fc_year] = diff_fc
    n_high = sum(1 for p in lr_fc.values() if p["score"] >= 0.5)
    print("  Forecast %d: %d parcels, %d high-risk (LR)" % (fc_year, len(lr_fc), n_high))

# ---- Save per-parcel CSV ----
csv_path = os.path.join(OUT_DIR, "generative_per_parcel.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["parcel_id", "year", "y_true", "y_prob_logreg", "y_prob_diffusion", "lat", "lon"])
    w.writeheader()
    for row in per_parcel_csv_rows:
        w.writerow(row)
print("\nPer-parcel CSV: %s (%d rows)" % (csv_path, len(per_parcel_csv_rows)))

# ---- Build map JSON ----
print("Building map data...")

def build_map_json(model_data):
    map_years = {}
    for year, parcels in sorted(model_data.items()):
        high = [{"la": round(p["lat"],5), "lo": round(p["lon"],5), "s": round(p["score"],3), "a": p["actual"]}
                for p in parcels.values() if p["score"] >= 0.3]
        medium = [{"la": round(p["lat"],5), "lo": round(p["lon"],5), "s": round(p["score"],3), "a": p["actual"]}
                  for p in parcels.values() if 0.05 <= p["score"] < 0.3]
        if len(medium) > 15000:
            medium = random.sample(medium, 15000)
        actuals = [{"la": round(p["lat"],5), "lo": round(p["lon"],5), "s": round(p["score"],3)}
                   for p in parcels.values() if p["actual"] == 1]
        map_years[str(year)] = {"high": high, "med": medium, "act": actuals}
        print("  Year %d: %d high, %d med, %d actual" % (year, len(high), len(medium), len(actuals)))
    return map_years

print("\nLogReg model data:")
lr_map = build_map_json(year_data["logreg"])
print("\nDiffusion model data:")
diff_map = build_map_json(year_data["diffusion"])

# ---- Generate HTML ----
html = """<!DOCTYPE html>
<html>
<head>
<title>Generative Model Timelapse — Zoning Opposition</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
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
    padding: 22px; color: #e0e0e0; width: 360px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.7);
    max-height: calc(100vh - 32px); overflow-y: auto;
  }
  .panel h2 { margin: 0 0 4px; color: #fff; font-size: 17px; font-weight: 700; }
  .panel .sub { font-size: 12px; color: #888; margin-bottom: 14px; }

  /* Model toggle */
  .model-toggle { display: flex; gap: 4px; margin-bottom: 14px; }
  .model-btn {
    flex: 1; padding: 8px 0; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.04); color: #888; cursor: pointer; text-align: center;
    font-family: inherit; font-size: 12px; font-weight: 600; transition: all 0.2s;
  }
  .model-btn:hover { background: rgba(255,255,255,0.08); color: #ccc; }
  .model-btn.active-lr { background: rgba(59,130,246,0.2); border-color: rgba(59,130,246,0.4); color: #60a5fa; }
  .model-btn.active-diff { background: rgba(236,72,153,0.2); border-color: rgba(236,72,153,0.4); color: #f472b6; }

  .year-display { text-align: center; margin: 10px 0 4px; }
  .year-num { font-size: 48px; font-weight: 800; color: #fff; letter-spacing: -2px; }
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
  .controls label { font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 8px; padding: 4px 0; }
  .controls input[type=checkbox] { accent-color: #7c3aed; }
  .legend { display: flex; gap: 12px; margin-top: 12px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 4px; font-size: 11px; color: #aaa; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%%; }
  .forecast-badge {
    display: inline-block; background: linear-gradient(135deg, #7c3aed, #ec4899);
    color: #fff; font-size: 10px; font-weight: 600; padding: 2px 8px;
    border-radius: 10px; margin-left: 6px; text-transform: uppercase; letter-spacing: 1px;
  }
  .delta-badge {
    display: inline-block; font-size: 10px; font-weight: 600; padding: 2px 6px;
    border-radius: 6px; margin-top: 4px;
  }
  .delta-up { background: rgba(34,197,94,0.15); color: #22c55e; }
  .delta-down { background: rgba(248,113,113,0.15); color: #f87171; }
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <h2>🗺️ Generative Model Timelapse</h2>
  <div class="sub">LogReg vs Diffusion-Augmented — Per-Parcel Risk</div>

  <div class="model-toggle">
    <button class="model-btn active-lr" id="btnLR" onclick="switchModel('logreg')">📊 LogReg</button>
    <button class="model-btn" id="btnDiff" onclick="switchModel('diffusion')">🧬 Diffusion</button>
  </div>

  <div class="year-display">
    <div class="year-label" id="yearLabel">Backtest</div>
    <div class="year-num" id="yearNum">2020</div>
  </div>
  <div class="slider-row">
    <button id="playBtn" onclick="togglePlay()">▶</button>
    <input type="range" id="yearSlider" min="2019" max="2028" value="2020" step="1">
  </div>
  <div class="stats">
    <div class="stat-card">
      <div class="stat-val" style="color:#ef4444" id="nHigh">—</div>
      <div class="stat-label">High Risk (≥30%%)</div>
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
      <div class="stat-label">Precision@30%%</div>
    </div>
  </div>
  <div id="deltaInfo" style="text-align:center;font-size:11px;color:#888;margin:4px 0;"></div>
  <div class="controls">
    <label><input type="checkbox" id="toggleHeat" checked onchange="updateMap()"> Heatmap (predicted risk)</label>
    <label><input type="checkbox" id="toggleActual" checked onchange="updateMap()"> Actual protests (blue dots)</label>
    <label><input type="checkbox" id="toggleMarkers" onchange="updateMap()"> Individual risk markers</label>
  </div>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>High ≥30%%</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>Med 5-30%%</div>
    <div class="legend-item"><div class="legend-dot" style="background:#3b82f6;border:2px solid #93c5fd"></div>Actual</div>
  </div>
</div>

<script>
const DATA_LR = __LR_DATA__;
const DATA_DIFF = __DIFF_DATA__;
let currentModel = 'logreg';

function getData() { return currentModel === 'logreg' ? DATA_LR : DATA_DIFF; }

function switchModel(m) {
  currentModel = m;
  document.getElementById('btnLR').className = 'model-btn' + (m==='logreg' ? ' active-lr' : '');
  document.getElementById('btnDiff').className = 'model-btn' + (m==='diffusion' ? ' active-diff' : '');
  updateMap();
}

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
  const data = getData();
  const d = data[year];
  if (!d) return;

  document.getElementById('yearNum').textContent = year;
  document.getElementById('yearLabel').innerHTML = year >= 2026
    ? 'Forecast <span class="forecast-badge">OOT h' + (year - 2025) + '</span>'
    : year === 2019 ? 'Actuals Only' : year === 2025 ? 'Backtest (no outcome)' : 'Backtest';

  const nHigh = d.high.length;
  const nActual = d.act.length;
  const tp = d.high.filter(p => p.a === 1).length;
  const prec = nHigh > 0 ? (tp / nHigh * 100).toFixed(1) + '%%' : '—';

  document.getElementById('nHigh').textContent = nHigh.toLocaleString();
  document.getElementById('nActual').textContent = year >= 2026 ? '?' : nActual.toLocaleString();
  document.getElementById('nTP').textContent = year >= 2026 ? '?' : tp.toLocaleString();
  document.getElementById('nPrecision').textContent = year >= 2025 || year === 2019 ? '?' : prec;

  // Show delta between models
  const otherData = currentModel === 'logreg' ? DATA_DIFF : DATA_LR;
  const otherD = otherData[year];
  if (otherD && year > 2019) {
    const otherHigh = otherD.high.length;
    const delta = nHigh - otherHigh;
    const otherName = currentModel === 'logreg' ? 'Diffusion' : 'LogReg';
    const cls = delta > 0 ? 'delta-up' : delta < 0 ? 'delta-down' : '';
    document.getElementById('deltaInfo').innerHTML = delta !== 0
      ? '<span class="delta-badge ' + cls + '">' + (delta>0?'+':'') + delta + ' vs ' + otherName + '</span>'
      : '<span style="color:#666">Same count as ' + otherName + '</span>';
  } else {
    document.getElementById('deltaInfo').innerHTML = '';
  }

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

  // Markers
  markerLayer.clearLayers();
  if (document.getElementById('toggleMarkers').checked) {
    d.high.forEach(p => {
      const c = p.a === 1 ? '#22c55e' : '#ef4444';
      L.circleMarker([p.la, p.lo], {radius:4, color:c, fillColor:c, fillOpacity:0.7, weight:1})
        .bindPopup('Risk: ' + (p.s*100).toFixed(1) + '%%<br>Actual: ' + (p.a === 1 ? 'YES' : p.a === -1 ? 'TBD' : 'no')
          + '<br>Model: ' + currentModel)
        .addTo(markerLayer);
    });
    markerLayer.addTo(map);
  }

  // Actuals
  actualLayer.clearLayers();
  if (document.getElementById('toggleActual').checked && year < 2026) {
    d.act.forEach(p => {
      L.circleMarker([p.la, p.lo], {radius:5, color:'#93c5fd', fillColor:'#3b82f6', fillOpacity:0.9, weight:2})
        .bindPopup('Actual protest<br>Predicted: ' + (p.s*100).toFixed(1) + '%%')
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
    btn.textContent = '⏸'; btn.classList.add('active');
    let slider = document.getElementById('yearSlider');
    playInterval = setInterval(() => {
      let v = parseInt(slider.value) + 1;
      if (v > 2028) v = 2019;
      slider.value = v;
      updateMap();
    }, 1500);
  } else {
    btn.textContent = '▶'; btn.classList.remove('active');
    clearInterval(playInterval);
  }
}

updateMap();
</script>
</body>
</html>"""

html = html.replace("__LR_DATA__", json.dumps(lr_map))
html = html.replace("__DIFF_DATA__", json.dumps(diff_map))

out_path = os.path.join(OUT_DIR, "generative_timelapse.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print("\nTimelapse saved to %s (%.1f KB)" % (out_path, os.path.getsize(out_path) / 1024))
