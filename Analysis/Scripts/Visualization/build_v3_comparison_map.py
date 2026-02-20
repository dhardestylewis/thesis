"""
Build interactive webmap: heatmap/bloom of predicted risk + actual protest dots on top.
Includes calibration stats per model per year and year slider.
"""
import csv, json, os, sys, random

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
random.seed(42)

SCORES_PATH = "Analysis/Results/Diffusion_v3/per_parcel_scores.csv"
OUT_PATH = "Analysis/Results/Diffusion_v3/v3_comparison_map.html"

# ---- Load ALL scores ----
print("Loading scores...")
year_data = {}
n_rows = 0
with open(SCORES_PATH, "r") as f:
    for row in csv.DictReader(f):
        year = int(row["year"])
        if year not in year_data:
            year_data[year] = []
        year_data[year].append({
            "pid": row["parcel_id"],
            "lr": float(row["lr_score"]),
            "diff": float(row["diff_score"]),
            "ens": float(row["ensemble_score"]),
            "actual": int(row["actual"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
        })
        n_rows += 1

print(f"Loaded {n_rows:,} rows across {len(year_data)} years")

# ---- Compute calibration stats ----
print("\nCalibration analysis:")
calibration = {}
thresholds = [0.2, 0.3, 0.5, 0.7, 0.9]

for year, rows in sorted(year_data.items()):
    calibration[str(year)] = {}
    for model_key, model_name in [("lr", "LogReg"), ("diff", "Diffusion"), ("ens", "Ensemble")]:
        cal = {}
        for thresh in thresholds:
            above = [r for r in rows if r[model_key] > thresh]
            n_above = len(above)
            n_protest = sum(1 for r in above if r["actual"] == 1)
            actual_rate = n_protest / n_above if n_above > 0 else 0
            cal[str(thresh)] = {
                "n": n_above,
                "n_protest": n_protest,
                "actual_rate": round(actual_rate, 4),
            }
        calibration[str(year)][model_key] = cal
        
        print(f"  {year} {model_name}:")
        for thresh in thresholds:
            c = cal[str(thresh)]
            if c["n"] > 0:
                print(f"    score>{thresh:.0%}: {c['n']:,} parcels, {c['n_protest']} protests ({c['actual_rate']:.1%})")

# ---- Sample for heatmap (keep more points for density) ----
print("\nSampling for heatmap...")
MAX_HEAT = 12000  # More points for a good heatmap

heat_data = {}  # [lat, lon, score] for heatmap
dot_data = {}   # actual protests only, with scores

for year, rows in year_data.items():
    positives = [r for r in rows if r["actual"] == 1]
    
    # For heatmap: sample from all parcels with any meaningful score
    scored = [r for r in rows if max(r["lr"], r["diff"], r["ens"]) > 0.05]
    if len(scored) > MAX_HEAT:
        # Keep all high-risk, sample the rest
        high = [r for r in scored if max(r["lr"], r["diff"], r["ens"]) > 0.3]
        rest = [r for r in scored if max(r["lr"], r["diff"], r["ens"]) <= 0.3]
        n_rest = min(len(rest), MAX_HEAT - len(high))
        scored = high + random.sample(rest, n_rest)
    
    heat_data[str(year)] = [
        [round(r["lat"], 5), round(r["lon"], 5),
         round(r["lr"], 4), round(r["diff"], 4), round(r["ens"], 4)]
        for r in scored
    ]
    
    dot_data[str(year)] = [
        [round(r["lat"], 5), round(r["lon"], 5),
         round(r["lr"], 4), round(r["diff"], 4), round(r["ens"], 4)]
        for r in positives
    ]
    
    print(f"  {year}: {len(heat_data[str(year)])} heat points, {len(dot_data[str(year)])} protest dots")

# Load metrics
METRICS_PATH = "Analysis/Results/Diffusion_v3/classification_metrics.json"
with open(METRICS_PATH) as f:
    metrics = json.load(f)

metrics_by_year = {}
for m in metrics:
    y = m["eval_year"]
    if y not in metrics_by_year:
        metrics_by_year[y] = {}
    metrics_by_year[y][m["model"]] = m

# ---- Build HTML ----
print("Building HTML...")

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Diffusion v3 — Protest Risk Heatmap</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Inter', -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; }}
  #map {{ position: absolute; top: 0; left: 0; right: 0; bottom: 0; z-index: 1; }}
  
  .controls {{
    position: absolute; top: 16px; right: 16px; z-index: 1000;
    background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(14px);
    border-radius: 14px; padding: 20px; width: 360px;
    border: 1px solid rgba(148, 163, 184, 0.12);
    box-shadow: 0 8px 40px rgba(0,0,0,0.6);
    max-height: calc(100vh - 32px); overflow-y: auto;
  }}
  .controls h2 {{ font-size: 15px; font-weight: 700; margin-bottom: 14px; color: #f8fafc; }}
  .controls h3 {{ 
    font-size: 11px; font-weight: 600; margin: 16px 0 6px; color: #94a3b8; 
    text-transform: uppercase; letter-spacing: 0.8px; 
  }}
  
  .year-slider {{ width: 100%; accent-color: #818cf8; height: 6px; }}
  .year-label {{ 
    text-align: center; font-size: 36px; font-weight: 900; 
    color: #818cf8; margin: 6px 0 14px;
    text-shadow: 0 0 20px rgba(129,140,248,0.3);
  }}
  
  .model-btns {{ display: flex; gap: 6px; margin-bottom: 12px; }}
  .model-btn {{
    flex: 1; padding: 10px 4px; border-radius: 10px; border: 1px solid rgba(148,163,184,0.15);
    background: rgba(30, 41, 59, 0.8); color: #94a3b8; cursor: pointer;
    font-size: 12px; font-weight: 600; text-align: center; transition: all 0.2s;
  }}
  .model-btn.active {{ 
    background: linear-gradient(135deg, #6366f1, #818cf8); 
    color: white; border-color: transparent; 
    box-shadow: 0 2px 12px rgba(99,102,241,0.3);
  }}
  .model-btn:hover:not(.active) {{ border-color: #6366f1; color: #c7d2fe; }}
  
  .legend {{ display: flex; gap: 10px; margin: 10px 0; align-items: center; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 11px; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
  
  .toggle-row {{ display: flex; gap: 14px; margin: 8px 0; }}
  .toggle {{ display: flex; align-items: center; gap: 6px; font-size: 12px; cursor: pointer; }}
  .toggle input {{ accent-color: #818cf8; }}
  
  table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 6px; }}
  th {{ text-align: left; padding: 4px 5px; color: #64748b; font-weight: 500; border-bottom: 1px solid rgba(148,163,184,0.08); }}
  td {{ padding: 4px 5px; font-variant-numeric: tabular-nums; }}
  .best {{ color: #34d399; font-weight: 700; }}
  .rate {{ font-weight: 700; }}
  .good {{ color: #34d399; }}
  .ok {{ color: #eab308; }}
  .bad {{ color: #f97316; }}
  
  .stat {{ font-size: 11px; color: #94a3b8; margin: 3px 0; }}
  .stat b {{ color: #e2e8f0; }}
  
  .heat-label {{
    display: flex; align-items: center; gap: 4px; margin: 6px 0;
    font-size: 11px; color: #94a3b8;
  }}
  .heat-gradient {{
    flex: 1; height: 8px; border-radius: 4px;
    background: linear-gradient(to right, transparent, #22c55e, #eab308, #f97316, #ef4444, #dc2626);
  }}
  
  .intensity-slider {{ width: 100%; accent-color: #f97316; }}
</style>
</head>
<body>
<div id="map"></div>
<div class="controls">
  <h2>🔬 Diffusion v3 — Protest Risk Heatmap</h2>
  
  <h3>Evaluation Year</h3>
  <input type="range" class="year-slider" id="yearSlider" min="2021" max="2024" value="2021" step="1">
  <div class="year-label" id="yearLabel">2021</div>
  
  <h3>Model View</h3>
  <div class="model-btns">
    <div class="model-btn active" data-model="diff" id="btnDiff">Diffusion</div>
    <div class="model-btn" data-model="lr" id="btnLR">LogReg</div>
    <div class="model-btn" data-model="ens" id="btnEns">Ensemble</div>
  </div>
  
  <h3>Heatmap Intensity</h3>
  <input type="range" class="intensity-slider" id="intensitySlider" min="5" max="40" value="18" step="1">
  <div class="heat-label">
    <span>Low</span>
    <div class="heat-gradient"></div>
    <span>High</span>
  </div>
  
  <div class="toggle-row">
    <label class="toggle"><input type="checkbox" id="showDots" checked> Actual protests ●</label>
    <label class="toggle"><input type="checkbox" id="showHeat" checked> Risk heatmap</label>
  </div>
  
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#fff; border: 3px solid #ef4444;"></div>Actual protest</div>
    <div class="legend-item"><div class="legend-dot" style="background: radial-gradient(#ef4444, #f97316, transparent);"></div>Predicted risk</div>
  </div>
  
  <h3>Performance</h3>
  <table id="metricsTable">
    <thead><tr><th>Metric</th><th>LR</th><th>Diff</th><th>Ens</th></tr></thead>
    <tbody id="metricsBody"></tbody>
  </table>
  
  <h3>Calibration — "Score > X% → actual protest rate"</h3>
  <table id="calTable">
    <thead><tr><th>Score ></th><th>Flagged</th><th>Protests</th><th>Actual %</th></tr></thead>
    <tbody id="calBody"></tbody>
  </table>
  
  <div id="stats" style="margin-top: 12px;"></div>
</div>

<script>
const HEAT = {json.dumps(heat_data)};
const DOTS = {json.dumps(dot_data)};
const METRICS = {json.dumps(metrics_by_year)};
const CALIBRATION = {json.dumps(calibration)};

const map = L.map('map', {{ zoomControl: false }}).setView([30.30, -97.74], 11);
L.control.zoom({{ position: 'bottomright' }}).addTo(map);

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}@2x.png', {{
  maxZoom: 19, attribution: '&copy; CARTO'
}}).addTo(map);

let currentYear = '2021';
let currentModel = 'diff';
let showDots = true;
let showHeat = true;
let heatLayer = null;
let dotMarkers = [];
let intensity = 18;

function getModelIdx(model) {{
  return model === 'lr' ? 2 : model === 'diff' ? 3 : 4;
}}

function updateMap() {{
  // Clear existing
  if (heatLayer) {{ map.removeLayer(heatLayer); heatLayer = null; }}
  dotMarkers.forEach(m => map.removeLayer(m));
  dotMarkers = [];
  
  const heatRows = HEAT[currentYear] || [];
  const dotRows = DOTS[currentYear] || [];
  const mIdx = getModelIdx(currentModel);
  
  // Build heatmap
  if (showHeat && heatRows.length > 0) {{
    const heatPoints = heatRows
      .filter(r => r[mIdx] > 0.05)
      .map(r => [r[0], r[1], r[mIdx]]);
    
    heatLayer = L.heatLayer(heatPoints, {{
      radius: intensity,
      blur: intensity * 0.8,
      maxZoom: 15,
      max: 1.0,
      minOpacity: 0.15,
      gradient: {{
        0.0: 'transparent',
        0.15: '#064e3b',
        0.3: '#22c55e',
        0.45: '#84cc16',
        0.6: '#eab308',
        0.75: '#f97316',
        0.9: '#ef4444',
        1.0: '#dc2626',
      }}
    }}).addTo(map);
  }}
  
  // Build protest dots on top
  if (showDots) {{
    dotRows.forEach(row => {{
      const [lat, lon, lr, diff, ens] = row;
      const score = row[mIdx];
      
      const marker = L.circleMarker([lat, lon], {{
        radius: 7,
        fillColor: '#ffffff',
        color: '#ef4444',
        weight: 3,
        opacity: 1,
        fillOpacity: 0.95,
      }});
      
      const pct = v => (v*100).toFixed(1) + '%';
      marker.bindTooltip(
        `<div style="min-width:180px">` +
        `<b style="font-size:13px;color:#ef4444;">🔴 ACTUAL PROTEST</b><br>` +
        `<hr style="border-color:rgba(255,255,255,0.1);margin:4px 0">` +
        `<table style="width:100%;font-size:11px">` +
        `<tr><td>LogReg:</td><td style="text-align:right"><b>${{pct(lr)}}</b></td></tr>` +
        `<tr><td>Diffusion:</td><td style="text-align:right"><b>${{pct(diff)}}</b></td></tr>` +
        `<tr><td>Ensemble:</td><td style="text-align:right"><b>${{pct(ens)}}</b></td></tr>` +
        `</table></div>`,
        {{ direction: 'top', className: 'dark-tooltip' }}
      );
      
      marker.addTo(map);
      dotMarkers.push(marker);
    }});
  }}
  
  // Stats
  const nProtest = dotRows.length;
  const nCaught = dotRows.filter(r => r[mIdx] > 0.3).length;
  const nHeat = heatRows.filter(r => r[mIdx] > 0.05).length;
  
  document.getElementById('stats').innerHTML = 
    `<div class="stat"><b>${{nProtest}}</b> actual protests | ` +
    `<b style="color:#34d399">${{nCaught}}</b> flagged (score>30%) | ` +
    `<b style="color:#f97316">${{nProtest - nCaught}}</b> missed</div>` +
    `<div class="stat">Detection rate: <b style="color:#818cf8">${{nProtest>0 ? ((nCaught/nProtest)*100).toFixed(1) : 0}}%</b> | ` +
    `Heat points: <b>${{nHeat.toLocaleString()}}</b></div>`;
  
  updateMetrics();
  updateCalibration();
}}

function updateMetrics() {{
  const ym = METRICS[currentYear];
  if (!ym) return;
  
  const lr = ym['LogReg'] || {{}};
  const diff = ym['Diffusion_v3'] || {{}};
  const ens = ym['Ensemble'] || {{}};
  
  const rows = [
    ['AUC-ROC', 'auc_roc', false],
    ['AUC-PR', 'auc_pr', false],
    ['Brier', 'brier_score', true],
    ['P@1x', 'precision@1x', false],
    ['R@1x', 'recall@1x', false],
    ['R@5x', 'recall@5x', false],
  ];
  
  let html = '';
  rows.forEach(([label, key, lowerBetter]) => {{
    const lv = lr[key], dv = diff[key], ev = ens[key];
    if (lv === undefined) return;
    const vals = [lv, dv, ev];
    const bestIdx = lowerBetter 
      ? vals.indexOf(Math.min(...vals))
      : vals.indexOf(Math.max(...vals));
    html += `<tr>
      <td>${{label}}</td>
      <td class="${{bestIdx===0?'best':''}}">${{lv?.toFixed(3)}}</td>
      <td class="${{bestIdx===1?'best':''}}">${{dv?.toFixed(3)}}</td>
      <td class="${{bestIdx===2?'best':''}}">${{ev?.toFixed(3)}}</td>
    </tr>`;
  }});
  document.getElementById('metricsBody').innerHTML = html;
}}

function updateCalibration() {{
  const cal = CALIBRATION[currentYear];
  if (!cal) return;
  const modelCal = cal[currentModel];
  if (!modelCal) return;
  
  const thresholds = ['0.2', '0.3', '0.5', '0.7', '0.9'];
  let html = '';
  thresholds.forEach(t => {{
    const c = modelCal[t];
    if (!c || c.n === 0) return;
    const rate = c.actual_rate;
    const rateClass = rate > 0.3 ? 'good' : rate > 0.1 ? 'ok' : 'bad';
    html += `<tr>
      <td>${{(parseFloat(t)*100).toFixed(0)}}%</td>
      <td>${{c.n.toLocaleString()}}</td>
      <td>${{c.n_protest}}</td>
      <td class="rate ${{rateClass}}">${{(rate*100).toFixed(1)}}%</td>
    </tr>`;
  }});
  document.getElementById('calBody').innerHTML = html;
}}

// Events
document.getElementById('yearSlider').addEventListener('input', e => {{
  currentYear = e.target.value;
  document.getElementById('yearLabel').textContent = currentYear;
  updateMap();
}});

document.getElementById('intensitySlider').addEventListener('input', e => {{
  intensity = parseInt(e.target.value);
  updateMap();
}});

document.querySelectorAll('.model-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentModel = btn.dataset.model;
    updateMap();
  }});
}});

document.getElementById('showDots').addEventListener('change', e => {{
  showDots = e.target.checked;
  updateMap();
}});

document.getElementById('showHeat').addEventListener('change', e => {{
  showHeat = e.target.checked;
  updateMap();
}});

const style = document.createElement('style');
style.textContent = `.dark-tooltip {{ background: rgba(15,23,42,0.95) !important; color: #e2e8f0 !important; border: 1px solid rgba(148,163,184,0.2) !important; border-radius: 10px !important; padding: 10px 14px !important; font-size: 12px !important; line-height: 1.6 !important; box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important; }}`;
document.head.appendChild(style);

updateMap();
</script>
</body>
</html>
"""

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
print(f"\nWrote {OUT_PATH} ({size_mb:.1f}MB)")
print("Done!")
