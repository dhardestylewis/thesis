"""
Diffusion vs LogReg Comparison Heatmap
=======================================
Generates a side-by-side Leaflet.js heatmap comparing LogReg and Diffusion v2
risk scores from per_parcel_scores.csv.

Features:
  - Year slider (2021-2024)
  - Model toggle (LogReg / Diffusion / Residual)
  - Color-coded risk tiers
  - Score statistics overlay
"""
import csv, json, sys, os
from collections import defaultdict
from datetime import datetime

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

SCORES_PATH = "Analysis/Results/Diffusion_v2/per_parcel_scores.csv"
OUT_PATH = "Analysis/Results/Diffusion_v2/diffusion_comparison_map.html"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ---- Load scores ----
log("Loading scores...")
data_by_year = defaultdict(list)
with open(SCORES_PATH, "r") as f:
    for row in csv.DictReader(f):
        year = int(row["year"])
        data_by_year[year].append({
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "lr": float(row["lr_score"]),
            "diff": float(row["diff_score"]),
            "actual": int(row["actual"]),
        })

for y in sorted(data_by_year):
    n = len(data_by_year[y])
    n_pos = sum(1 for d in data_by_year[y] if d["actual"] == 1)
    log(f"  Year {y}: {n:,} parcels, {n_pos} positive")

# ---- Subsample for rendering performance ----
import numpy as np
np.random.seed(42)

MAX_POINTS = 15000  # per year for rendering speed
sampled_data = {}
for year in sorted(data_by_year):
    rows = data_by_year[year]
    if len(rows) > MAX_POINTS:
        # Keep all positives, sample negatives
        pos = [r for r in rows if r["actual"] == 1]
        neg = [r for r in rows if r["actual"] == 0]
        n_neg = min(len(neg), MAX_POINTS - len(pos))
        idx = np.random.choice(len(neg), n_neg, replace=False)
        sampled = pos + [neg[i] for i in idx]
    else:
        sampled = rows
    sampled_data[year] = sampled
    log(f"  Sampled {year}: {len(sampled):,} parcels")

# ---- Build HTML ----
log("Building HTML...")

# Serialize data as JSON for JavaScript
js_data = {}
for year, rows in sampled_data.items():
    js_data[str(year)] = [[r["lat"], r["lon"], r["lr"], r["diff"], r["actual"]] for r in rows]

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>LogReg vs Diffusion v2 — Protest Risk Comparison</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0a0a0a; color: #e0e0e0; }}
  #map {{ width: 100%; height: 100vh; }}
  
  .control-panel {{
    position: fixed; top: 16px; left: 16px; z-index: 1000;
    background: rgba(15, 15, 25, 0.95); backdrop-filter: blur(12px);
    border: 1px solid rgba(100, 100, 255, 0.2); border-radius: 12px;
    padding: 16px; min-width: 300px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
  }}
  
  .control-panel h2 {{
    font-size: 14px; font-weight: 600; color: #8888ff;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 12px;
  }}
  
  .control-group {{ margin-bottom: 12px; }}
  .control-group label {{ display: block; font-size: 11px; color: #888; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
  
  .btn-group {{ display: flex; gap: 4px; }}
  .btn {{
    flex: 1; padding: 8px 12px; border: 1px solid rgba(100, 100, 255, 0.3);
    background: rgba(30, 30, 50, 0.8); color: #aaa; font-size: 12px;
    border-radius: 6px; cursor: pointer; transition: all 0.2s;
    text-align: center;
  }}
  .btn:hover {{ background: rgba(50, 50, 80, 0.8); color: #ccc; }}
  .btn.active {{ background: rgba(80, 80, 200, 0.4); color: #fff; border-color: #6666ff; }}
  
  .stats-box {{
    background: rgba(20, 20, 40, 0.8); border-radius: 8px;
    padding: 10px; margin-top: 8px; font-size: 11px;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    line-height: 1.5;
  }}
  .stats-box .metric {{ color: #aaa; }}
  .stats-box .value {{ color: #fff; font-weight: 600; }}
  .stats-box .good {{ color: #44cc88; }}
  .stats-box .bad {{ color: #ff6666; }}
  .stats-box .neutral {{ color: #8888ff; }}
  
  .legend {{
    position: fixed; bottom: 24px; right: 16px; z-index: 1000;
    background: rgba(15, 15, 25, 0.95); backdrop-filter: blur(12px);
    border: 1px solid rgba(100, 100, 255, 0.2); border-radius: 12px;
    padding: 12px 16px;
  }}
  .legend-title {{ font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase; }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; font-size: 11px; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
  
  input[type="range"] {{
    width: 100%; height: 6px; -webkit-appearance: none;
    background: rgba(100, 100, 255, 0.2); border-radius: 3px; outline: none;
  }}
  input[type="range"]::-webkit-slider-thumb {{
    -webkit-appearance: none; width: 16px; height: 16px;
    background: #6666ff; border-radius: 50%; cursor: pointer;
  }}
  .year-display {{ font-size: 24px; font-weight: 700; color: #fff; text-align: center; }}
</style>
</head>
<body>

<div id="map"></div>

<div class="control-panel">
  <h2>🏠 Protest Risk Comparison</h2>
  
  <div class="control-group">
    <label>Eval Year</label>
    <div class="year-display" id="yearDisplay">2021</div>
    <input type="range" id="yearSlider" min="2021" max="2024" value="2021" step="1">
  </div>
  
  <div class="control-group">
    <label>Model View</label>
    <div class="btn-group">
      <div class="btn active" data-mode="lr" onclick="setMode('lr')">LogReg</div>
      <div class="btn" data-mode="diff" onclick="setMode('diff')">Diffusion</div>
      <div class="btn" data-mode="residual" onclick="setMode('residual')">Residual</div>
      <div class="btn" data-mode="actual" onclick="setMode('actual')">Actual</div>
    </div>
  </div>
  
  <div class="stats-box" id="statsBox">
    Loading...
  </div>
</div>

<div class="legend" id="legend">
  <div class="legend-title" id="legendTitle">Risk Score</div>
  <div class="legend-item"><div class="legend-dot" style="background:#1a1a2e"></div> 0.0 — Very Low</div>
  <div class="legend-item"><div class="legend-dot" style="background:#2d4a22"></div> 0.2 — Low</div>
  <div class="legend-item"><div class="legend-dot" style="background:#8a8a00"></div> 0.4 — Moderate</div>
  <div class="legend-item"><div class="legend-dot" style="background:#cc6600"></div> 0.6 — High</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ff2200"></div> 0.8 — Very High</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ff00ff"></div> 1.0 — Extreme</div>
</div>

<script>
const DATA = {json.dumps(js_data)};

// Map init
const map = L.map('map', {{
  center: [30.27, -97.74],
  zoom: 11,
  zoomControl: false,
}});

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; CartoDB',
  subdomains: 'abcd',
  maxZoom: 19,
}}).addTo(map);

L.control.zoom({{ position: 'bottomleft' }}).addTo(map);

let currentYear = 2021;
let currentMode = 'lr';
let markers = [];

function scoreToColor(score) {{
  if (score < 0) score = 0;
  if (score > 1) score = 1;
  
  // Blue → Green → Yellow → Orange → Red → Magenta
  const r = Math.round(score < 0.5 ? score * 2 * 200 : 200 + (score - 0.5) * 2 * 55);
  const g = Math.round(score < 0.3 ? 30 + score * 500 : score < 0.6 ? 180 - (score - 0.3) * 400 : 0);
  const b = Math.round(score < 0.2 ? 50 - score * 200 : score > 0.8 ? (score - 0.8) * 5 * 255 : 0);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function residualToColor(resid) {{
  // -1 (blue, model under-predicted) → 0 (gray) → +1 (red, model over-predicted)
  const abs_r = Math.min(Math.abs(resid), 1);
  if (resid > 0) {{
    return `rgb(${{Math.round(128 + 127 * abs_r)}}, ${{Math.round(128 * (1 - abs_r))}}, ${{Math.round(128 * (1 - abs_r))}})`;
  }} else {{
    return `rgb(${{Math.round(128 * (1 - abs_r))}}, ${{Math.round(128 * (1 - abs_r))}}, ${{Math.round(128 + 127 * abs_r)}})`;
  }}
}}

function renderPoints() {{
  // Clear existing
  markers.forEach(m => map.removeLayer(m));
  markers = [];
  
  const yearData = DATA[currentYear.toString()];
  if (!yearData) return;
  
  let n_pos = 0, n_total = yearData.length;
  let score_sum = 0, score_values = [];
  
  yearData.forEach(d => {{
    const [lat, lon, lr, diff, actual] = d;
    let score, color, opacity;
    
    if (currentMode === 'lr') {{
      score = lr;
      color = scoreToColor(score);
      opacity = 0.3 + score * 0.6;
    }} else if (currentMode === 'diff') {{
      score = diff;
      color = scoreToColor(score);
      opacity = 0.3 + score * 0.6;
    }} else if (currentMode === 'residual') {{
      const lr_resid = lr - actual;
      const diff_resid = diff - actual;
      score = diff_resid - lr_resid;  // where diffusion disagrees with LR
      color = residualToColor(score);
      opacity = 0.3 + Math.abs(score) * 0.6;
    }} else if (currentMode === 'actual') {{
      score = actual;
      color = actual === 1 ? '#ff2200' : '#1a1a2e';
      opacity = actual === 1 ? 0.9 : 0.15;
    }}
    
    score_values.push(score);
    if (actual === 1) n_pos++;
    
    const radius = currentMode === 'actual' ? (actual === 1 ? 5 : 1.5) : 2 + score * 3;
    
    const circle = L.circleMarker([lat, lon], {{
      radius: radius,
      fillColor: color,
      fillOpacity: opacity,
      stroke: actual === 1,
      color: '#ffffff',
      weight: actual === 1 ? 1 : 0,
    }}).addTo(map);
    
    circle.bindTooltip(
      `LR: ${{lr.toFixed(3)}} | Diff: ${{diff.toFixed(3)}} | Actual: ${{actual}}`,
      {{ direction: 'top', offset: [0, -8] }}
    );
    
    markers.push(circle);
  }});
  
  // Update stats
  const mean_score = score_values.reduce((a, b) => a + b, 0) / score_values.length;
  const std_score = Math.sqrt(score_values.reduce((a, b) => a + (b - mean_score) ** 2, 0) / score_values.length);
  
  // Get metrics for this year from both models
  const lr_scores = yearData.map(d => d[2]);
  const diff_scores = yearData.map(d => d[3]);
  const actuals = yearData.map(d => d[4]);
  
  const lr_mean_pos = lr_scores.filter((_, i) => actuals[i] === 1).reduce((a, b) => a + b, 0) / n_pos;
  const lr_mean_neg = lr_scores.filter((_, i) => actuals[i] === 0).reduce((a, b) => a + b, 0) / (n_total - n_pos);
  const diff_mean_pos = diff_scores.filter((_, i) => actuals[i] === 1).reduce((a, b) => a + b, 0) / n_pos;
  const diff_mean_neg = diff_scores.filter((_, i) => actuals[i] === 0).reduce((a, b) => a + b, 0) / (n_total - n_pos);
  
  document.getElementById('statsBox').innerHTML = `
    <div><span class="metric">Parcels:</span> <span class="value">${{n_total.toLocaleString()}}</span> (<span class="bad">${{n_pos}} protests</span>)</div>
    <div><span class="metric">Prevalence:</span> <span class="value">${{(n_pos / n_total * 100).toFixed(2)}}%</span></div>
    <hr style="border-color: rgba(100,100,255,0.2); margin: 6px 0;">
    <div><span class="metric">LogReg:</span> <span class="good">pos=${{lr_mean_pos.toFixed(3)}}</span> / <span class="neutral">neg=${{lr_mean_neg.toFixed(3)}}</span> / <span class="value">gap=${{(lr_mean_pos - lr_mean_neg).toFixed(3)}}</span></div>
    <div><span class="metric">Diffusion:</span> <span class="good">pos=${{diff_mean_pos.toFixed(3)}}</span> / <span class="neutral">neg=${{diff_mean_neg.toFixed(3)}}</span> / <span class="value">gap=${{(diff_mean_pos - diff_mean_neg).toFixed(3)}}</span></div>
    <hr style="border-color: rgba(100,100,255,0.2); margin: 6px 0;">
    <div><span class="metric">Current view:</span> <span class="value">${{currentMode}}</span></div>
    <div><span class="metric">Score mean:</span> <span class="value">${{mean_score.toFixed(4)}}</span> ± ${{std_score.toFixed(4)}}</div>
  `;
  
  // Update legend
  if (currentMode === 'residual') {{
    document.getElementById('legendTitle').textContent = 'Model Disagreement';
  }} else if (currentMode === 'actual') {{
    document.getElementById('legendTitle').textContent = 'Actual Protests';
  }} else {{
    document.getElementById('legendTitle').textContent = currentMode === 'lr' ? 'LogReg Score' : 'Diffusion Score';
  }}
}}

function setMode(mode) {{
  currentMode = mode;
  document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`[data-mode="${{mode}}"]`).classList.add('active');
  renderPoints();
}}

document.getElementById('yearSlider').addEventListener('input', function() {{
  currentYear = parseInt(this.value);
  document.getElementById('yearDisplay').textContent = currentYear;
  renderPoints();
}});

// Initial render
renderPoints();
</script>
</body>
</html>"""

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

file_size = os.path.getsize(OUT_PATH)
log(f"Wrote {OUT_PATH} ({file_size / 1024 / 1024:.1f}MB)")
log("Done!")
