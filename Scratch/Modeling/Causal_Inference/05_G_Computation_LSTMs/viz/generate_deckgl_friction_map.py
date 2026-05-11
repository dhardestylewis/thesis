"""
generate_deckgl_friction_map.py

Generates a standalone Deck.gl HTML map of the Austin Friction Map
using ACTUAL historic zoning case parcel boundaries (GeoJSON).

Encoding:
  - Polygon Extrusion Height: Expected Bureaucratic Delay CATE (days)
  - Polygon Color: Withdrawal Probability (dark purple = safe, bright red = lethal)
  - Base Map: CARTO Dark Matter
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from econml.dml import CausalForestDML
from sklearn.model_selection import StratifiedKFold
import geopandas as gpd

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    model_y_multi = CatBoostRegressor(iterations=200, depth=5, loss_function='MultiRMSE', verbose=0)
    model_t = CatBoostClassifier(iterations=200, depth=5, verbose=0)
    model_y_bin = CatBoostRegressor(iterations=200, depth=5, verbose=0)
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.multioutput import MultiOutputRegressor
    model_y_multi = MultiOutputRegressor(GradientBoostingRegressor(max_depth=4, n_estimators=100))
    model_t = GradientBoostingClassifier(max_depth=4, n_estimators=100)
    model_y_bin = GradientBoostingRegressor(max_depth=4, n_estimators=100)

ROOT = Path(r"c:\Users\dhl\data\Thesis\thesis")
DOSE_THRESHOLD = 0.20

print("Loading data...", flush=True)
panel_path = ROOT / "Data/Panel/biweekly_panel_patched.csv"
if not panel_path.exists():
    panel_path = ROOT / "Data/Panel/biweekly_panel.csv"
df = pd.read_csv(panel_path, low_memory=False)

zoning_df = pd.read_csv(ROOT / "Data/Zoning_Cases/Processed_Data/CSV/zoning_land_use_merged_data.csv", low_memory=False)
zoning_df['start'] = pd.to_datetime(zoning_df['application_start_date'], errors='coerce')
zoning_df['end'] = pd.to_datetime(zoning_df['status_date'], errors='coerce')
zoning_df['days_to_resolution'] = (zoning_df['end'] - zoning_df['start']).dt.days.clip(0, 3650)
zoning_dates = zoning_df[['case_number', 'days_to_resolution']].drop_duplicates('case_number')

status_df = pd.read_csv(ROOT / "Data/Zoning_Cases/Processed_Data/CSV/zoning_case_statuses.csv", low_memory=False)

print("Collapsing panel...", flush=True)
cs = df.groupby('case_number').agg({
    'cumulative_unofficial_protest_intensity': 'max',
    'Delta_Approved_Height': 'last',
    'Delta_Requested_Height': 'last',
    'latitude': 'first',
    'longitude': 'first',
    'median_household_income': 'first',
    'race_white': 'first',
    'renter_share': 'first',
    'year': 'first',
    'cumulative_min_signer_dist': 'max',
    'cumulative_signers_outside_200ft': 'max',
    'cumulative_protester_embed_dim1': 'max',
    'cumulative_protester_embed_dim2': 'max',
    'cumulative_petition_attempted': 'max',
    'cumulative_mobilization_failure': 'max'
}).reset_index()

mask_withdrawn = cs['Delta_Requested_Height'].notna() & cs['Delta_Approved_Height'].isna()
cs.loc[mask_withdrawn, 'Delta_Approved_Height'] = 0

cs = pd.merge(cs, zoning_dates, on='case_number', how='left')
cs = pd.merge(cs, status_df[['case_number', 'detailed_status']], on='case_number', how='left')

def fraction_01(s):
    x = pd.to_numeric(s, errors='coerce').fillna(0.0)
    non_zero = x[x > 0]
    if len(non_zero) > 0 and non_zero.quantile(0.99) > 1.0:
        x = x / 100.0
    return x.clip(0.0, 1.0)

cs['petition_dose'] = fraction_01(cs['cumulative_unofficial_protest_intensity'])
cs['Height_Attrition'] = cs['Delta_Requested_Height'] - cs['Delta_Approved_Height']
cs['Withdrawal_Binary'] = (cs['detailed_status'] == 'Withdrawn').astype(float)

for c in ['median_household_income', 'race_white', 'renter_share']:
    cs[c] = cs[c].fillna(cs[c].median())
for c in ['cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
          'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
          'cumulative_petition_attempted', 'cumulative_mobilization_failure']:
    cs[c] = cs[c].fillna(0.0)

confounders = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'renter_share',
    'cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
    'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
    'cumulative_petition_attempted', 'cumulative_mobilization_failure'
]

cs = cs.dropna(subset=confounders + ['Delta_Approved_Height', 'Height_Attrition', 'petition_dose', 'days_to_resolution', 'year', 'latitude', 'longitude'])

print(f"Running Causal Forest at dose threshold = {DOSE_THRESHOLD}...", flush=True)
X = cs[confounders].values
D = cs['petition_dose'].values
D_bin = (D >= DOSE_THRESHOLD).astype(float)

surv_mask = ~cs['detailed_status'].isin(['Withdrawn', 'Denied', 'Expired', 'VOID'])
cs_surv = cs[surv_mask]
X_surv = cs_surv[confounders].values
Y_surv_joint = cs_surv[['Height_Attrition', 'days_to_resolution']].values
D_bin_surv = (cs_surv['petition_dose'] >= DOSE_THRESHOLD).astype(float).values
Y_withd = cs['Withdrawal_Binary'].values

cf_joint = CausalForestDML(
    model_y=model_y_multi, model_t=model_t,
    discrete_treatment=True, n_estimators=100,
    cv=StratifiedKFold(n_splits=2), random_state=42
)
cf_withd = CausalForestDML(
    model_y=model_y_bin, model_t=model_t,
    discrete_treatment=True, n_estimators=100,
    cv=StratifiedKFold(n_splits=2), random_state=42
)

print("  Fitting Joint Forest (Height + Delay)...", flush=True)
cf_joint.fit(Y_surv_joint, D_bin_surv, X=X_surv)
cate_multi = cf_joint.effect(X)
cs['cate_height'] = np.clip(cate_multi[:, 0], -500, 1500)
cs['cate_delay']  = np.clip(cate_multi[:, 1], -365, 3650)

print("  Fitting Withdrawal Forest...", flush=True)
cf_withd.fit(Y_withd, D_bin, X=X)
cs['cate_withd'] = np.clip(cf_withd.effect(X), -1.0, 1.0)

# ── Color & Styling Logic ───────────────────────────────────────────────────
GRAVEYARD_THRESHOLD = 0.10
HEIGHT_SCALE = 0.5
MIN_HEIGHT = 10

colors = []
for i, row in cs.iterrows():
    delay = row['cate_delay']
    withd = row['cate_withd']
    is_dead = withd > GRAVEYARD_THRESHOLD
    
    if is_dead:
        color = [80, 80, 80, 180]
    else:
        norm_d = max(0.0, min(1.0, (delay - (-365)) / (3650 - (-365))))
        if norm_d < 0.25:
            r = int(20 + norm_d * 4 * (100 - 20))
            g = int(11 + norm_d * 4 * (24 - 11))
            b = int(121 + norm_d * 4 * (221 - 121))
        elif norm_d < 0.5:
            t = (norm_d - 0.25) * 4
            r = int(100 + t * (188 - 100))
            g = int(24 + t * (55 - 24))
            b = int(221 + t * (84 - 221))
        elif norm_d < 0.75:
            t = (norm_d - 0.5) * 4
            r = int(188 + t * (253 - 188))
            g = int(55 + t * (141 - 55))
            b = int(84 + t * (33 - 84))
        else:
            t = (norm_d - 0.75) * 4
            r = int(253 + t * (252 - 253))
            g = int(141 + t * (255 - 141))
            b = int(33 + t * (164 - 33))
        color = [min(255, r), min(255, g), min(255, b), 220]
    colors.append(color)

cs['color'] = colors
cs['is_dead'] = cs['cate_withd'] > GRAVEYARD_THRESHOLD

# ── Polygon Mapping ─────────────────────────────────────────────────────────
print("Loading real parcel geometries...", flush=True)
gdf = gpd.read_file(ROOT / "Data/Zoning_Cases/zoning_cases_master_polygons.geojson")

# Austin data is usually in state plane, so ensure it's unprojected to WGS84 for Deck.gl
if gdf.crs and gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

cs['year'] = cs['year'].astype(str)

cs_mapped = gdf[['case_number', 'geometry']].merge(
    cs[['case_number', 'petition_dose', 'cate_delay', 'cate_height', 'cate_withd', 'is_dead', 'color', 'year']],
    on='case_number',
    how='inner'
)
print(f"  Joined {len(cs_mapped)} cases to exact parcel geometries.", flush=True)

data_json = cs_mapped.to_json()

# ── HTML Template ─────────────────────────────────────────────────────────────
print("Rendering HTML...", flush=True)
html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Austin Friction Map — Predictive Causal ML Surface</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet" />
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #0d1117; font-family: 'Inter', 'Segoe UI', sans-serif; color: #e0e0e0; overflow: hidden; }}
    #map {{ width: 100vw; height: 100vh; }}
    #panel {{
      position: fixed; top: 20px; left: 20px; width: 300px;
      background: rgba(13, 17, 23, 0.92);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px; padding: 20px; z-index: 1000;
      backdrop-filter: blur(12px);
    }}
    #panel h1 {{ font-size: 14px; font-weight: 700; letter-spacing: 0.04em; color: #f0f0f0; margin-bottom: 2px; }}
    #panel .subtitle {{ font-size: 11px; color: #888; margin-bottom: 18px; }}
    .slider-row {{ margin-bottom: 14px; }}
    .slider-label {{ display: flex; justify-content: space-between; align-items: baseline; font-size: 11px; color: #aaa; margin-bottom: 5px; }}
    .slider-label span {{ font-size: 13px; font-weight: 700; color: #fff; }}
    input[type=range] {{ width: 100%; accent-color: #f97316; cursor: pointer; }}
    .metrics {{ margin-top: 16px; border-top: 1px solid rgba(255,255,255,0.07); padding-top: 14px; }}
    .metric {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
    .metric-label {{ font-size: 11px; color: #888; }}
    .metric-value {{ font-size: 14px; font-weight: 700; color: #f97316; }}
    .legend {{ margin-top: 16px; border-top: 1px solid rgba(255,255,255,0.07); padding-top: 14px; font-size: 11px; color: #888; }}
    .legend-title {{ color: #aaa; font-weight: 600; margin-bottom: 8px; }}
    .legend-bar {{ height: 10px; border-radius: 5px; margin: 6px 0 2px 0;
      background: linear-gradient(to right, #14097899, #bc3768ff, #fd8d21ff, #fcff37ff); }}
    .legend-bar-labels {{ display: flex; justify-content: space-between; font-size: 10px; color: #666; }}
    .graveyard-swatch {{ display: inline-block; width: 12px; height: 12px;
      background: rgba(80,80,80,0.9); border-radius: 2px; margin-right: 6px; vertical-align: middle; }}
    #tooltip {{
      position: fixed; pointer-events: none; display: none;
      background: rgba(13,17,23,0.95); border: 1px solid rgba(255,255,255,0.1);
      border-radius: 8px; padding: 12px 14px; font-size: 12px;
      line-height: 1.6; max-width: 220px; z-index: 2000;
    }}
    #tooltip .case {{ font-weight: 700; color: #f0f0f0; margin-bottom: 6px; font-size: 13px; }}
    #tooltip .killed {{ color: #ef4444; font-weight: 700; }}
    .credits {{ position: fixed; bottom: 8px; right: 10px; font-size: 9px; color: #666; z-index: 999; }}
  </style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <h1>Austin Friction Map</h1>
  <div class="subtitle">Predictive Causal Machine Learning Surface</div>

  <div class="slider-row">
    <div class="slider-label">Proposed Additional Height (ft) <span id="height-val">29</span></div>
    <input id="height-slider" type="range" min="5" max="120" value="29" step="1">
    <div style="font-size:10px;color:#555;margin-top:3px;">Simulates the requested upzone scale.</div>
  </div>

  <div class="metrics">
    <div class="metric"><span class="metric-label">Expected Delay Toll</span><span class="metric-value" id="m-delay">-- Days</span></div>
    <div class="metric"><span class="metric-label">Withdrawal Risk</span><span class="metric-value" id="m-withd">-- %</span></div>
    <div class="metric"><span class="metric-label">Height Attrition</span><span class="metric-value" id="m-height">-- ft</span></div>
  </div>

  <div class="legend">
    <div class="legend-title">Map Legend</div>
    <div>Extrusion Height: Net Approved Height (ft)</div>
    <div style="margin-top:8px;">Color: Expected Delay (Days)</div>
    <div class="legend-bar"></div>
    <div class="legend-bar-labels"><span>Fast</span><span>Years Delayed</span></div>
    <div style="margin-top:8px;"><span class="graveyard-swatch"></span> Killed (Withdrawn)</div>
  </div>
</div>

<div id="tooltip"></div>
<div class="credits">© CARTO, © OpenStreetMap contributors</div>

<script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
<script src="https://unpkg.com/deck.gl@^8.9/dist.min.js"></script>
<script>
const RAW_DATA = {data_json};

let currentDose = 0.20;
let currentHeight = 29;

function computeStats() {{
  const features = RAW_DATA.features;
  if (features.length === 0) return {{ delay: null, withd: null, height: null }};
  const avgDelay  = features.reduce((s,f) => s + f.properties.cate_delay,  0) / features.length;
  const avgWithd  = features.reduce((s,f) => s + f.properties.cate_withd,  0) / features.length;
  const avgHeight = features.reduce((s,f) => s + f.properties.cate_height, 0) / features.length;
  return {{ delay: avgDelay, withd: avgWithd, height: avgHeight }};
}}

function getColor(f) {{
  if (f.properties.cate_withd > 0.10) return [80, 80, 80, 180]; // Graveyard
  
  // Map delay to an Inferno color gradient (0 to 1800 days)
  const delay = f.properties.cate_delay;
  const norm_d = Math.max(0, Math.min(1, delay / 1800));
  
  let r, g, b;
  if (norm_d < 0.25) {{
      const t = norm_d * 4;
      r = 20 + t * (100 - 20); g = 11 + t * (24 - 11); b = 121 + t * (221 - 121);
  }} else if (norm_d < 0.5) {{
      const t = (norm_d - 0.25) * 4;
      r = 100 + t * (188 - 100); g = 24 + t * (55 - 24); b = 221 + t * (84 - 221);
  }} else if (norm_d < 0.75) {{
      const t = (norm_d - 0.5) * 4;
      r = 188 + t * (253 - 188); g = 55 + t * (141 - 55); b = 84 + t * (33 - 84);
  }} else {{
      const t = (norm_d - 0.75) * 4;
      r = 253 + t * (252 - 253); g = 141 + t * (255 - 141); b = 33 + t * (164 - 33);
  }}
  return [r, g, b, 220];
}}

function getElevation(f) {{
  if (f.properties.cate_withd > 0.10) return 5; // Graveyard columns are flat
  
  // Extrude by the requested height MINUS the causal height attrition
  // cate_height is how much height was LOST (so we subtract it)
  // Convert feet to meters (* 0.3048)
  const net_height_ft = Math.max(0, currentHeight - f.properties.cate_height);
  return Math.max(5, net_height_ft * 0.3048);
}}

const tooltip = document.getElementById('tooltip');

const map = new maplibregl.Map({{
  container: 'map',
  style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
  center: [-97.743, 30.275],
  zoom: 11.5,
  pitch: 55,
  bearing: -15,
  antialias: true
}});

const overlay = new deck.MapboxOverlay({{
  interleaved: false,
  layers: [],
  onHover: (info) => {{
    if (info.object) {{
      const d = info.object.properties;
      const isKilled = d.cate_withd > 0.10;
      const netHeight = Math.max(0, currentHeight - d.cate_height);
      tooltip.style.display = 'block';
      tooltip.style.left = (info.x + 14) + 'px';
      tooltip.style.top  = (info.y - 10) + 'px';
      tooltip.innerHTML = `
        <div class="case">Case: ${{d.case_number}}</div>
        <div>Year: ${{Math.round(d.year)}}</div>
        <div>Petition Dose: ${{(d.petition_dose * 100).toFixed(1)}}%</div>
        ${{isKilled
          ? '<div class="killed">&#x26B0; Killed (Withdrawn)</div>'
          : `<div>Net Approved Height: ${{netHeight.toFixed(1)}} ft</div>
             <div>Height Attrition: -${{d.cate_height.toFixed(1)}} ft</div>
             <div>Expected Delay: ${{d.cate_delay.toFixed(0)}} days</div>
             <div>Withd Risk: ${{(d.cate_withd * 100).toFixed(1)}}%</div>`
        }}
      `;
    }} else {{
      tooltip.style.display = 'none';
    }}
  }}
}});

map.addControl(overlay);

function render() {{
  overlay.setProps({{
    layers: [
      new deck.GeoJsonLayer({{
        id: 'friction-polygons',
        data: RAW_DATA,
        stroked: false,
        filled: true,
        extruded: true,
        getElevation: f => getElevation(f),
        getFillColor: f => getColor(f),
        updateTriggers: {{
          getElevation: [currentDose, currentHeight],
          getFillColor: [currentDose]
        }},
        transitions: {{ getElevation: 500, getFillColor: 300 }}
      }})
    ]
  }});

  const stats = computeStats(currentDose);
  document.getElementById('m-delay').textContent  = stats.delay  !== null ? Math.round(stats.delay) + ' Days' : '-- Days';
  document.getElementById('m-withd').textContent  = stats.withd  !== null ? (stats.withd * 100).toFixed(1) + '%' : '-- %';
  document.getElementById('m-height').textContent = stats.height !== null ? stats.height.toFixed(1) + ' ft' : '-- ft';
}}

map.on('load', render);

document.getElementById('height-slider').addEventListener('input', e => {{
  currentHeight = parseInt(e.target.value);
  document.getElementById('height-val').textContent = e.target.value;
  render();
}});
</script>
</body>
</html>"""

out_path = r'C:\Users\dhl\.gemini\antigravity\brain\1632e32a-ef31-4422-854b-ea7296224fe1\deckgl_friction_map.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\nSaved Deck.gl GeoJSON Friction Map to:\n  {out_path}")
