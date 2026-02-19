"""
Build Interactive Benchmark Dashboard (v2)
===========================================
Combines:
  1. Random-split ROC/PR/calibration/histogram from CVAE + Diffusion CSVs
  2. Temporal expanding-window metrics from generative_backtest_results.csv
All in one interactive HTML page with tabs.
"""
import csv, json, os, math
import numpy as np

RESULTS_DIR = "Analysis/Results/Backtests"
CVAE_CSV = os.path.join(RESULTS_DIR, "cvae_benchmark_results.csv")
DIFF_CSV = os.path.join(RESULTS_DIR, "diffusion_benchmark_results.csv")
TEMPORAL_CSV = os.path.join(RESULTS_DIR, "generative_backtest_results.csv")
OUT_HTML = os.path.join(RESULTS_DIR, "benchmark_dashboard.html")

# ---- Helpers ----

def load_csv(path, prob_cols):
    y_true, probs = [], {c: [] for c in prob_cols}
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            y_true.append(int(row["y_true"]))
            for c in prob_cols:
                probs[c].append(float(row[c]))
    return np.array(y_true), {c: np.array(v) for c, v in probs.items()}


def roc_curve(y_true, y_prob, n_points=200):
    thresholds = np.linspace(1, 0, n_points)
    points = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        points.append({"x": round(fpr, 4), "y": round(tpr, 4)})
    auc = sum((points[i]["x"] - points[i-1]["x"]) * (points[i]["y"] + points[i-1]["y"]) / 2 for i in range(1, len(points)))
    return points, round(abs(auc), 4)


def pr_curve(y_true, y_prob, n_points=200):
    thresholds = np.linspace(1, 0, n_points)
    points = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        points.append({"x": round(rec, 4), "y": round(prec, 4)})
    auc = sum((points[i]["x"] - points[i-1]["x"]) * (points[i]["y"] + points[i-1]["y"]) / 2 for i in range(1, len(points)))
    return points, round(abs(auc), 4)


def calibration_curve(y_true, y_prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    points, ece, n = [], 0, len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= edges[i]) & (y_prob < edges[i+1])
        if mask.sum() == 0: continue
        mp, mt = float(y_prob[mask].mean()), float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(mt - mp)
        points.append({"x": round(mp, 4), "y": round(mt, 4), "n": int(mask.sum())})
    return points, round(ece, 4)


def brier_score(y_true, y_prob):
    return round(float(np.mean((y_prob - y_true) ** 2)), 5)


def score_histogram(y_prob, n_bins=50):
    counts, edges = np.histogram(y_prob, bins=n_bins, range=(0, 1))
    return [{"x": round((edges[i] + edges[i+1]) / 2, 3), "y": int(counts[i])} for i in range(len(counts))]


# ---- Load Random-Split Data ----
print("Loading random-split benchmark results...")
cvae_yt, cvae_p = load_csv(CVAE_CSV, ["y_prob"])
diff_yt, diff_p = load_csv(DIFF_CSV, ["y_prob_base", "y_prob_aug"])

models = {}
for name, yt, yp, color in [
    ("LogReg Baseline", diff_yt, diff_p["y_prob_base"], "#3b82f6"),
    ("CVAE", cvae_yt, cvae_p["y_prob"], "#8b5cf6"),
    ("Diffusion (Aug)", diff_yt, diff_p["y_prob_aug"], "#ec4899"),
]:
    rp, ra = roc_curve(yt, yp)
    pp, pa = pr_curve(yt, yp)
    cp, ce = calibration_curve(yt, yp)
    models[name] = {
        "color": color, "roc": rp, "roc_auc": ra, "pr": pp, "pr_auc": pa,
        "cal": cp, "ece": ce, "brier": brier_score(yt, yp), "hist": score_histogram(yp),
    }

summary = {n: {"roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "brier": m["brier"], "ece": m["ece"], "color": m["color"]} for n, m in models.items()}

# ---- Load Temporal Backtest Data ----
print("Loading temporal backtest results...")
temporal_rows = []
with open(TEMPORAL_CSV, "r") as f:
    for row in csv.DictReader(f):
        temporal_rows.append(row)

# Structure: by horizon -> by model -> list of fold metrics
temporal_data = {}
for row in temporal_rows:
    h = int(row["horizon"])
    model = row["model"]
    if h not in temporal_data:
        temporal_data[h] = {}
    if model not in temporal_data[h]:
        temporal_data[h][model] = []
    temporal_data[h][model].append({
        "eval_year": int(row["eval_year"]),
        "train_end": int(row["train_end"]),
        "roc_auc": round(float(row.get("roc_auc", 0)), 4),
        "pr_auc": round(float(row.get("pr_auc", 0)), 4),
        "brier": round(float(row.get("brier_score", 0)), 5),
        "ece": round(float(row.get("ece", 0)), 4),
        "f1": round(float(row.get("f1", 0)), 4),
        "precision": round(float(row.get("precision", 0)), 4),
        "recall": round(float(row.get("recall", 0)), 4),
        "lift_1": round(float(row.get("lift@1%", 0)), 1),
        "lift_5": round(float(row.get("lift@5%", 0)), 1),
        "tp": int(row.get("tp", 0)),
        "fp": int(row.get("fp", 0)),
        "fn": int(row.get("fn", 0)),
        "tn": int(row.get("tn", 0)),
        "n_total": int(row.get("n_total", 0)),
        "n_positive": int(row.get("n_positive", 0)),
        "base_rate": round(float(row.get("base_rate", 0)), 6),
        "elapsed_s": round(float(row.get("elapsed_s", 0)), 1),
        "scenario_chain": row.get("scenario_chain", "False") == "True",
    })

# Compute means
temporal_means = {}
model_colors = {"LogReg": "#3b82f6", "CVAE": "#8b5cf6", "Diffusion": "#ec4899"}
for h in sorted(temporal_data.keys()):
    temporal_means[h] = {}
    for model in temporal_data[h]:
        folds = temporal_data[h][model]
        temporal_means[h][model] = {
            "color": model_colors.get(model, "#888"),
            "roc_auc_mean": round(np.mean([f["roc_auc"] for f in folds]), 4),
            "roc_auc_std": round(np.std([f["roc_auc"] for f in folds]), 4),
            "pr_auc_mean": round(np.mean([f["pr_auc"] for f in folds]), 4),
            "ece_mean": round(np.mean([f["ece"] for f in folds]), 4),
            "brier_mean": round(np.mean([f["brier"] for f in folds]), 5),
            "lift1_mean": round(np.mean([f["lift_1"] for f in folds]), 1),
            "f1_mean": round(np.mean([f["f1"] for f in folds]), 4),
            "n_folds": len(folds),
        }

print("Generating dashboard HTML...")

# ---- Generate HTML ----
html = """<!DOCTYPE html>
<html>
<head>
<title>Benchmark Dashboard — Zoning Opposition Prediction</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', system-ui, sans-serif;
    background: linear-gradient(135deg, #0a0a0f 0%%, #111118 50%%, #0d0d14 100%%);
    color: #e0e0e0; min-height: 100vh; padding: 24px;
  }
  .header { text-align: center; margin-bottom: 24px; }
  .header h1 {
    font-size: 28px; font-weight: 800; color: #fff;
    background: linear-gradient(135deg, #7c3aed, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .header .sub { font-size: 13px; color: #888; margin-top: 4px; }

  /* Tab navigation */
  .tab-nav { display: flex; justify-content: center; gap: 4px; margin-bottom: 24px; }
  .tab-btn {
    padding: 10px 24px; border-radius: 8px 8px 0 0; border: 1px solid rgba(255,255,255,0.08);
    border-bottom: none; background: rgba(255,255,255,0.02); color: #666; cursor: pointer;
    font-family: inherit; font-size: 13px; font-weight: 600; transition: all 0.2s;
  }
  .tab-btn:hover { background: rgba(255,255,255,0.05); color: #aaa; }
  .tab-btn.active { background: rgba(124,58,237,0.15); color: #c084fc; border-color: rgba(124,58,237,0.3); }
  .tab-content { display: none; }
  .tab-content.active { display: block; animation: fadeIn 0.3s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

  /* Model toggles */
  .toggle-bar { display: flex; justify-content: center; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
  .toggle-btn {
    padding: 8px 18px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.04); color: #aaa; cursor: pointer;
    font-family: inherit; font-size: 13px; font-weight: 500; transition: all 0.2s;
  }
  .toggle-btn:hover { background: rgba(255,255,255,0.08); color: #fff; }
  .toggle-btn.active { border-color: var(--c); color: #fff; background: color-mix(in srgb, var(--c) 20%%, transparent); }
  .toggle-btn .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%%; margin-right: 6px; }

  /* Summary cards */
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .summary-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 16px; text-align: center;
  }
  .summary-card .name { font-size: 13px; font-weight: 600; margin-bottom: 8px; }
  .summary-card .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .metric-val { font-size: 18px; font-weight: 700; }
  .metric-label { font-size: 9px; color: #888; text-transform: uppercase; letter-spacing: 1px; }

  /* Chart grid */
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 900px) { .chart-grid { grid-template-columns: 1fr; } }
  .chart-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 18px;
  }
  .chart-card h3 { font-size: 13px; font-weight: 600; margin-bottom: 10px; color: #ccc; }
  .chart-wrap { position: relative; height: 300px; }
  .chart-card.full { grid-column: 1 / -1; }
  .chart-card.full .chart-wrap { height: 350px; }

  /* Horizon tabs */
  .horizon-tabs { display: flex; gap: 6px; margin-bottom: 16px; justify-content: center; }
  .horizon-tab {
    padding: 6px 16px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03); color: #888; cursor: pointer;
    font-family: inherit; font-size: 12px; font-weight: 600; transition: all 0.2s;
  }
  .horizon-tab.active { background: rgba(236,72,153,0.15); color: #f472b6; border-color: rgba(236,72,153,0.3); }

  /* Data table */
  .data-table {
    width: 100%%; border-collapse: collapse; font-size: 12px; margin-top: 8px;
  }
  .data-table th, .data-table td {
    padding: 8px 10px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);
  }
  .data-table th { color: #888; font-weight: 600; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }
  .data-table tr:hover { background: rgba(255,255,255,0.02); }
  .val-good { color: #34d399; }
  .val-bad { color: #f87171; }
  .val-mid { color: #fbbf24; }

  /* Note bar */
  .note {
    text-align: center; font-size: 12px; color: #666; margin-top: 20px;
    padding: 10px; border: 1px solid rgba(255,255,255,0.05); border-radius: 8px;
    background: rgba(255,255,255,0.02);
  }
  .note strong { color: #f59e0b; }

  /* Comparison banner */
  .comparison-banner {
    display: grid; grid-template-columns: 1fr auto 1fr; gap: 16px; align-items: center;
    margin-bottom: 20px; padding: 16px; border-radius: 12px;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
  }
  .comparison-side { text-align: center; }
  .comparison-side .label { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
  .comparison-side .value { font-size: 28px; font-weight: 800; }
  .comparison-vs { font-size: 14px; color: #666; font-weight: 700; }
  .comparison-delta { font-size: 11px; margin-top: 2px; }
</style>
</head>
<body>

<div class="header">
  <h1>📊 Benchmark Dashboard</h1>
  <div class="sub">Zoning Opposition Prediction — Model Comparison</div>
</div>

<div class="tab-nav">
  <button class="tab-btn active" onclick="switchTab('random')">Random Split</button>
  <button class="tab-btn" onclick="switchTab('temporal')">Temporal Backtest</button>
  <button class="tab-btn" onclick="switchTab('compare')">Side-by-Side</button>
</div>

<!-- ============ TAB 1: RANDOM SPLIT ============ -->
<div class="tab-content active" id="tab-random">
  <div class="toggle-bar" id="toggleBar"></div>
  <div class="summary-grid" id="summaryGrid"></div>
  <div class="chart-grid">
    <div class="chart-card"><h3>ROC Curve</h3><div class="chart-wrap"><canvas id="rocChart"></canvas></div></div>
    <div class="chart-card"><h3>Precision-Recall Curve</h3><div class="chart-wrap"><canvas id="prChart"></canvas></div></div>
    <div class="chart-card"><h3>Calibration Diagram</h3><div class="chart-wrap"><canvas id="calChart"></canvas></div></div>
    <div class="chart-card"><h3>Score Distribution</h3><div class="chart-wrap"><canvas id="histChart"></canvas></div></div>
  </div>
  <div class="note"><strong>⚠️</strong> Random 80/20 split — spatial autocorrelation inflates metrics</div>
</div>

<!-- ============ TAB 2: TEMPORAL BACKTEST ============ -->
<div class="tab-content" id="tab-temporal">
  <div class="horizon-tabs" id="horizonTabs"></div>
  <div class="summary-grid" id="temporalSummary"></div>
  <div class="chart-grid">
    <div class="chart-card"><h3>ROC-AUC by Eval Year</h3><div class="chart-wrap"><canvas id="temporalAucChart"></canvas></div></div>
    <div class="chart-card"><h3>ECE by Eval Year (↓ better)</h3><div class="chart-wrap"><canvas id="temporalEceChart"></canvas></div></div>
    <div class="chart-card"><h3>Lift@1%% by Eval Year</h3><div class="chart-wrap"><canvas id="temporalLiftChart"></canvas></div></div>
    <div class="chart-card"><h3>Brier Score by Eval Year (↓ better)</h3><div class="chart-wrap"><canvas id="temporalBrierChart"></canvas></div></div>
  </div>
  <div class="chart-card full" style="margin-top: 14px;">
    <h3>Detailed Fold Results</h3>
    <div id="foldTable"></div>
  </div>
  <div class="note">✅ Expanding-window temporal validation — <strong>no future data leakage</strong>. Scenario chaining active for h≥2.</div>
</div>

<!-- ============ TAB 3: COMPARISON ============ -->
<div class="tab-content" id="tab-compare">
  <div id="compareBanners"></div>
  <div class="chart-grid">
    <div class="chart-card full"><h3>Random Split vs Temporal — ROC-AUC by Model</h3><div class="chart-wrap"><canvas id="compareAucChart"></canvas></div></div>
    <div class="chart-card"><h3>Calibration Comparison (ECE ↓)</h3><div class="chart-wrap"><canvas id="compareEceChart"></canvas></div></div>
    <div class="chart-card"><h3>Brier Score Comparison (↓)</h3><div class="chart-wrap"><canvas id="compareBrierChart"></canvas></div></div>
  </div>
  <div class="note"><strong>Key insight:</strong> Random-split AUC is inflated by 13-43pp due to spatial leakage. Temporal validation reveals true generalization.</div>
</div>

<script>
// ===== DATA =====
const MODELS = %s;
const SUMMARY = %s;
const TEMPORAL = %s;
const TEMPORAL_MEANS = %s;
const MODEL_ORDER = ['LogReg', 'CVAE', 'Diffusion'];
const MODEL_COLORS = {LogReg: '#3b82f6', CVAE: '#8b5cf6', Diffusion: '#ec4899'};

// ===== TAB SWITCHING =====
function switchTab(tab) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  event.target.classList.add('active');
}

// ===== RANDOM SPLIT TAB =====
const active = {};
Object.keys(MODELS).forEach(k => active[k] = true);

const bar = document.getElementById('toggleBar');
Object.keys(MODELS).forEach(name => {
  const btn = document.createElement('button');
  btn.className = 'toggle-btn active';
  btn.style.setProperty('--c', MODELS[name].color);
  btn.innerHTML = '<span class="dot" style="background:'+MODELS[name].color+'"></span>' + name;
  btn.onclick = () => { active[name] = !active[name]; btn.classList.toggle('active'); updateRandomCharts(); };
  bar.appendChild(btn);
});

function buildSummary() {
  const grid = document.getElementById('summaryGrid');
  grid.innerHTML = '';
  Object.keys(SUMMARY).forEach(name => {
    if (!active[name]) return;
    const s = SUMMARY[name];
    const card = document.createElement('div');
    card.className = 'summary-card';
    card.innerHTML = '<div class="name" style="color:'+s.color+'">'+name+'</div><div class="metrics"><div><div class="metric-val" style="color:'+s.color+'">'+s.roc_auc+'</div><div class="metric-label">ROC-AUC</div></div><div><div class="metric-val" style="color:'+s.color+'">'+s.pr_auc+'</div><div class="metric-label">PR-AUC</div></div><div><div class="metric-val" style="color:'+s.color+'">'+s.brier+'</div><div class="metric-label">Brier</div></div><div><div class="metric-val" style="color:'+s.color+'">'+s.ece+'</div><div class="metric-label">ECE</div></div></div>';
    grid.appendChild(card);
  });
}

Chart.defaults.color = '#888';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = 'Inter';
const chartOpts = { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } };
const gridColor = 'rgba(255,255,255,0.04)';

function makeScatter(ctx, xL, yL) {
  return new Chart(ctx, { type: 'scatter', data: { datasets: [] }, options: { ...chartOpts,
    scales: { x: { min:0, max:1, title:{display:true,text:xL,color:'#888'}, grid:{color:gridColor} },
              y: { min:0, max:1, title:{display:true,text:yL,color:'#888'}, grid:{color:gridColor} } } } });
}
function makeBar(ctx) {
  return new Chart(ctx, { type: 'bar', data: { labels: [], datasets: [] }, options: { ...chartOpts,
    scales: { x: { title:{display:true,text:'Predicted Probability',color:'#888'}, grid:{color:gridColor} },
              y: { title:{display:true,text:'Count',color:'#888'}, grid:{color:gridColor} } } } });
}

const rocChart = makeScatter(document.getElementById('rocChart'), 'False Positive Rate', 'True Positive Rate');
const prChart = makeScatter(document.getElementById('prChart'), 'Recall', 'Precision');
const calChart = makeScatter(document.getElementById('calChart'), 'Mean Predicted', 'Fraction Positive');
const histChart = makeBar(document.getElementById('histChart'));

function updateRandomCharts() {
  buildSummary();
  const diagLine = {data:[{x:0,y:0},{x:1,y:1}],borderColor:'rgba(255,255,255,0.15)',borderDash:[5,5],pointRadius:0,showLine:true,borderWidth:1};

  rocChart.data.datasets = [diagLine, ...Object.keys(MODELS).filter(n=>active[n]).map(n => ({
    data:MODELS[n].roc, borderColor:MODELS[n].color, pointRadius:0, showLine:true, borderWidth:2, tension:0.1
  }))]; rocChart.update('none');

  prChart.data.datasets = Object.keys(MODELS).filter(n=>active[n]).map(n => ({
    data:MODELS[n].pr, borderColor:MODELS[n].color, pointRadius:0, showLine:true, borderWidth:2, tension:0.1
  })); prChart.update('none');

  calChart.data.datasets = [diagLine, ...Object.keys(MODELS).filter(n=>active[n]).map(n => ({
    data:MODELS[n].cal, borderColor:MODELS[n].color, backgroundColor:MODELS[n].color+'40', pointRadius:5, showLine:true, borderWidth:2
  }))]; calChart.update('none');

  const am = Object.keys(MODELS).filter(n=>active[n]);
  if (am.length) {
    histChart.data.labels = MODELS[am[0]].hist.map(b=>b.x);
    histChart.data.datasets = am.map(n=>({data:MODELS[n].hist.map(b=>b.y),backgroundColor:MODELS[n].color+'60',borderColor:MODELS[n].color,borderWidth:1}));
  } else { histChart.data.labels=[]; histChart.data.datasets=[]; }
  histChart.update('none');
}
updateRandomCharts();

// ===== TEMPORAL TAB =====
let currentHorizon = 1;
const horizons = Object.keys(TEMPORAL).map(Number).sort();

// Build horizon tabs
const htabs = document.getElementById('horizonTabs');
horizons.forEach(h => {
  const btn = document.createElement('button');
  btn.className = 'horizon-tab' + (h===1?' active':'');
  btn.textContent = 'h=' + h + ' (' + h + ' yr ahead)';
  btn.onclick = () => { currentHorizon = h; document.querySelectorAll('.horizon-tab').forEach(b=>b.classList.remove('active')); btn.classList.add('active'); updateTemporalCharts(); };
  htabs.appendChild(btn);
});

// Temporal charts
const tAucChart = new Chart(document.getElementById('temporalAucChart'), { type: 'bar', data:{labels:[],datasets:[]}, options:{...chartOpts, scales:{x:{grid:{color:gridColor}},y:{min:0,max:1,title:{display:true,text:'ROC-AUC',color:'#888'},grid:{color:gridColor}}}, plugins:{legend:{display:true,labels:{usePointStyle:true,pointStyle:'circle',boxWidth:8,color:'#aaa',font:{size:11}}}}} });
const tEceChart = new Chart(document.getElementById('temporalEceChart'), { type: 'bar', data:{labels:[],datasets:[]}, options:{...chartOpts, scales:{x:{grid:{color:gridColor}},y:{min:0,max:0.6,title:{display:true,text:'ECE',color:'#888'},grid:{color:gridColor}}}, plugins:{legend:{display:true,labels:{usePointStyle:true,pointStyle:'circle',boxWidth:8,color:'#aaa',font:{size:11}}}}} });
const tLiftChart = new Chart(document.getElementById('temporalLiftChart'), { type: 'bar', data:{labels:[],datasets:[]}, options:{...chartOpts, scales:{x:{grid:{color:gridColor}},y:{title:{display:true,text:'Lift@1%%',color:'#888'},grid:{color:gridColor}}}, plugins:{legend:{display:true,labels:{usePointStyle:true,pointStyle:'circle',boxWidth:8,color:'#aaa',font:{size:11}}}}} });
const tBrierChart = new Chart(document.getElementById('temporalBrierChart'), { type: 'bar', data:{labels:[],datasets:[]}, options:{...chartOpts, scales:{x:{grid:{color:gridColor}},y:{min:0,max:0.3,title:{display:true,text:'Brier',color:'#888'},grid:{color:gridColor}}}, plugins:{legend:{display:true,labels:{usePointStyle:true,pointStyle:'circle',boxWidth:8,color:'#aaa',font:{size:11}}}}} });

function updateTemporalCharts() {
  const h = currentHorizon;
  const hData = TEMPORAL[h] || {};

  // Summary cards
  const tSum = document.getElementById('temporalSummary');
  tSum.innerHTML = '';
  const means = TEMPORAL_MEANS[h] || {};
  MODEL_ORDER.forEach(m => {
    if (!means[m]) return;
    const s = means[m];
    const card = document.createElement('div');
    card.className = 'summary-card';
    card.innerHTML = '<div class="name" style="color:'+s.color+'">'+m+' (h='+h+', n='+s.n_folds+' folds)</div><div class="metrics"><div><div class="metric-val" style="color:'+s.color+'">'+s.roc_auc_mean+'</div><div class="metric-label">AUC ±'+s.roc_auc_std+'</div></div><div><div class="metric-val" style="color:'+s.color+'">'+s.pr_auc_mean+'</div><div class="metric-label">PR-AUC</div></div><div><div class="metric-val" style="color:'+s.color+'">'+s.brier_mean+'</div><div class="metric-label">Brier</div></div><div><div class="metric-val" style="color:'+s.color+'">'+s.ece_mean+'</div><div class="metric-label">ECE</div></div></div>';
    tSum.appendChild(card);
  });

  // Get eval years
  const years = [...new Set(Object.values(hData).flat().map(f=>f.eval_year))].sort();
  const yearLabels = years.map(y => String(y));

  // AUC chart
  tAucChart.data.labels = yearLabels;
  tAucChart.data.datasets = MODEL_ORDER.filter(m=>hData[m]).map(m => ({
    label: m, data: years.map(y => { const f=hData[m].find(f=>f.eval_year===y); return f?f.roc_auc:null; }),
    backgroundColor: MODEL_COLORS[m]+'90', borderColor: MODEL_COLORS[m], borderWidth: 1, borderRadius: 4,
  })); tAucChart.update('none');

  tEceChart.data.labels = yearLabels;
  tEceChart.data.datasets = MODEL_ORDER.filter(m=>hData[m]).map(m => ({
    label: m, data: years.map(y => { const f=hData[m].find(f=>f.eval_year===y); return f?f.ece:null; }),
    backgroundColor: MODEL_COLORS[m]+'90', borderColor: MODEL_COLORS[m], borderWidth: 1, borderRadius: 4,
  })); tEceChart.update('none');

  tLiftChart.data.labels = yearLabels;
  tLiftChart.data.datasets = MODEL_ORDER.filter(m=>hData[m]).map(m => ({
    label: m, data: years.map(y => { const f=hData[m].find(f=>f.eval_year===y); return f?f.lift_1:null; }),
    backgroundColor: MODEL_COLORS[m]+'90', borderColor: MODEL_COLORS[m], borderWidth: 1, borderRadius: 4,
  })); tLiftChart.update('none');

  tBrierChart.data.labels = yearLabels;
  tBrierChart.data.datasets = MODEL_ORDER.filter(m=>hData[m]).map(m => ({
    label: m, data: years.map(y => { const f=hData[m].find(f=>f.eval_year===y); return f?f.brier:null; }),
    backgroundColor: MODEL_COLORS[m]+'90', borderColor: MODEL_COLORS[m], borderWidth: 1, borderRadius: 4,
  })); tBrierChart.update('none');

  // Fold table
  const table = document.getElementById('foldTable');
  let html = '<table class="data-table"><thead><tr><th>Year</th><th>Model</th><th>Train→</th><th>ROC-AUC</th><th>PR-AUC</th><th>Brier ↓</th><th>ECE ↓</th><th>Lift@1%%</th><th>F1</th><th>TP</th><th>FP</th><th>Chain</th></tr></thead><tbody>';
  years.forEach(y => {
    MODEL_ORDER.forEach(m => {
      if (!hData[m]) return;
      const f = hData[m].find(f=>f.eval_year===y);
      if (!f) return;
      const aucClass = f.roc_auc > 0.9 ? 'val-good' : f.roc_auc > 0.7 ? 'val-mid' : 'val-bad';
      const eceClass = f.ece < 0.05 ? 'val-good' : f.ece < 0.3 ? 'val-mid' : 'val-bad';
      html += '<tr><td>'+y+'</td><td style="color:'+MODEL_COLORS[m]+'">'+m+'</td><td>≤'+f.train_end+'</td><td class="'+aucClass+'">'+f.roc_auc+'</td><td>'+f.pr_auc+'</td><td>'+f.brier+'</td><td class="'+eceClass+'">'+f.ece+'</td><td>'+f.lift_1+'×</td><td>'+f.f1+'</td><td>'+f.tp+'</td><td>'+f.fp+'</td><td>'+(f.scenario_chain?'✓':'—')+'</td></tr>';
    });
  });
  html += '</tbody></table>';
  table.innerHTML = html;
}
updateTemporalCharts();

// ===== COMPARISON TAB =====
function buildComparison() {
  // Banners: random vs temporal AUC for each model
  const banners = document.getElementById('compareBanners');
  let bhtml = '';
  const randomAucs = {'LogReg': SUMMARY['LogReg Baseline']?.roc_auc, 'CVAE': SUMMARY['CVAE']?.roc_auc, 'Diffusion': SUMMARY['Diffusion (Aug)']?.roc_auc};
  MODEL_ORDER.forEach(m => {
    const rAuc = randomAucs[m] || 0;
    const tAuc = TEMPORAL_MEANS[1]?.[m]?.roc_auc_mean || 0;
    const delta = (tAuc - rAuc).toFixed(3);
    const deltaColor = delta < 0 ? '#f87171' : '#34d399';
    bhtml += '<div class="comparison-banner"><div class="comparison-side"><div class="label">Random Split</div><div class="value" style="color:'+MODEL_COLORS[m]+'">'+rAuc+'</div></div><div><div class="comparison-vs">vs</div><div class="comparison-delta" style="color:'+deltaColor+'">Δ '+delta+'</div></div><div class="comparison-side"><div class="label">Temporal h=1</div><div class="value" style="color:'+MODEL_COLORS[m]+'">'+tAuc+'</div></div></div>';
  });
  banners.innerHTML = bhtml;

  // Grouped bar: random vs temporal AUC
  const cAuc = new Chart(document.getElementById('compareAucChart'), {
    type: 'bar',
    data: {
      labels: MODEL_ORDER,
      datasets: [
        { label: 'Random Split', data: MODEL_ORDER.map(m=>randomAucs[m]||0), backgroundColor: 'rgba(255,255,255,0.15)', borderColor: 'rgba(255,255,255,0.3)', borderWidth: 1, borderRadius: 4 },
        { label: 'Temporal h=1', data: MODEL_ORDER.map(m=>TEMPORAL_MEANS[1]?.[m]?.roc_auc_mean||0), backgroundColor: MODEL_ORDER.map(m=>MODEL_COLORS[m]+'90'), borderColor: MODEL_ORDER.map(m=>MODEL_COLORS[m]), borderWidth: 1, borderRadius: 4 },
        { label: 'Temporal h=2', data: MODEL_ORDER.map(m=>TEMPORAL_MEANS[2]?.[m]?.roc_auc_mean||0), backgroundColor: MODEL_ORDER.map(m=>MODEL_COLORS[m]+'60'), borderColor: MODEL_ORDER.map(m=>MODEL_COLORS[m]), borderWidth: 1, borderRadius: 4, borderDash: [3,3] },
        { label: 'Temporal h=3', data: MODEL_ORDER.map(m=>TEMPORAL_MEANS[3]?.[m]?.roc_auc_mean||0), backgroundColor: MODEL_ORDER.map(m=>MODEL_COLORS[m]+'30'), borderColor: MODEL_ORDER.map(m=>MODEL_COLORS[m]), borderWidth: 1, borderRadius: 4 },
      ]
    },
    options: { ...chartOpts, scales: { x:{grid:{color:gridColor}}, y:{min:0,max:1,title:{display:true,text:'ROC-AUC',color:'#888'},grid:{color:gridColor}} }, plugins:{legend:{display:true,labels:{usePointStyle:true,pointStyle:'circle',boxWidth:8,color:'#aaa',font:{size:11}}}} }
  });

  // ECE comparison
  const randomEces = {'LogReg': SUMMARY['LogReg Baseline']?.ece, 'CVAE': SUMMARY['CVAE']?.ece, 'Diffusion': SUMMARY['Diffusion (Aug)']?.ece};
  new Chart(document.getElementById('compareEceChart'), {
    type: 'bar',
    data: {
      labels: MODEL_ORDER,
      datasets: [
        { label: 'Random', data: MODEL_ORDER.map(m=>randomEces[m]||0), backgroundColor: 'rgba(255,255,255,0.15)', borderColor:'rgba(255,255,255,0.3)', borderWidth:1, borderRadius:4 },
        { label: 'Temporal h=1', data: MODEL_ORDER.map(m=>TEMPORAL_MEANS[1]?.[m]?.ece_mean||0), backgroundColor: MODEL_ORDER.map(m=>MODEL_COLORS[m]+'70'), borderColor:MODEL_ORDER.map(m=>MODEL_COLORS[m]), borderWidth:1, borderRadius:4 },
      ]
    },
    options: { ...chartOpts, scales: { x:{grid:{color:gridColor}}, y:{min:0,max:0.6,title:{display:true,text:'ECE (↓ better)',color:'#888'},grid:{color:gridColor}} }, plugins:{legend:{display:true,labels:{usePointStyle:true,pointStyle:'circle',boxWidth:8,color:'#aaa',font:{size:11}}}} }
  });

  // Brier comparison
  const randomBriers = {'LogReg': SUMMARY['LogReg Baseline']?.brier, 'CVAE': SUMMARY['CVAE']?.brier, 'Diffusion': SUMMARY['Diffusion (Aug)']?.brier};
  new Chart(document.getElementById('compareBrierChart'), {
    type: 'bar',
    data: {
      labels: MODEL_ORDER,
      datasets: [
        { label: 'Random', data: MODEL_ORDER.map(m=>randomBriers[m]||0), backgroundColor: 'rgba(255,255,255,0.15)', borderColor:'rgba(255,255,255,0.3)', borderWidth:1, borderRadius:4 },
        { label: 'Temporal h=1', data: MODEL_ORDER.map(m=>TEMPORAL_MEANS[1]?.[m]?.brier_mean||0), backgroundColor: MODEL_ORDER.map(m=>MODEL_COLORS[m]+'70'), borderColor:MODEL_ORDER.map(m=>MODEL_COLORS[m]), borderWidth:1, borderRadius:4 },
      ]
    },
    options: { ...chartOpts, scales: { x:{grid:{color:gridColor}}, y:{min:0,max:0.3,title:{display:true,text:'Brier (↓ better)',color:'#888'},grid:{color:gridColor}} }, plugins:{legend:{display:true,labels:{usePointStyle:true,pointStyle:'circle',boxWidth:8,color:'#aaa',font:{size:11}}}} }
  });
}
buildComparison();
</script>
</body>
</html>
""" % (
    json.dumps(models),
    json.dumps(summary),
    json.dumps(temporal_data),
    json.dumps(temporal_means),
)

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print("Dashboard saved to %s" % OUT_HTML)
