"""
Distributional Validation: Do forecasts look like history?
==========================================================
Compares the distribution of calibrated protest scores against
actual protest rates across:
  1. By year (does predicted rate match actual rate?)
  2. Across years (is the distribution stable?)
  3. By geography (spatial grid cells)
  4. By geotemporal (spatial × year)
  5. By property type

Uses isotonic-calibrated scores from exp02.
"""
import csv, json, os, sys
import numpy as np
from collections import defaultdict

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

CALIBRATED_SCORES = "Analysis/Results/Experiments/exp02_isotonic/per_parcel_scores.csv"
RAW_SCORES = "Analysis/Results/Diffusion_v3/per_parcel_scores.csv"
PANEL_PATH = "Data/Panel/Output/Property_Year_Panel_Enriched.csv"
OUT_DIR = "Analysis/Results/Diffusion_v3/diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load calibrated scores ----
print("Loading calibrated scores...")
scores = []
with open(CALIBRATED_SCORES) as f:
    for row in csv.DictReader(f):
        scores.append({
            "pid": row["pid"],
            "year": int(row["year"]),
            "lr": float(row["lr"]),
            "diff_raw": float(row["diff"]),
            "diff_cal": float(row["diff_calibrated"]),
            "ens_cal": float(row["ens_calibrated"]),
            "actual": float(row["actual"]),
        })

# Load lat/lon from raw scores
latlon = {}
with open(RAW_SCORES) as f:
    for row in csv.DictReader(f):
        latlon[row["parcel_id"]] = (float(row.get("lat", 0)), float(row.get("lon", 0)))

for s in scores:
    ll = latlon.get(s["pid"], (0, 0))
    s["lat"], s["lon"] = ll

# Load property categories
print("Loading property categories...")
panel_cats = {}
with open(PANEL_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pid = row.get("standardized_tcad_id", "").strip()
        year = int(row["year"])
        if pid:
            panel_cats[(pid, year)] = row.get("property_category_code", "")

for s in scores:
    s["pcat"] = panel_cats.get((s["pid"], s["year"]), "A")

print(f"Loaded {len(scores):,} calibrated scores")

by_year = defaultdict(list)
for s in scores:
    by_year[s["year"]].append(s)

results = {}

# ======================
# 1. BY YEAR
# ======================
print("\n=== 1. BY YEAR: Predicted rate vs Actual rate ===")
year_comparison = {}
for year in sorted(by_year):
    rows = by_year[year]
    actuals = np.array([s["actual"] for s in rows])
    cal_probs = np.array([s["diff_cal"] for s in rows])
    lr_probs = np.array([s["lr"] for s in rows])
    ens_probs = np.array([s["ens_cal"] for s in rows])

    actual_rate = float(actuals.mean())
    cal_mean = float(cal_probs.mean())
    lr_mean = float(lr_probs.mean())
    ens_mean = float(ens_probs.mean())

    year_comparison[year] = {
        "actual_rate": round(actual_rate * 100, 3),
        "diff_cal_predicted_rate": round(cal_mean * 100, 3),
        "lr_predicted_rate": round(lr_mean * 100, 3),
        "ens_predicted_rate": round(ens_mean * 100, 3),
        "rate_ratio_cal": round(cal_mean / max(actual_rate, 1e-6), 3),
        "n": len(rows),
        "n_pos": int(actuals.sum()),
        "pct_above_1pct": round(float((cal_probs > 0.01).mean()) * 100, 1),
        "pct_above_5pct": round(float((cal_probs > 0.05).mean()) * 100, 1),
        "pct_above_10pct": round(float((cal_probs > 0.10).mean()) * 100, 1),
        "pct_above_50pct": round(float((cal_probs > 0.50).mean()) * 100, 1),
    }

    ratio = year_comparison[year]["rate_ratio_cal"]
    match = "GOOD" if 0.8 < ratio < 1.2 else ("HIGH" if ratio > 1.2 else "LOW")
    print(f"  {year}: actual={actual_rate*100:.2f}%  predicted={cal_mean*100:.2f}%  ratio={ratio:.3f}  [{match}]")
    print(f"         >1%: {year_comparison[year]['pct_above_1pct']}%  >5%: {year_comparison[year]['pct_above_5pct']}%  >10%: {year_comparison[year]['pct_above_10pct']}%  >50%: {year_comparison[year]['pct_above_50pct']}%")

results["by_year"] = year_comparison

# ======================
# 2. ACROSS YEARS: Distribution stability
# ======================
print("\n=== 2. ACROSS YEARS: Score distribution stability ===")
print(f"  {'Year':>6s}  {'p10':>8s}  {'p25':>8s}  {'p50':>8s}  {'p75':>8s}  {'p90':>8s}  {'p99':>8s}")
stability = {}
for year in sorted(by_year):
    cal = np.array([s["diff_cal"] for s in by_year[year]])
    ps = np.percentile(cal, [10, 25, 50, 75, 90, 99])
    stability[year] = {f"p{p}": round(float(v), 5) for p, v in zip([10,25,50,75,90,99], ps)}
    print(f"  {year:>6d}  {ps[0]:>8.5f}  {ps[1]:>8.5f}  {ps[2]:>8.5f}  {ps[3]:>8.5f}  {ps[4]:>8.5f}  {ps[5]:>8.5f}")

# KL divergence between years (binned)
print("\n  KL divergence between years (lower = more similar):")
bins = np.linspace(0, 1, 51)
year_list = sorted(by_year.keys())
kl_matrix = {}
for y1 in year_list:
    for y2 in year_list:
        if y1 >= y2:
            continue
        h1, _ = np.histogram([s["diff_cal"] for s in by_year[y1]], bins=bins, density=True)
        h2, _ = np.histogram([s["diff_cal"] for s in by_year[y2]], bins=bins, density=True)
        h1 = h1 / h1.sum() + 1e-10
        h2 = h2 / h2.sum() + 1e-10
        kl = float(np.sum(h1 * np.log(h1 / h2)))
        kl_matrix[f"{y1}-{y2}"] = round(kl, 5)
        print(f"    {y1} vs {y2}: KL={kl:.5f}")

results["stability"] = {"percentiles": stability, "kl_divergence": kl_matrix}

# ======================
# 3. BY GEOGRAPHY (spatial grid)
# ======================
print("\n=== 3. BY GEOGRAPHY: Protest rates by spatial cell ===")
lats = np.array([s["lat"] for s in scores])
lons = np.array([s["lon"] for s in scores])

n_grid = 10
lat_bins = np.linspace(lats[lats > 0].min(), lats[lats > 0].max() + 1e-9, n_grid + 1)
lon_bins = np.linspace(lons[lons < 0].min(), lons[lons < 0].max() + 1e-9, n_grid + 1)

geo_cells = defaultdict(list)
for s in scores:
    if s["lat"] == 0:
        continue
    li = min(int(np.digitize(s["lat"], lat_bins)) - 1, n_grid - 1)
    lo = min(int(np.digitize(s["lon"], lon_bins)) - 1, n_grid - 1)
    geo_cells[(li, lo)].append(s)

geo_results = {}
print(f"  {'Cell':>8s}  {'N':>6s}  {'Actual':>8s}  {'Predicted':>10s}  {'Ratio':>6s}  {'Match':>6s}")
for (li, lo), rows in sorted(geo_cells.items(), key=lambda x: -len(x[1]))[:20]:
    actuals = np.array([s["actual"] for s in rows])
    predicted = np.array([s["diff_cal"] for s in rows])
    actual_rate = float(actuals.mean())
    pred_rate = float(predicted.mean())
    ratio = pred_rate / max(actual_rate, 1e-6)
    match = "GOOD" if 0.5 < ratio < 2.0 else "POOR"

    cell_key = f"{li},{lo}"
    geo_results[cell_key] = {
        "n": len(rows), "actual_rate": round(actual_rate * 100, 3),
        "predicted_rate": round(pred_rate * 100, 3), "ratio": round(ratio, 3),
        "lat_range": f"{lat_bins[li]:.3f}-{lat_bins[li+1]:.3f}",
        "lon_range": f"{lon_bins[lo]:.3f}-{lon_bins[lo+1]:.3f}",
    }
    print(f"  ({li:>2d},{lo:>2d})  {len(rows):>6d}  {actual_rate*100:>7.2f}%  {pred_rate*100:>9.2f}%  {ratio:>5.2f}  [{match}]")

results["by_geography"] = geo_results

# ======================
# 4. GEOTEMPORAL: Spatial × Year
# ======================
print("\n=== 4. GEOTEMPORAL: Rate change over time by area ===")
geotemp = {}
# Pick top 5 most populated cells
top_cells = sorted(geo_cells.items(), key=lambda x: -len(x[1]))[:5]
for (li, lo), all_rows in top_cells:
    cell_key = f"{li},{lo}"
    cell_by_year = defaultdict(list)
    for s in all_rows:
        cell_by_year[s["year"]].append(s)

    print(f"\n  Cell ({li},{lo}) — {len(all_rows)} parcels:")
    geotemp[cell_key] = {}
    for year in sorted(cell_by_year):
        rows = cell_by_year[year]
        actual_rate = np.mean([s["actual"] for s in rows])
        pred_rate = np.mean([s["diff_cal"] for s in rows])
        geotemp[cell_key][year] = {
            "actual_rate": round(actual_rate * 100, 3),
            "predicted_rate": round(pred_rate * 100, 3),
            "n": len(rows),
        }
        delta = "—" if year == min(cell_by_year) else f"{pred_rate*100 - geotemp[cell_key][year-1]['predicted_rate']:+.2f}pp"
        print(f"    {year}: actual={actual_rate*100:.2f}%  pred={pred_rate*100:.2f}%  trend={delta}")

results["geotemporal"] = geotemp

# ======================
# 5. BY PROPERTY TYPE
# ======================
print("\n=== 5. BY PROPERTY TYPE ===")
by_pcat = defaultdict(list)
for s in scores:
    by_pcat[s["pcat"]].append(s)

pcat_results = {}
print(f"  {'Cat':>6s}  {'N':>7s}  {'Actual':>8s}  {'Predicted':>10s}  {'Ratio':>6s}")
for pcat in sorted(by_pcat, key=lambda x: -len(by_pcat[x]))[:10]:
    rows = by_pcat[pcat]
    actual_rate = np.mean([s["actual"] for s in rows])
    pred_rate = np.mean([s["diff_cal"] for s in rows])
    ratio = pred_rate / max(actual_rate, 1e-6)

    pcat_results[pcat] = {
        "n": len(rows), "actual_rate": round(actual_rate * 100, 3),
        "predicted_rate": round(pred_rate * 100, 3), "ratio": round(ratio, 3),
    }
    print(f"  {pcat:>6s}  {len(rows):>7d}  {actual_rate*100:>7.2f}%  {pred_rate*100:>9.2f}%  {ratio:>5.2f}")

results["by_property_type"] = pcat_results

# ---- Save ----
out_path = os.path.join(OUT_DIR, "distributional_validation.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
print("Done.")
