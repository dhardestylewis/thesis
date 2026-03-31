"""
Diffusion Model Residual Analysis
==================================
Diagnoses whether diffusion model errors are structured or noise.

Outputs:
  - Residual distributions (LogReg vs Diffusion)
  - Spatial autocorrelation (grid-binned Moran's I approximation)
  - Temporal persistence (same parcels erring each year?)
  - Calibration curves (predicted probability vs actual rate)
  - Error feature correlations
  - Summary JSON with all diagnostics
"""
import csv, json, os, sys
import numpy as np
from collections import defaultdict
from datetime import datetime

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

SCORES_PATH = "Analysis/Results/Diffusion_v2/per_parcel_scores.csv"
PANEL_PATH = "Data/Panel/Output/Property_Year_Panel_v3.csv"
OUT_DIR = "Analysis/Results/Diffusion_v2/diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ---- Load scores ----
log("Loading per-parcel scores...")
scores = []
with open(SCORES_PATH, "r") as f:
    for row in csv.DictReader(f):
        scores.append({
            "pid": row["parcel_id"],
            "year": int(row["year"]),
            "lr": float(row["lr_score"]),
            "diff": float(row["diff_score"]),
            "actual": int(row["actual"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
        })

log(f"Loaded {len(scores):,} scored parcels")

# ---- 1. Residual distributions ----
log("\n=== RESIDUAL ANALYSIS ===")

by_year = defaultdict(list)
for s in scores:
    by_year[s["year"]].append(s)

residual_summary = {}
for year in sorted(by_year):
    rows = by_year[year]
    lr_resid = np.array([s["lr"] - s["actual"] for s in rows])
    diff_resid = np.array([s["diff"] - s["actual"] for s in rows])
    actuals = np.array([s["actual"] for s in rows])
    n_pos = int(actuals.sum())
    
    # For positives only (how far off on the actual protests?)
    pos_mask = actuals == 1
    neg_mask = actuals == 0
    
    lr_on_pos = np.array([s["lr"] for s in rows])[pos_mask] if pos_mask.any() else np.array([])
    diff_on_pos = np.array([s["diff"] for s in rows])[pos_mask] if pos_mask.any() else np.array([])
    lr_on_neg = np.array([s["lr"] for s in rows])[neg_mask] if neg_mask.any() else np.array([])
    diff_on_neg = np.array([s["diff"] for s in rows])[neg_mask] if neg_mask.any() else np.array([])
    
    year_diag = {
        "year": year,
        "n_total": len(rows),
        "n_positive": n_pos,
        "prevalence_pct": round(n_pos / len(rows) * 100, 3),
        
        # LogReg residuals
        "lr_resid_mean": round(float(np.mean(lr_resid)), 5),
        "lr_resid_std": round(float(np.std(lr_resid)), 5),
        "lr_resid_median": round(float(np.median(lr_resid)), 5),
        "lr_resid_skew": round(float(np.mean(((lr_resid - np.mean(lr_resid)) / (np.std(lr_resid) + 1e-8)) ** 3)), 4),
        "lr_mean_on_positives": round(float(np.mean(lr_on_pos)), 4) if len(lr_on_pos) > 0 else None,
        "lr_mean_on_negatives": round(float(np.mean(lr_on_neg)), 4) if len(lr_on_neg) > 0 else None,
        
        # Diffusion residuals
        "diff_resid_mean": round(float(np.mean(diff_resid)), 5),
        "diff_resid_std": round(float(np.std(diff_resid)), 5),
        "diff_resid_median": round(float(np.median(diff_resid)), 5),
        "diff_resid_skew": round(float(np.mean(((diff_resid - np.mean(diff_resid)) / (np.std(diff_resid) + 1e-8)) ** 3)), 4),
        "diff_mean_on_positives": round(float(np.mean(diff_on_pos)), 4) if len(diff_on_pos) > 0 else None,
        "diff_mean_on_negatives": round(float(np.mean(diff_on_neg)), 4) if len(diff_on_neg) > 0 else None,
    }
    residual_summary[year] = year_diag
    
    log(f"\nYear {year} (n={len(rows):,}, pos={n_pos}):")
    log(f"  LogReg scores: mean_pos={year_diag['lr_mean_on_positives']}, mean_neg={year_diag['lr_mean_on_negatives']}, "
        f"separation={round(year_diag['lr_mean_on_positives'] - year_diag['lr_mean_on_negatives'], 4) if year_diag['lr_mean_on_positives'] else 'N/A'}")
    log(f"  Diffusion scores: mean_pos={year_diag['diff_mean_on_positives']}, mean_neg={year_diag['diff_mean_on_negatives']}, "
        f"separation={round(year_diag['diff_mean_on_positives'] - year_diag['diff_mean_on_negatives'], 4) if year_diag['diff_mean_on_positives'] else 'N/A'}")

# ---- 2. Calibration curves ----
log("\n=== CALIBRATION ANALYSIS ===")
log("  (bin predictions by decile, compare to actual positive rate)")

calibration = {}
for model_name, score_key in [("LogReg", "lr"), ("Diffusion", "diff")]:
    cal_data = []
    for year in sorted(by_year):
        rows = by_year[year]
        probs = np.array([s[score_key] for s in rows])
        actuals = np.array([s["actual"] for s in rows])
        
        # Sort into 10 bins
        n_bins = 10
        sorted_idx = np.argsort(probs)
        bin_size = len(sorted_idx) // n_bins
        
        for b in range(n_bins):
            start = b * bin_size
            end = (b + 1) * bin_size if b < n_bins - 1 else len(sorted_idx)
            idx = sorted_idx[start:end]
            
            mean_pred = float(np.mean(probs[idx]))
            actual_rate = float(np.mean(actuals[idx]))
            cal_data.append({
                "year": year,
                "bin": b,
                "mean_predicted": round(mean_pred, 5),
                "actual_rate": round(actual_rate, 5),
                "n": int(len(idx)),
                "gap": round(abs(mean_pred - actual_rate), 5),
            })
    
    calibration[model_name] = cal_data
    
    # Summary: expected calibration error
    for year in sorted(by_year):
        year_bins = [c for c in cal_data if c["year"] == year]
        ece = np.mean([c["gap"] for c in year_bins])
        log(f"  {model_name} {year}: ECE={ece:.5f} | "
            f"Top decile: pred={year_bins[-1]['mean_predicted']:.4f}, actual={year_bins[-1]['actual_rate']:.4f}")

# ---- 3. Spatial autocorrelation (grid-binned) ----
log("\n=== SPATIAL STRUCTURE ===")
log("  (do errors cluster spatially?)")

spatial_diag = {}
for year in sorted(by_year):
    rows = by_year[year]
    lats = np.array([s["lat"] for s in rows])
    lons = np.array([s["lon"] for s in rows])
    diff_resid = np.array([s["diff"] - s["actual"] for s in rows])
    lr_resid = np.array([s["lr"] - s["actual"] for s in rows])
    
    # Grid-bin approach: divide into ~20x20 grid cells
    n_grid = 20
    lat_bins = np.linspace(lats.min(), lats.max() + 1e-8, n_grid + 1)
    lon_bins = np.linspace(lons.min(), lons.max() + 1e-8, n_grid + 1)
    
    # Compute mean residual per grid cell
    grid_means_diff = np.full((n_grid, n_grid), np.nan)
    grid_means_lr = np.full((n_grid, n_grid), np.nan)
    grid_counts = np.zeros((n_grid, n_grid), dtype=int)
    
    lat_idx = np.digitize(lats, lat_bins) - 1
    lon_idx = np.digitize(lons, lon_bins) - 1
    lat_idx = np.clip(lat_idx, 0, n_grid - 1)
    lon_idx = np.clip(lon_idx, 0, n_grid - 1)
    
    for i in range(len(rows)):
        li, lo = lat_idx[i], lon_idx[i]
        grid_counts[li, lo] += 1
    
    # Compute per-cell means
    cell_resids_diff = defaultdict(list)
    cell_resids_lr = defaultdict(list)
    for i in range(len(rows)):
        cell_resids_diff[(lat_idx[i], lon_idx[i])].append(diff_resid[i])
        cell_resids_lr[(lat_idx[i], lon_idx[i])].append(lr_resid[i])
    
    for (li, lo), vals in cell_resids_diff.items():
        grid_means_diff[li, lo] = np.mean(vals)
    for (li, lo), vals in cell_resids_lr.items():
        grid_means_lr[li, lo] = np.mean(vals)
    
    # Approximate Moran's I: correlation between neighboring cells
    # For each cell, compute avg of 4 cardinal neighbors
    def approx_morans_i(grid):
        valid = ~np.isnan(grid)
        if valid.sum() < 10:
            return float('nan')
        
        mean_g = np.nanmean(grid)
        numerator = 0.0
        denominator = 0.0
        n_pairs = 0
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isnan(grid[i, j]):
                    continue
                denominator += (grid[i, j] - mean_g) ** 2
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1] and not np.isnan(grid[ni, nj]):
                        numerator += (grid[i, j] - mean_g) * (grid[ni, nj] - mean_g)
                        n_pairs += 1
        
        if denominator == 0 or n_pairs == 0:
            return float('nan')
        
        N = valid.sum()
        return round(float(N * numerator / (n_pairs * denominator)), 4)
    
    mi_diff = approx_morans_i(grid_means_diff)
    mi_lr = approx_morans_i(grid_means_lr)
    
    # Also: what fraction of variance in errors is explained by location?
    # Simple R² from grid cell assignment
    diff_overall_var = np.var(diff_resid)
    diff_within_var = 0.0
    total_in_cells = 0
    for vals in cell_resids_diff.values():
        if len(vals) > 1:
            diff_within_var += np.var(vals) * len(vals)
            total_in_cells += len(vals)
    diff_within_var /= max(total_in_cells, 1)
    spatial_r2_diff = 1 - diff_within_var / (diff_overall_var + 1e-10)
    
    lr_overall_var = np.var(lr_resid)
    lr_within_var = 0.0
    total_in_cells = 0
    for vals in cell_resids_lr.values():
        if len(vals) > 1:
            lr_within_var += np.var(vals) * len(vals)
            total_in_cells += len(vals)
    lr_within_var /= max(total_in_cells, 1)
    spatial_r2_lr = 1 - lr_within_var / (lr_overall_var + 1e-10)
    
    spatial_diag[year] = {
        "morans_i_diff": mi_diff,
        "morans_i_lr": mi_lr,
        "spatial_r2_diff": round(float(spatial_r2_diff), 4),
        "spatial_r2_lr": round(float(spatial_r2_lr), 4),
        "n_occupied_cells": int((grid_counts > 0).sum()),
    }
    
    log(f"  Year {year}:")
    log(f"    Moran's I: Diffusion={mi_diff}, LogReg={mi_lr} (>0.3 = strong clustering)")
    log(f"    Spatial R²: Diffusion={spatial_r2_diff:.4f}, LogReg={spatial_r2_lr:.4f} (location explains X% of error variance)")

# ---- 4. Temporal persistence ----
log("\n=== TEMPORAL PERSISTENCE ===")
log("  (do the same parcels get high errors across years?)")

# For parcels present in ≥2 years, compute correlation of errors
pid_errors_diff = defaultdict(list)
pid_errors_lr = defaultdict(list)
for s in scores:
    pid_errors_diff[s["pid"]].append(s["diff"] - s["actual"])
    pid_errors_lr[s["pid"]].append(s["lr"] - s["actual"])

# Parcels in all 4 years
multi_year_pids = [pid for pid, errs in pid_errors_diff.items() if len(errs) >= 3]
log(f"  Parcels in ≥3 eval years: {len(multi_year_pids):,}")

if len(multi_year_pids) > 0:
    # Compute avg correlation between consecutive year errors
    year_list = sorted(by_year.keys())
    pid_year_diff = defaultdict(dict)
    pid_year_lr = defaultdict(dict)
    for s in scores:
        pid_year_diff[s["pid"]][s["year"]] = s["diff"] - s["actual"]
        pid_year_lr[s["pid"]][s["year"]] = s["lr"] - s["actual"]
    
    temporal_corrs_diff = []
    temporal_corrs_lr = []
    for i in range(len(year_list) - 1):
        y1, y2 = year_list[i], year_list[i + 1]
        diffs1, diffs2 = [], []
        lrs1, lrs2 = [], []
        for pid in multi_year_pids:
            if y1 in pid_year_diff[pid] and y2 in pid_year_diff[pid]:
                diffs1.append(pid_year_diff[pid][y1])
                diffs2.append(pid_year_diff[pid][y2])
                lrs1.append(pid_year_lr[pid][y1])
                lrs2.append(pid_year_lr[pid][y2])
        
        if len(diffs1) > 10:
            d_corr = float(np.corrcoef(diffs1, diffs2)[0, 1])
            l_corr = float(np.corrcoef(lrs1, lrs2)[0, 1])
            temporal_corrs_diff.append(round(d_corr, 4))
            temporal_corrs_lr.append(round(l_corr, 4))
            log(f"  {y1}→{y2}: Diff error corr={d_corr:.4f}, LR error corr={l_corr:.4f}")

temporal_diag = {
    "n_multi_year_parcels": len(multi_year_pids),
    "diff_temporal_corrs": temporal_corrs_diff if len(multi_year_pids) > 0 else [],
    "lr_temporal_corrs": temporal_corrs_lr if len(multi_year_pids) > 0 else [],
    "diff_avg_temporal_corr": round(float(np.mean(temporal_corrs_diff)), 4) if temporal_corrs_diff else None,
    "lr_avg_temporal_corr": round(float(np.mean(temporal_corrs_lr)), 4) if temporal_corrs_lr else None,
}

log(f"\n  Avg temporal error correlation:")
log(f"    Diffusion: {temporal_diag['diff_avg_temporal_corr']}")
log(f"    LogReg: {temporal_diag['lr_avg_temporal_corr']}")
log(f"  (>0.5 = errors are persistent/structured, <0.2 = errors are random)")

# ---- 5. Score distribution analysis ----
log("\n=== SCORE DISTRIBUTIONS ===")

for year in sorted(by_year):
    rows = by_year[year]
    lr_scores = np.array([s["lr"] for s in rows])
    diff_scores = np.array([s["diff"] for s in rows])
    
    # What fraction of diffusion scores are near 0.5 (uninformative)?
    diff_near_half = np.mean(np.abs(diff_scores - 0.5) < 0.1)
    lr_near_half = np.mean(np.abs(lr_scores - 0.5) < 0.1)
    
    # Histogram bins
    diff_below_01 = np.mean(diff_scores < 0.1)
    diff_above_09 = np.mean(diff_scores > 0.9)
    lr_below_01 = np.mean(lr_scores < 0.1)
    lr_above_09 = np.mean(lr_scores > 0.9)
    
    log(f"  Year {year}:")
    log(f"    LR:   <0.1={lr_below_01:.3f}, 0.4-0.6={lr_near_half:.3f}, >0.9={lr_above_09:.3f}, mean={np.mean(lr_scores):.4f}")
    log(f"    Diff: <0.1={diff_below_01:.3f}, 0.4-0.6={diff_near_half:.3f}, >0.9={diff_above_09:.3f}, mean={np.mean(diff_scores):.4f}")

# ---- Save all diagnostics ----
all_diagnostics = {
    "residuals": residual_summary,
    "calibration": calibration,
    "spatial": spatial_diag,
    "temporal": temporal_diag,
    "diagnosis": {
        "is_error_structured": None,  # will be set based on results
        "is_data_issue": None,
        "is_model_issue": None,
    }
}

# Determine diagnosis
avg_morans_diff = np.mean([v["morans_i_diff"] for v in spatial_diag.values() if not np.isnan(v["morans_i_diff"])])
avg_temporal_diff = temporal_diag["diff_avg_temporal_corr"]

if avg_morans_diff is not None and avg_morans_diff > 0.3:
    all_diagnostics["diagnosis"]["is_error_structured"] = True
    log("\n⚠️  STRUCTURED ERRORS DETECTED: Diffusion errors cluster spatially (Moran's I > 0.3)")
elif avg_morans_diff is not None and avg_morans_diff > 0.1:
    all_diagnostics["diagnosis"]["is_error_structured"] = True
    log("\n⚠️  MODERATE STRUCTURE: Some spatial clustering in diffusion errors")
else:
    all_diagnostics["diagnosis"]["is_error_structured"] = False
    log("\n✓  Errors appear spatially random (Moran's I < 0.1)")

if avg_temporal_diff is not None and avg_temporal_diff > 0.5:
    log("⚠️  PERSISTENT ERRORS: Same parcels get wrong across years → model bias")
    all_diagnostics["diagnosis"]["is_model_issue"] = True
elif avg_temporal_diff is not None and avg_temporal_diff > 0.3:
    log("⚠️  MODERATE PERSISTENCE: Some parcels consistently wrong")
    all_diagnostics["diagnosis"]["is_model_issue"] = True
else:
    log("✓  Errors not strongly persistent across years")

# Final verdict
log("\n" + "=" * 60)
log("DIAGNOSIS VERDICT")
log("=" * 60)

# Check if diffusion scores are concentrated (uninformative)
all_diff_scores = np.array([s["diff"] for s in scores])
score_range = np.percentile(all_diff_scores, 95) - np.percentile(all_diff_scores, 5)
log(f"\nDiffusion score 5-95 percentile range: {score_range:.4f}")
if score_range < 0.3:
    log("⚠️  NARROW SCORE RANGE: Diffusion scores are concentrated → low discrimination")
    all_diagnostics["diagnosis"]["is_model_issue"] = True
else:
    log("✓  Score range is acceptable for discrimination")

# Save
diag_path = os.path.join(OUT_DIR, "residual_analysis.json")
with open(diag_path, "w") as f:
    json.dump(all_diagnostics, f, indent=2, default=str)
log(f"\nSaved diagnostics to {diag_path}")
log("Done!")
