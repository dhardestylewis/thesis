"""
Diffusion v3 Residual Analysis
================================
Comprehensive diagnostic of where the diffusion model underperforms LogReg.

Reads existing per-parcel scores (lr_score, diff_score, ensemble_score, actual)
and joins with panel data for feature-conditioned analysis.

Outputs:
  - Residual distributions (score separation)
  - Calibration by decile
  - Disagreement analysis (where LogReg right but Diffusion wrong, and vice versa)
  - Spatial error clustering (Moran's I)
  - Temporal persistence of errors
  - Feature-conditioned error (by property type, council district, land use, value change)
  - Diagnosis summary with architectural recommendations

No model re-run needed — works on existing CSV artifacts.
"""
import csv, json, sys, os
import numpy as np
from collections import defaultdict
from datetime import datetime

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

SCORES_PATH = "Analysis/Results/Diffusion_v3/per_parcel_scores.csv"
PANEL_PATH = "Data/Panel/Output/Property_Year_Panel_Enriched.csv"
OUT_DIR = "Analysis/Results/Diffusion_v3/diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ---- Load scores ----
log("Loading per-parcel scores...")
scores = []
with open(SCORES_PATH, "r") as f:
    for row in csv.DictReader(f):
        try:
            scores.append({
                "pid": row["parcel_id"],
                "year": int(row["year"]),
                "lr": float(row["lr_score"]),
                "diff": float(row["diff_score"]),
                "ens": float(row["ensemble_score"]),
                "actual": float(row["actual"]),
                "lat": float(row.get("lat", 0)),
                "lon": float(row.get("lon", 0)),
            })
        except (ValueError, KeyError):
            continue

log(f"Loaded {len(scores):,} scored parcels")

# ---- Load panel data for feature-conditioned analysis ----
log("Loading panel features...")
panel_features = {}  # (pid, year) -> features
with open(PANEL_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        year = int(row["year"])
        pid = row.get("standardized_tcad_id", "").strip()
        if pid and year >= 2021:
            try:
                mv = float(row.get("market_value", 0) or 0)
            except ValueError:
                mv = 0.0
            panel_features[(pid, year)] = {
                "property_category": row.get("property_category_code", ""),
                "land_use": row.get("lui_general_land_use", ""),
                "council_district": row.get("council_district", ""),
                "market_value": mv,
            }
log(f"Panel features: {len(panel_features):,} parcel-years")

# Enrich scores with panel features
enriched = 0
for s in scores:
    key = (s["pid"], s["year"])
    if key in panel_features:
        s.update(panel_features[key])
        enriched += 1
    else:
        s["property_category"] = ""
        s["land_use"] = ""
        s["council_district"] = ""
        s["market_value"] = 0.0
log(f"Enriched {enriched:,}/{len(scores):,} scores with panel features")

# Also compute value change for parcels with prior year data
for s in scores:
    prev_key = (s["pid"], s["year"] - 1)
    if prev_key in panel_features:
        prev_mv = panel_features[prev_key]["market_value"]
        if prev_mv > 0:
            s["value_change_pct"] = (s["market_value"] - prev_mv) / prev_mv
        else:
            s["value_change_pct"] = 0.0
    else:
        s["value_change_pct"] = 0.0

# ---- Group by year ----
by_year = defaultdict(list)
for s in scores:
    by_year[s["year"]].append(s)

# =======================================================
# 1. RESIDUAL ANALYSIS
# =======================================================
log("\n=== 1. RESIDUAL ANALYSIS ===")
residual_summary = {}

for year in sorted(by_year):
    rows = by_year[year]
    lr_probs = np.array([s["lr"] for s in rows])
    diff_probs = np.array([s["diff"] for s in rows])
    actuals = np.array([s["actual"] for s in rows])

    pos_mask = actuals == 1
    neg_mask = actuals == 0
    n_pos = int(pos_mask.sum())

    year_diag = {
        "year": year,
        "n_total": len(rows),
        "n_positive": n_pos,
        "prevalence_pct": round(n_pos / len(rows) * 100, 3),
        # Score separation
        "lr_mean_on_positives": round(float(np.mean(lr_probs[pos_mask])), 4) if n_pos > 0 else None,
        "lr_mean_on_negatives": round(float(np.mean(lr_probs[neg_mask])), 4),
        "lr_separation": None,
        "diff_mean_on_positives": round(float(np.mean(diff_probs[pos_mask])), 4) if n_pos > 0 else None,
        "diff_mean_on_negatives": round(float(np.mean(diff_probs[neg_mask])), 4),
        "diff_separation": None,
        # Score stats
        "lr_median": round(float(np.median(lr_probs)), 5),
        "diff_median": round(float(np.median(diff_probs)), 5),
        "lr_std": round(float(np.std(lr_probs)), 5),
        "diff_std": round(float(np.std(diff_probs)), 5),
    }
    if n_pos > 0:
        year_diag["lr_separation"] = round(year_diag["lr_mean_on_positives"] - year_diag["lr_mean_on_negatives"], 4)
        year_diag["diff_separation"] = round(year_diag["diff_mean_on_positives"] - year_diag["diff_mean_on_negatives"], 4)

    residual_summary[year] = year_diag
    log(f"\nYear {year} (n={len(rows):,}, pos={n_pos}):")
    log(f"  LogReg:    mean_pos={year_diag['lr_mean_on_positives']}, mean_neg={year_diag['lr_mean_on_negatives']}, separation={year_diag['lr_separation']}")
    log(f"  Diffusion: mean_pos={year_diag['diff_mean_on_positives']}, mean_neg={year_diag['diff_mean_on_negatives']}, separation={year_diag['diff_separation']}")

# =======================================================
# 2. CALIBRATION ANALYSIS
# =======================================================
log("\n=== 2. CALIBRATION ANALYSIS ===")
calibration = {}

for model_name, score_key in [("LogReg", "lr"), ("Diffusion", "diff"), ("Ensemble", "ens")]:
    cal_data = []
    for year in sorted(by_year):
        rows = by_year[year]
        probs = np.array([s[score_key] for s in rows])
        actuals = np.array([s["actual"] for s in rows])

        # 10 bins
        sorted_idx = np.argsort(probs)
        bin_size = len(probs) // 10
        year_bins = []
        for b in range(10):
            start = b * bin_size
            end = start + bin_size if b < 9 else len(probs)
            idx = sorted_idx[start:end]
            mean_pred = float(np.mean(probs[idx]))
            actual_rate = float(np.mean(actuals[idx]))
            year_bins.append({
                "year": year,
                "bin": b,
                "mean_predicted": round(mean_pred, 5),
                "actual_rate": round(actual_rate, 5),
                "n": len(idx),
                "gap": round(abs(mean_pred - actual_rate), 5),
            })
            cal_data.append(year_bins[-1])

        ece = np.mean([c["gap"] for c in year_bins])
        log(f"  {model_name} {year}: ECE={ece:.5f} | "
            f"Top decile: pred={year_bins[-1]['mean_predicted']:.4f}, actual={year_bins[-1]['actual_rate']:.4f}")

    calibration[model_name] = cal_data

# =======================================================
# 3. DISAGREEMENT ANALYSIS (key diagnostic)
# =======================================================
log("\n=== 3. DISAGREEMENT ANALYSIS ===")
log("  (where LogReg and Diffusion disagree on ranking)")
disagreement = {}

for year in sorted(by_year):
    rows = by_year[year]
    lr_probs = np.array([s["lr"] for s in rows])
    diff_probs = np.array([s["diff"] for s in rows])
    actuals = np.array([s["actual"] for s in rows])

    # Rank each model's predictions
    lr_ranks = np.argsort(np.argsort(-lr_probs))  # rank 0 = highest score
    diff_ranks = np.argsort(np.argsort(-diff_probs))

    n = len(rows)
    n_pos = int(actuals.sum())
    # Top-k analysis: how many positives does each model capture?
    for k_mult in [1, 2, 5, 10]:
        k = min(n_pos * k_mult, n)
        lr_top_k = set(np.where(lr_ranks < k)[0])
        diff_top_k = set(np.where(diff_ranks < k)[0])
        pos_set = set(np.where(actuals == 1)[0])

        lr_captures = len(lr_top_k & pos_set)
        diff_captures = len(diff_top_k & pos_set)
        both_capture = len(lr_top_k & diff_top_k & pos_set)
        lr_only = len((lr_top_k - diff_top_k) & pos_set)
        diff_only = len((diff_top_k - lr_top_k) & pos_set)
        neither = len(pos_set - lr_top_k - diff_top_k)

        log(f"  {year} @{k_mult}x (k={k}): LR={lr_captures}, Diff={diff_captures}, "
            f"Both={both_capture}, LR-only={lr_only}, Diff-only={diff_only}, Neither={neither}")

    # Rank correlation
    from scipy.stats import kendalltau, spearmanr
    tau, _ = kendalltau(lr_probs, diff_probs)
    rho, _ = spearmanr(lr_probs, diff_probs)

    # Score correlation on positives only
    if n_pos > 5:
        pos_tau, _ = kendalltau(lr_probs[actuals == 1], diff_probs[actuals == 1])
        pos_rho, _ = spearmanr(lr_probs[actuals == 1], diff_probs[actuals == 1])
    else:
        pos_tau = pos_rho = float('nan')

    # "Diffusion-wrong" analysis: cases where LR is right (high score for actual pos) but Diff is wrong
    # Define "right" as score > median of positive scores, "wrong" as score < median of all scores
    if n_pos > 0:
        lr_pos_scores = lr_probs[actuals == 1]
        diff_pos_scores = diff_probs[actuals == 1]
        lr_median_pos = np.median(lr_pos_scores)
        diff_median_all = np.median(diff_probs)

        # Positives that LR scores high but Diffusion scores low
        lr_right_diff_wrong = 0
        lr_wrong_diff_right = 0
        for i in np.where(actuals == 1)[0]:
            lr_high = lr_probs[i] > lr_median_pos
            diff_low = diff_probs[i] < diff_median_all
            diff_high = diff_probs[i] > np.median(diff_pos_scores)
            lr_low = lr_probs[i] < np.median(lr_probs)
            if lr_high and diff_low:
                lr_right_diff_wrong += 1
            if diff_high and lr_low:
                lr_wrong_diff_right += 1
    else:
        lr_right_diff_wrong = 0
        lr_wrong_diff_right = 0

    disagreement[year] = {
        "kendall_tau": round(float(tau), 4),
        "spearman_rho": round(float(rho), 4),
        "kendall_tau_positives": round(float(pos_tau), 4) if not np.isnan(pos_tau) else None,
        "spearman_rho_positives": round(float(pos_rho), 4) if not np.isnan(pos_rho) else None,
        "lr_right_diff_wrong": lr_right_diff_wrong,
        "lr_wrong_diff_right": lr_wrong_diff_right,
        "n_positive": n_pos,
    }
    log(f"  {year} rank corr: tau={tau:.4f}, rho={rho:.4f} | on positives: tau={pos_tau:.4f}")
    log(f"    LR✓ Diff✗: {lr_right_diff_wrong}/{n_pos}, LR✗ Diff✓: {lr_wrong_diff_right}/{n_pos}")

# =======================================================
# 4. SPATIAL ERROR CLUSTERING
# =======================================================
log("\n=== 4. SPATIAL STRUCTURE ===")
spatial_diag = {}

for year in sorted(by_year):
    rows = by_year[year]
    lats = np.array([s["lat"] for s in rows])
    lons = np.array([s["lon"] for s in rows])
    diff_resid = np.array([s["diff"] - s["actual"] for s in rows])
    lr_resid = np.array([s["lr"] - s["actual"] for s in rows])

    # Grid-based spatial analysis
    n_grid = 20
    lat_bins = np.linspace(lats.min(), lats.max() + 1e-9, n_grid + 1)
    lon_bins = np.linspace(lons.min(), lons.max() + 1e-9, n_grid + 1)
    lat_idx = np.digitize(lats, lat_bins) - 1
    lon_idx = np.digitize(lons, lon_bins) - 1

    grid_means_diff = np.full((n_grid, n_grid), np.nan)
    grid_means_lr = np.full((n_grid, n_grid), np.nan)
    grid_counts = np.zeros((n_grid, n_grid))

    cell_resids_diff = defaultdict(list)
    cell_resids_lr = defaultdict(list)
    for i in range(len(rows)):
        li, lo = min(lat_idx[i], n_grid-1), min(lon_idx[i], n_grid-1)
        cell_resids_diff[(li, lo)].append(diff_resid[i])
        cell_resids_lr[(li, lo)].append(lr_resid[i])
        grid_counts[li, lo] += 1

    for (li, lo), vals in cell_resids_diff.items():
        grid_means_diff[li, lo] = np.mean(vals)
    for (li, lo), vals in cell_resids_lr.items():
        grid_means_lr[li, lo] = np.mean(vals)

    # Moran's I approximation
    def approx_morans_i(grid):
        valid = ~np.isnan(grid)
        if valid.sum() < 10:
            return 0.0
        mean = np.nanmean(grid)
        vals = []
        neighbor_vals = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not valid[i, j]:
                    continue
                neighbors = []
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1] and valid[ni, nj]:
                        neighbors.append(grid[ni, nj] - mean)
                if neighbors:
                    vals.append(grid[i, j] - mean)
                    neighbor_vals.append(np.mean(neighbors))
        if len(vals) < 5:
            return 0.0
        vals = np.array(vals)
        neighbor_vals = np.array(neighbor_vals)
        denom = np.sum(vals**2)
        if denom < 1e-12:
            return 0.0
        return round(float(len(vals) * np.sum(vals * neighbor_vals) / denom), 4)

    mi_diff = approx_morans_i(grid_means_diff)
    mi_lr = approx_morans_i(grid_means_lr)

    spatial_diag[year] = {
        "morans_i_diff": mi_diff,
        "morans_i_lr": mi_lr,
        "n_occupied_cells": int((grid_counts > 0).sum()),
    }
    log(f"  Year {year}: Moran's I: Diff={mi_diff}, LR={mi_lr} (>0.3 = strong clustering)")

# =======================================================
# 5. TEMPORAL PERSISTENCE
# =======================================================
log("\n=== 5. TEMPORAL PERSISTENCE ===")
pid_errors_diff = defaultdict(list)
pid_errors_lr = defaultdict(list)
pid_years = defaultdict(list)

for s in scores:
    pid_errors_diff[s["pid"]].append((s["year"], s["diff"] - s["actual"]))
    pid_errors_lr[s["pid"]].append((s["year"], s["lr"] - s["actual"]))
    pid_years[s["pid"]].append(s["year"])

multi_year_pids = [pid for pid, years in pid_years.items() if len(set(years)) >= 3]
log(f"  Parcels in >=3 eval years: {len(multi_year_pids):,}")

temporal_diag = {"n_multi_year_parcels": len(multi_year_pids)}
if len(multi_year_pids) > 0:
    year_list = sorted(by_year.keys())
    pid_year_diff = defaultdict(dict)
    pid_year_lr = defaultdict(dict)
    for s in scores:
        pid_year_diff[s["pid"]][s["year"]] = s["diff"] - s["actual"]
        pid_year_lr[s["pid"]][s["year"]] = s["lr"] - s["actual"]

    temporal_corrs_diff = []
    temporal_corrs_lr = []
    for i in range(len(year_list) - 1):
        y1, y2 = year_list[i], year_list[i+1]
        d_vec1, d_vec2, l_vec1, l_vec2 = [], [], [], []
        for pid in multi_year_pids:
            if y1 in pid_year_diff[pid] and y2 in pid_year_diff[pid]:
                d_vec1.append(pid_year_diff[pid][y1])
                d_vec2.append(pid_year_diff[pid][y2])
                l_vec1.append(pid_year_lr[pid][y1])
                l_vec2.append(pid_year_lr[pid][y2])
        if len(d_vec1) > 10:
            d_corr = round(float(np.corrcoef(d_vec1, d_vec2)[0, 1]), 4)
            l_corr = round(float(np.corrcoef(l_vec1, l_vec2)[0, 1]), 4)
            temporal_corrs_diff.append(d_corr)
            temporal_corrs_lr.append(l_corr)
            log(f"  {y1}->{y2}: Diff error corr={d_corr:.4f}, LR error corr={l_corr:.4f}")

    temporal_diag["diff_temporal_corrs"] = temporal_corrs_diff
    temporal_diag["lr_temporal_corrs"] = temporal_corrs_lr
    temporal_diag["diff_avg_temporal_corr"] = round(float(np.mean(temporal_corrs_diff)), 4) if temporal_corrs_diff else None
    temporal_diag["lr_avg_temporal_corr"] = round(float(np.mean(temporal_corrs_lr)), 4) if temporal_corrs_lr else None

# =======================================================
# 6. FEATURE-CONDITIONED ERROR (key for architecture)
# =======================================================
log("\n=== 6. FEATURE-CONDITIONED ERROR ===")
feature_diag = {}

for feature_name in ["property_category", "land_use", "council_district"]:
    log(f"\n  --- {feature_name} ---")
    group_stats = defaultdict(lambda: {"lr_errors": [], "diff_errors": [], "actuals": [], "n": 0})

    for s in scores:
        val = s.get(feature_name, "")
        if not val:
            val = "MISSING"
        group_stats[val]["lr_errors"].append(s["lr"] - s["actual"])
        group_stats[val]["diff_errors"].append(s["diff"] - s["actual"])
        group_stats[val]["actuals"].append(s["actual"])
        group_stats[val]["n"] += 1

    feature_results = {}
    for val in sorted(group_stats, key=lambda x: -group_stats[x]["n"])[:15]:
        g = group_stats[val]
        n = g["n"]
        n_pos = int(sum(g["actuals"]))
        lr_mae = float(np.mean(np.abs(g["lr_errors"])))
        diff_mae = float(np.mean(np.abs(g["diff_errors"])))
        lr_bias = float(np.mean(g["lr_errors"]))
        diff_bias = float(np.mean(g["diff_errors"]))
        # Which model is better for this group?
        diff_advantage = lr_mae - diff_mae  # positive = diffusion is better

        feature_results[val] = {
            "n": n, "n_pos": n_pos, "prevalence_pct": round(n_pos / n * 100, 3),
            "lr_mae": round(lr_mae, 5), "diff_mae": round(diff_mae, 5),
            "lr_bias": round(lr_bias, 5), "diff_bias": round(diff_bias, 5),
            "diff_advantage": round(diff_advantage, 5),
        }
        winner = "DIFF" if diff_advantage > 0 else "LR"
        log(f"    {val:>6s}: n={n:>6,} pos={n_pos:>4} | LR_MAE={lr_mae:.4f} Diff_MAE={diff_mae:.4f} | "
            f"Winner={winner} (gap={abs(diff_advantage):.4f})")

    feature_diag[feature_name] = feature_results

# Value change bins
log("\n  --- value_change_pct ---")
vc_bins = [(-999, -0.1), (-0.1, 0.0), (0.0, 0.1), (0.1, 0.3), (0.3, 999)]
vc_labels = ["<-10%", "-10% to 0%", "0% to 10%", "10% to 30%", ">30%"]
vc_results = {}
for (lo, hi), label in zip(vc_bins, vc_labels):
    matching = [s for s in scores if lo <= s.get("value_change_pct", 0) < hi]
    if not matching:
        continue
    n = len(matching)
    n_pos = sum(1 for s in matching if s["actual"] == 1)
    lr_mae = float(np.mean([abs(s["lr"] - s["actual"]) for s in matching]))
    diff_mae = float(np.mean([abs(s["diff"] - s["actual"]) for s in matching]))
    diff_advantage = lr_mae - diff_mae

    vc_results[label] = {
        "n": n, "n_pos": n_pos, "prevalence_pct": round(n_pos / n * 100, 3),
        "lr_mae": round(lr_mae, 5), "diff_mae": round(diff_mae, 5),
        "diff_advantage": round(diff_advantage, 5),
    }
    winner = "DIFF" if diff_advantage > 0 else "LR"
    log(f"    {label:>12s}: n={n:>6,} pos={n_pos:>4} | LR_MAE={lr_mae:.4f} Diff_MAE={diff_mae:.4f} | "
        f"Winner={winner} (gap={abs(diff_advantage):.4f})")

feature_diag["value_change_pct"] = vc_results

# =======================================================
# 7. SCORE DISTRIBUTION ANALYSIS
# =======================================================
log("\n=== 7. SCORE DISTRIBUTIONS ===")
dist_diag = {}
for year in sorted(by_year):
    rows = by_year[year]
    lr_probs = np.array([s["lr"] for s in rows])
    diff_probs = np.array([s["diff"] for s in rows])

    dist_diag[year] = {
        "lr_percentiles": {f"p{p}": round(float(np.percentile(lr_probs, p)), 5) for p in [1, 5, 25, 50, 75, 95, 99]},
        "diff_percentiles": {f"p{p}": round(float(np.percentile(diff_probs, p)), 5) for p in [1, 5, 25, 50, 75, 95, 99]},
        "lr_fraction_above_50pct": round(float(np.mean(lr_probs > 0.5)), 5),
        "diff_fraction_above_50pct": round(float(np.mean(diff_probs > 0.5)), 5),
        "lr_fraction_above_10pct": round(float(np.mean(lr_probs > 0.1)), 5),
        "diff_fraction_above_10pct": round(float(np.mean(diff_probs > 0.1)), 5),
    }
    log(f"  {year}:")
    log(f"    LR   p50={dist_diag[year]['lr_percentiles']['p50']:.4f}, "
        f"p95={dist_diag[year]['lr_percentiles']['p95']:.4f}, >50%={dist_diag[year]['lr_fraction_above_50pct']:.4f}")
    log(f"    Diff p50={dist_diag[year]['diff_percentiles']['p50']:.4f}, "
        f"p95={dist_diag[year]['diff_percentiles']['p95']:.4f}, >50%={dist_diag[year]['diff_fraction_above_50pct']:.4f}")

# =======================================================
# 8. DIAGNOSIS SYNTHESIS
# =======================================================
log("\n" + "="*60)
log("DIAGNOSIS SYNTHESIS")
log("="*60)

# Determine key findings
avg_lr_sep = np.mean([v["lr_separation"] for v in residual_summary.values() if v["lr_separation"] is not None])
avg_diff_sep = np.mean([v["diff_separation"] for v in residual_summary.values() if v["diff_separation"] is not None])

avg_tau = np.mean([v["kendall_tau"] for v in disagreement.values()])
avg_mi_diff = np.mean([v["morans_i_diff"] for v in spatial_diag.values()])
avg_mi_lr = np.mean([v["morans_i_lr"] for v in spatial_diag.values()])

diagnosis = {
    "avg_lr_score_separation": round(avg_lr_sep, 4),
    "avg_diff_score_separation": round(avg_diff_sep, 4),
    "separation_ratio": round(avg_diff_sep / avg_lr_sep, 3) if avg_lr_sep > 0 else None,
    "avg_rank_correlation": round(avg_tau, 4),
    "avg_morans_i_diff": round(avg_mi_diff, 4),
    "avg_morans_i_lr": round(avg_mi_lr, 4),
    "spatial_errors_clustered": avg_mi_diff > 0.15,
    "temporal_errors_persistent": (temporal_diag.get("diff_avg_temporal_corr") or 0) > 0.3,
}

log(f"\n  Score separation: LR={avg_lr_sep:.4f}, Diff={avg_diff_sep:.4f} (ratio={diagnosis['separation_ratio']})")
log(f"  Rank correlation between models: tau={avg_tau:.4f} (1.0=identical)")
log(f"  Spatial clustering: Diff Moran's I={avg_mi_diff:.4f}, LR={avg_mi_lr:.4f}")
log(f"  Temporal persistence: Diff={temporal_diag.get('diff_avg_temporal_corr')}, LR={temporal_diag.get('lr_avg_temporal_corr')}")

# Architectural implications
log("\n  ARCHITECTURAL IMPLICATIONS:")
if diagnosis["separation_ratio"] and diagnosis["separation_ratio"] < 0.7:
    log("  [!] Diffusion score separation is <70% of LogReg → classifier head is weak")
    log("      → Consider: end-to-end training, larger classifier, or direct discriminative head")
elif diagnosis["separation_ratio"] and diagnosis["separation_ratio"] > 0.9:
    log("  [✓] Diffusion score separation is close to LogReg → ranking is similar")
    log("      → Problem is likely calibration, not architecture")

if avg_tau > 0.7:
    log("  [✓] Models largely agree on ranking → Diffusion is seeing the same signal")
    log("      → Calibration fix (Platt scaling) may be sufficient")
elif avg_tau < 0.4:
    log("  [!] Models disagree substantially → Diffusion is learning different features")
    log("      → Need deeper feature attribution analysis")

if diagnosis["spatial_errors_clustered"]:
    log("  [!] Diffusion errors cluster spatially → model misses geographic signal")
    log("      → Consider: spatial features, neighborhood-aware conditioning")

if diagnosis["temporal_errors_persistent"]:
    log("  [!] Diffusion errors persist across years for same parcels → systematic bias")
    log("      → Consider: parcel-specific bias correction or memory mechanism")

# ---- Save results ----
output = {
    "residuals": residual_summary,
    "calibration": calibration,
    "disagreement": disagreement,
    "spatial": spatial_diag,
    "temporal": temporal_diag,
    "feature_conditioned": feature_diag,
    "score_distributions": dist_diag,
    "diagnosis": diagnosis,
}

out_path = os.path.join(OUT_DIR, "v3_residual_analysis.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
log(f"\nSaved to {out_path}")

log("\nDone.")
