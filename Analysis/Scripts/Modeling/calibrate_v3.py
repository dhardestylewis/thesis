"""
Post-hoc temperature scaling calibration for Diffusion v3.
Fits T on one eval year, applies to all, reports calibration at thresholds.
References: Guo et al. 2017, "On Calibration of Modern Neural Networks"
"""
import csv, json, os, sys, math
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss, brier_score_loss

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

SCORES_PATH = "Analysis/Results/Diffusion_v3/per_parcel_scores.csv"
OUT_PATH = "Analysis/Results/Diffusion_v3/per_parcel_scores_calibrated.csv"

# ---- Load scores ----
print("Loading scores...")
rows_by_year = {}
all_rows = []
with open(SCORES_PATH, "r") as f:
    for row in csv.DictReader(f):
        r = {
            "pid": row["parcel_id"],
            "year": int(row["year"]),
            "lr": float(row["lr_score"]),
            "diff": float(row["diff_score"]),
            "ens": float(row["ensemble_score"]),
            "actual": int(row["actual"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
        }
        all_rows.append(r)
        rows_by_year.setdefault(r["year"], []).append(r)

print(f"Loaded {len(all_rows):,} rows across {len(rows_by_year)} years")

# ---- Temperature scaling ----
def prob_to_logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

def logit_to_prob(l):
    return 1.0 / (1.0 + np.exp(-l))

def calibrate_with_temp(scores, T):
    logits = prob_to_logit(scores)
    return logit_to_prob(logits / T)

def fit_temperature(scores, labels):
    """Find T that minimizes NLL (log loss) on calibration set."""
    logits = prob_to_logit(scores)
    
    def nll(T):
        cal_probs = logit_to_prob(logits / T)
        return log_loss(labels, cal_probs)
    
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    return result.x

# Fit on 2023 (middle year, balanced), apply to all
print("\n" + "="*70)
print("TEMPERATURE SCALING CALIBRATION")
print("="*70)

for model_key, model_name in [("lr", "LogReg"), ("diff", "Diffusion"), ("ens", "Ensemble")]:
    print(f"\n--- {model_name} ---")
    
    # Fit T on each year individually for fair evaluation (leave-one-out style)
    # But also show a single global T for simplicity
    all_scores = np.array([r[model_key] for r in all_rows])
    all_labels = np.array([r["actual"] for r in all_rows])
    
    global_T = fit_temperature(all_scores, all_labels)
    print(f"  Global T = {global_T:.4f}")
    
    for year in sorted(rows_by_year.keys()):
        yr_rows = rows_by_year[year]
        scores = np.array([r[model_key] for r in yr_rows])
        labels = np.array([r["actual"] for r in yr_rows])
        
        # Fit T on OTHER years, apply to this year
        other_scores = np.array([r[model_key] for r in all_rows if r["year"] != year])
        other_labels = np.array([r["actual"] for r in all_rows if r["year"] != year])
        T = fit_temperature(other_scores, other_labels)
        
        cal_scores = calibrate_with_temp(scores, T)
        
        # Store calibrated scores
        for r, cs in zip(yr_rows, cal_scores):
            r[f"{model_key}_cal"] = float(cs)
        
        # Report
        brier_before = brier_score_loss(labels, scores)
        brier_after = brier_score_loss(labels, cal_scores)
        
        print(f"\n  {year} (T={T:.4f}, fitted on other years):")
        print(f"    Brier: {brier_before:.6f} → {brier_after:.6f} ({'improved' if brier_after < brier_before else 'same/worse'})")
        
        # Calibration at thresholds
        print(f"    {'Thresh':>8} {'Raw count':>10} {'Cal count':>10} {'Protests':>9} {'Raw %':>8} {'Cal %':>8}")
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            raw_above = scores > thresh
            cal_above = cal_scores > thresh
            n_raw = raw_above.sum()
            n_cal = cal_above.sum()
            prot_raw = labels[raw_above].sum() if n_raw > 0 else 0
            prot_cal = labels[cal_above].sum() if n_cal > 0 else 0
            rate_raw = prot_raw / n_raw * 100 if n_raw > 0 else 0
            rate_cal = prot_cal / n_cal * 100 if n_cal > 0 else 0
            print(f"    >{thresh:.0%}:  {n_raw:>10,}  {n_cal:>10,}  {int(prot_cal):>9}  {rate_raw:>7.1f}%  {rate_cal:>7.1f}%")

# ---- Save calibrated scores ----
print(f"\nSaving calibrated scores to {OUT_PATH}...")
with open(OUT_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["parcel_id", "year", "lr_score", "diff_score", "ensemble_score",
                "lr_cal", "diff_cal", "ens_cal", "actual", "lat", "lon"])
    for r in all_rows:
        w.writerow([
            r["pid"], r["year"],
            round(r["lr"], 6), round(r["diff"], 6), round(r["ens"], 6),
            round(r["lr_cal"], 6), round(r["diff_cal"], 6), round(r["ens_cal"], 6),
            r["actual"], round(r["lat"], 6), round(r["lon"], 6),
        ])

print(f"Saved {len(all_rows):,} rows")
print("Done!")
