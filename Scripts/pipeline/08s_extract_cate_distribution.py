"""
08s_extract_cate_distribution.py
Extracts heterogeneous treatment effects (CATE) from the saved Causal Forest
to support the thesis heterogeneous effects section reframing.
"""
import joblib, pathlib, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]

# ── Load model and panel ──────────────────────────────────────────────────────
print("Loading causal model and panel...", flush=True)
m = joblib.load(ROOT / "Data/Zoning_Cases/causal_models.pkl")
cf = m['cf_joint']
panel = pd.read_csv(ROOT / "Data/Panel/cross_sectional_dml_panel.csv")

ex_ante = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'race_black', 'race_hispanic',
    'renter_share', 'rent_burden', 'total_population', 'median_age',
    'appraised_value', 'building_age', 'mortgage_rate_30yr', 'fed_funds_rate',
    'local_unemployment_rate', 'knn_petition_rate_1km', 'dist_petition_rate_lag1',
    'fire_hazard_severity', 'slope_degree', 'is_imagine_corridor', 'petition_dose'
]

cs_surv = panel[panel['Withdrawal_Binary'] == 0].dropna(
    subset=ex_ante + ['Height_Attrition']
).copy()
X = cs_surv[ex_ante].values

# ── Extract CATE ─────────────────────────────────────────────────────────────
print(f"Extracting CATE for N={len(cs_surv)} surviving cases...", flush=True)
cate_raw = cf.effect(X)  # shape (N, n_outcomes) or (N,)

if cate_raw.ndim == 2:
    cate_height = cate_raw[:, 0]
    cate_delay = cate_raw[:, 1]
else:
    cate_height = cate_raw
    cate_delay = np.zeros_like(cate_raw)

# Scale the marginal effect (per 1 unit dose) by the statutory threshold (0.20)
# to get the effect of a legally valid 20% petition.
cate_height_20pct = cate_height * 0.20
cate_delay_20pct = cate_delay * 0.20

# Convert log-delay effect to an absolute days penalty multiplier 
# using the surviving sample's median baseline days (223 days).
cate_delay_days = 223.0 * (np.exp(cate_delay_20pct) - 1.0)

cs_surv['cate_height'] = cate_height_20pct
cs_surv['cate_delay_days'] = cate_delay_days

# ── Summary stats ────────────────────────────────────────────────────────────
def print_stats(cate, name, unit):
    print(f"\n=== CATE Distribution ({name}) ===", flush=True)
    print(f"N:           {len(cate)}")
    print(f"Mean:        {cate.mean():.2f} {unit}")
    print(f"Median:      {np.median(cate):.2f} {unit}")
    print(f"Std Dev:     {cate.std():.2f} {unit}")
    print(f"Q10:         {np.percentile(cate, 10):.2f}")
    print(f"Q25:         {np.percentile(cate, 25):.2f}")
    print(f"Q75:         {np.percentile(cate, 75):.2f}")
    print(f"Q90:         {np.percentile(cate, 90):.2f}")
    print(f"Min:         {cate.min():.2f}  Max: {cate.max():.2f}")
    print(f"Pct > 0:     {(cate > 0).mean()*100:.1f}%")

print_stats(cate_height, "Height Attrition", "feet")
print_stats(cate_delay_days, "Processing Delay", "days")

# ── Income quartile breakdown ────────────────────────────────────────────────
cs_surv['inc_q'] = pd.qcut(cs_surv['median_household_income'], 4,
                            labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'])
print("\n--- CATE by Median Income Quartile ---")
print(cs_surv.groupby('inc_q', observed=True)[['cate_height', 'cate_delay_days']]
      .agg(['mean', 'median']).round(2).to_string())

# ── Minority share breakdown ─────────────────────────────────────────────────
cs_surv['minority_share'] = 1 - cs_surv['race_white']
cs_surv['min_q'] = pd.qcut(cs_surv['minority_share'], 4,
                            labels=['Q1 (least diverse)', 'Q2', 'Q3', 'Q4 (most diverse)'])
print("\n--- CATE by Minority Share Quartile ---")
print(cs_surv.groupby('min_q', observed=True)[['cate_height', 'cate_delay_days']]
      .agg(['mean', 'median']).round(2).to_string())

# ── Renter share breakdown ───────────────────────────────────────────────────
cs_surv['renter_q'] = pd.qcut(cs_surv['renter_share'], 4,
                               labels=['Q1 (least renter)', 'Q2', 'Q3', 'Q4 (most renter)'])
print("\n--- CATE by Renter Share Quartile ---")
print(cs_surv.groupby('renter_q', observed=True)[['cate_height', 'cate_delay_days']]
      .agg(['mean', 'median']).round(2).to_string())

# ── Export ───────────────────────────────────────────────────────────────────
out = cs_surv[['case_number', 'latitude', 'longitude', 'cate_height', 'cate_delay_days',
               'median_household_income', 'minority_share', 'renter_share']].copy()
out_path = ROOT / "Data/Zoning_Cases/cate_distribution.csv"
out.to_csv(out_path, index=False)
print(f"\nExported CATE distribution to: {out_path}", flush=True)
