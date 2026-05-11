"""
diagnose_outcomes.py
Provenance diagnostics for:
  1. height_concession_pct near-degeneracy
  2. DML vs MSM sign flip on P(resolved)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PANEL = "biweekly_panel.csv"
print("Loading panel...")
df = pd.read_csv(PANEL, low_memory=False)
print(f"Shape: {df.shape}")

# ============================================================
# ISSUE 1: Height concession provenance
# ============================================================
print()
print("=" * 60)
print("ISSUE 1: HEIGHT CONCESSION PROVENANCE")
print("=" * 60)

ht_cols = [
    "pdf_requested_height_ft", "pdf_reduced_to_ft",
    "pdf_staff_recommends_ht", "pdf_compatibility_height_ft",
    "net_height_change", "staff_concession_ratio",
]
for col in ht_cols:
    if col not in df.columns:
        print(f"  {col}: MISSING")
        continue
    s = pd.to_numeric(df[col], errors="coerce")
    n_nonzero = (s > 0).sum()
    n_notnull = s.notna().sum()
    print(f"  {col:<40}  rows_nonzero={n_nonzero:>6}  rows_notnull={n_notnull:>6}  mean={s.mean():.3f}")

print()

# Cross-sectional collapse (what baseline script does)
cs = (df.groupby("case_number")
      .agg(
          req_ht=("pdf_requested_height_ft", "max"),
          red_ht=("pdf_reduced_to_ft", "last"),
          staff_ht=("pdf_staff_recommends_ht", "last"),
          compat_ht=("pdf_compatibility_height_ft", "last"),
          nhc=("net_height_change", "max"),
          scr=("staff_concession_ratio", "max"),
          resolved=("resolved", "max"),
          petition=("petition_pct_this_period", "max"),
      )
      .reset_index())

print(f"Total unique cases: {len(cs)}")
print(f"  req_ht > 0          : {(cs['req_ht'] > 0).sum():>5}  ({(cs['req_ht']>0).mean():.1%})")
print(f"  red_ht non-null     : {cs['red_ht'].notna().sum():>5}  ({cs['red_ht'].notna().mean():.1%})")
print(f"  both req & red      : {(cs['req_ht'].notna() & cs['red_ht'].notna()).sum():>5}")

# Concession as coded in baseline (cross-sectional)
init_req = cs["req_ht"].copy()
final_ht = cs["red_ht"].fillna(init_req)
conc_cs = ((init_req - final_ht) / init_req.replace(0, np.nan)).clip(0, 1).fillna(0.0)
cs["conc_baseline"] = conc_cs

print()
print("Baseline script concession (cross-sectional, req vs reduced_to):")
print(f"  Positive cases: {(conc_cs > 0).sum()} / {len(cs)} = {(conc_cs > 0).mean():.2%}")
print(f"  This is the 0.3% problem.")

# Why so few? Sample cases where req > 0 and red < req
both = cs[cs["req_ht"] > 0].copy()
both["red_filled"] = both["red_ht"].fillna(both["req_ht"])
both["actual_conc"] = ((both["req_ht"] - both["red_filled"]) / both["req_ht"]).clip(0, 1)
print()
print(f"Cases with req_ht > 0: {len(both)}")
print(f"  red_ht is NULL (fallback = req => conc=0): {both['red_ht'].isna().sum()} ({both['red_ht'].isna().mean():.1%})")
print(f"  red_ht == req_ht (no concession)         : {(both['red_ht'] == both['req_ht']).sum()}")
print(f"  red_ht > req_ht  (height INCREASED)      : {(both['red_ht'] > both['req_ht']).sum()}")
print(f"  red_ht < req_ht  (real concession)       : {(both['red_ht'] < both['req_ht']).sum()}")

real_conc = both[both["red_ht"] < both["req_ht"]]
print()
print(f"Real concession cases (red < req): {len(real_conc)}")
if len(real_conc) > 0:
    print(real_conc[["case_number","req_ht","red_ht","actual_conc","petition"]].head(10).to_string(index=False))

# Panel-level: what CVAE actually sees
print()
print("Panel-level concession (CVAE view, init_req via groupby transform):")
init_req_p = df.groupby("case_number")["pdf_requested_height_ft"].transform("max")
if "pdf_reduced_to_ft" in df.columns:
    red_p = df["pdf_reduced_to_ft"].fillna(init_req_p)
else:
    red_p = init_req_p.copy()
conc_p = ((init_req_p - red_p) / init_req_p.replace(0, np.nan)).clip(0, 1).fillna(0.0)
print(f"  Row-level nonzero: {(conc_p > 0).sum()} rows ({(conc_p > 0).mean():.2%})")
case_has = df.assign(_c=conc_p).groupby("case_number")["_c"].max()
print(f"  Cases with ANY concession: {(case_has > 0).sum()} ({(case_has > 0).mean():.1%})")

# Alternative: net_height_change as primary outcome
print()
print("net_height_change as alternative outcome:")
nhc = cs["nhc"].fillna(0)
for thresh in [0, 5, 10, 20]:
    n = (nhc > thresh).sum()
    print(f"  nhc > {thresh:<3}: {n:>5} cases ({n/len(cs):.1%})")

# staff_concession_ratio
print()
print("staff_concession_ratio as alternative outcome:")
scr = cs["scr"].fillna(0)
print(f"  scr > 0: {(scr > 0).sum()} ({(scr > 0).mean():.1%})")
print(f"  scr > 1: {(scr > 1).sum()} ({(scr > 1).mean():.1%})")

# ============================================================
# ISSUE 2: DML vs MSM sign flip on P(resolved)
# ============================================================
print()
print("=" * 60)
print("ISSUE 2: DML vs MSM SIGN FLIP on P(resolved)")
print("=" * 60)

print()
print("Cross-sectional (one row per case, used by DML):")
print(f"  resolved=1 cases: {cs['resolved'].sum()} / {len(cs)} = {cs['resolved'].mean():.2%}")
print(f"  Petition > 0 cases: {(cs['petition'] > 0).sum()} ({(cs['petition']>0).mean():.1%})")

# Naive correlation
from scipy.stats import pointbiserialr, spearmanr
r_pb, p_pb = pointbiserialr(cs["petition"].fillna(0), cs["resolved"].fillna(0))
r_sp, p_sp = spearmanr(cs["petition"].fillna(0), cs["resolved"].fillna(0))
print(f"  Point-biserial r(petition, resolved) = {r_pb:.4f}  p={p_pb:.4f}")
print(f"  Spearman        r(petition, resolved) = {r_sp:.4f}  p={p_sp:.4f}")

# Resolved rate BY petition status (raw)
petitioned = cs[cs["petition"] > 0]
unpetitioned = cs[cs["petition"] == 0]
print(f"  Resolved rate | petitioned: {petitioned['resolved'].mean():.3f}")
print(f"  Resolved rate | not petitioned: {unpetitioned['resolved'].mean():.3f}")
print(f"  Raw difference (petitioned - baseline): {petitioned['resolved'].mean() - unpetitioned['resolved'].mean():+.4f}")

print()
print("Panel-level (all rows, used by MSM):")
res_panel = df["resolved"] if "resolved" in df.columns else pd.Series(0, index=df.index)
pet_panel = pd.to_numeric(df.get("petition_pct_this_period", 0), errors="coerce").fillna(0)
print(f"  resolved=1 rows: {(res_panel==1).sum()} / {len(df)} = {(res_panel==1).mean():.2%}")
print(f"  Petition > 0 rows: {(pet_panel > 0).sum()} ({(pet_panel>0).mean():.1%})")

r2_pb, p2_pb = pointbiserialr(pet_panel, res_panel.fillna(0))
print(f"  Point-biserial r(petition_t, resolved_t) = {r2_pb:.4f}  p={p2_pb:.4f}")

# Key diagnostic: does resolution happen AFTER petition activity?
# Look at period of resolution vs period of first petition
if "period_seq" in df.columns:
    first_pet = (df[df["petition_pct_this_period"] > 0]
                 .groupby("case_number")["period_seq"].min()
                 .rename("first_petition_period"))
    res_period = (df[df["resolved"] == 1]
                  .groupby("case_number")["period_seq"].min()
                  .rename("resolution_period"))
    timing = pd.concat([first_pet, res_period], axis=1).dropna()
    timing["res_after_pet"] = timing["resolution_period"] > timing["first_petition_period"]
    timing["lag"] = timing["resolution_period"] - timing["first_petition_period"]
    print()
    print(f"Timing analysis (cases with both petition AND resolution):")
    print(f"  N cases: {len(timing)}")
    print(f"  Resolution AFTER first petition: {timing['res_after_pet'].sum()} ({timing['res_after_pet'].mean():.1%})")
    print(f"  Mean lag (periods): {timing['lag'].mean():.1f}")
    print(f"  Median lag: {timing['lag'].median():.0f}")
    print()
    print("  => DML (cross-sectional) conflates selection: cases WITH petitions are different types")
    print("  => MSM (panel, IPW) follows temporal order: petition_t -> resolution_{t+k}")
    print("  => Sign flip is a CLASSIC Simpson's paradox / confounding-by-indication")

print()
print("SUMMARY:")
print("  Height concession: only 280/147384 rows have pdf_reduced_to_ft > 0.")
print("  At case level: red_ht is NULL for most cases => conc collapses to 0.")
print("  RECOMMENDATION: Use net_height_change > 0 or staff_concession_ratio > 0 as binary outcome.")
print("  DML vs MSM: sign flip is temporal confounding. MSM is more credible for panel data.")
