"""
causal_baselines.py
===================
Transparent causal baseline stack for the petition-pressure → zoning-concession thesis.

Stack (in order of interpretability):
  1. Hurdle DML        — ATE on P(concession>0) and E[concession|>0] separately
  2. Causal Forest     — CATE heterogeneity surface (where does petition pressure matter?)
  3. MSM / IPW         — longitudinal check with time-varying treatment and stabilised weights

Dataset: biweekly_panel.csv  (same panel used by causal_cfm_cvae.py)
Output:  output/baselines/
  ate_table.csv          — ATE + 95% CI for each estimator × each outcome
  cate_surface.csv       — per-case CATE from CausalForestDML
  msm_results.csv        — MSM coefficient table
  baselines_summary.txt  — human-readable digest

Runtime: ~10-25 min on CPU (no GPU needed).
Dependencies: pandas numpy scikit-learn lightgbm econml statsmodels
  pip install econml lightgbm statsmodels
"""

import os, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
PANEL_PATH = os.environ.get("PANEL_PATH", str(HERE / "biweekly_panel.csv"))
OUT_DIR    = Path(os.environ.get("OUT_DIR", str(HERE / "output"))) / "baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()

# ── Treatment & outcome definitions ──────────────────────────────────────────
TREATMENT        = "petition_pct_this_period"   # continuous dose ∈ [0,1]
OUTCOME_BINARY   = "height_concession_binary"   # P(concession>0)  [derived]
OUTCOME_CONT     = "Delta_Approved_Height"      # E[req_ht - init_ht | >0]
OUTCOME_RESOLVED = "resolved"                   # case resolved this period

# ── Confounders (cross-sectional, one row per case at filing) ─────────────────
CONFOUNDERS = [
    "latitude", "longitude",
    "race_white", "renter_share", "median_household_income",
    "proposed_max_far", "pdf_requested_height_ft",
    "mortgage_rate_30yr_momentum", "fed_funds_rate_momentum",
    "knn_petition_rate_1km", "dist_petition_rate_lag1",
]

print("=" * 60)
print("CAUSAL BASELINES -- petition pressure -> zoning concessions")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════════════
# 0. Load & prepare data
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[0/3] Loading panel...")
df_raw = pd.read_csv(PANEL_PATH, low_memory=False)
df_raw = df_raw.sort_values(["case_number", "period_seq"])

# Normalise petition dose to [0,1]
def fraction_01(s):
    x = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if x.quantile(0.99) > 1.0:
        x = x / 100.0
    return x.clip(0.0, 1.0)

df_raw[TREATMENT] = fraction_01(df_raw[TREATMENT])

# The target is now natively calculated in the biweekly panel with temporal tracking
if OUTCOME_CONT not in df_raw.columns:
    df_raw[OUTCOME_CONT] = 0.0

df_raw[OUTCOME_BINARY] = (df_raw[OUTCOME_CONT] > 0).astype(float)
if "resolved" not in df_raw.columns:
    df_raw["resolved"] = 0.0

# ── Cross-sectional collapse: one row per case (last observed period) ─────────
cs = (df_raw.groupby("case_number")
      .agg({
          TREATMENT:        "max",   # peak petition dose
          OUTCOME_BINARY:   "max",   # ever had concession
          OUTCOME_CONT:     "last",  # final requested height jump
          OUTCOME_RESOLVED: "max",
          **{c: "first" for c in CONFOUNDERS if c in df_raw.columns},
      })
      .reset_index())

available_conf = [c for c in CONFOUNDERS if c in cs.columns]
cs[available_conf] = cs[available_conf].fillna(cs[available_conf].median())
cs = cs.dropna(subset=[TREATMENT])

D = cs[TREATMENT].values
X = cs[available_conf].values
Y_bin  = cs[OUTCOME_BINARY].values
Y_cont = cs[OUTCOME_CONT].values
Y_res  = cs[OUTCOME_RESOLVED].values

# Positives-only mask for conditional stage
pos_mask = Y_bin > 0
print(f"  Cases: {len(cs)}  |  Confounders: {len(available_conf)}")
print(f"  Concession rate: {Y_bin.mean():.1%}  |  Positive concession cases: {pos_mask.sum()}")
print(f"  Mean petition dose: {D.mean():.4f}  |  Treated (D>0): {(D>0).mean():.1%}")

results = []   # accumulates ATE rows

# ═══════════════════════════════════════════════════════════════════════════════
# 1. HURDLE DML
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Hurdle DML...")
try:
    from econml.dml import LinearDML
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    lgbm_clf = LGBMRegressor(n_estimators=200, num_leaves=31, random_state=42, verbose=-1)
    lgbm_reg = LGBMRegressor(n_estimators=200, num_leaves=31, random_state=42, verbose=-1)

    # ── Stage A: P(concession > 0) ───────────────────────────────────────────
    print("  Stage A: P(concession>0)...")
    dml_bin = LinearDML(
        model_y=lgbm_clf,
        model_t=lgbm_reg,
        discrete_treatment=False,
        cv=5,
        random_state=42,
    )
    dml_bin.fit(Y_bin, D, X=X)
    ate_bin   = float(dml_bin.ate(X))
    ci_bin    = dml_bin.ate_interval(X, alpha=0.05)
    print(f"    ATE P(concession>0): {ate_bin:+.4f}  95% CI [{ci_bin[0]:.4f}, {ci_bin[1]:.4f}]")
    results.append({"estimator": "Hurdle-DML", "outcome": "P(concession>0)",
                    "ate": ate_bin, "ci_lo": float(ci_bin[0]), "ci_hi": float(ci_bin[1])})

    # ── Stage B: E[concession | concession > 0] ───────────────────────────────
    if pos_mask.sum() > 50:
        print("  Stage B: E[concession | >0]...")
        dml_cont = LinearDML(
            model_y=lgbm_reg,
            model_t=lgbm_reg,
            discrete_treatment=False,
            cv=5,
            random_state=42,
        )
        dml_cont.fit(Y_cont[pos_mask], D[pos_mask], X=X[pos_mask])
        ate_cont = float(dml_cont.ate(X[pos_mask]))
        ci_cont  = dml_cont.ate_interval(X[pos_mask], alpha=0.05)
        print(f"    ATE E[size|>0]:      {ate_cont:+.4f}  95% CI [{ci_cont[0]:.4f}, {ci_cont[1]:.4f}]")
        results.append({"estimator": "Hurdle-DML", "outcome": "E[concession_size|>0]",
                        "ate": ate_cont, "ci_lo": float(ci_cont[0]), "ci_hi": float(ci_cont[1])})

        # Combined E[Y(d)] = P(>0) × E[size|>0]  (marginal at mean D)
        marginal = ate_bin * Y_cont[pos_mask].mean() + Y_bin.mean() * ate_cont
        print(f"    Combined marginal ATE on E[Y]: {marginal:+.4f}")
        results.append({"estimator": "Hurdle-DML", "outcome": "E[Y] combined (marginal)",
                        "ate": marginal, "ci_lo": np.nan, "ci_hi": np.nan})

    # ── Resolved outcome ──────────────────────────────────────────────────────
    print("  Stage C: resolved...")
    dml_res = LinearDML(
        model_y=lgbm_clf,
        model_t=lgbm_reg,
        discrete_treatment=False,
        cv=5,
        random_state=42,
    )
    dml_res.fit(Y_res, D, X=X)
    ate_res = float(dml_res.ate(X))
    ci_res  = dml_res.ate_interval(X, alpha=0.05)
    print(f"    ATE P(resolved):     {ate_res:+.4f}  95% CI [{ci_res[0]:.4f}, {ci_res[1]:.4f}]")
    results.append({"estimator": "Hurdle-DML", "outcome": "P(resolved)",
                    "ate": ate_res, "ci_lo": float(ci_res[0]), "ci_hi": float(ci_res[1])})

except ImportError as e:
    print(f"  SKIP (missing dep): {e}")
    dml_bin = dml_cont = dml_res = None

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CAUSAL FOREST — CATE heterogeneity
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/3] Causal Forest (CATE heterogeneity)...")
cate_df = None
try:
    from econml.dml import CausalForestDML

    cf = CausalForestDML(
        n_estimators=500,
        min_samples_leaf=10,
        max_depth=None,
        random_state=42,
        cv=5,
        verbose=0,
    )
    cf.fit(Y_bin, D, X=X)
    cate_vals  = cf.effect(X).flatten()
    cate_lo, cate_hi = cf.effect_interval(X, alpha=0.05)

    ate_cf     = float(cate_vals.mean())
    ci_cf      = cf.ate_interval(X, alpha=0.05)
    print(f"  ATE (forest):  {ate_cf:+.4f}  95% CI [{ci_cf[0]:.4f}, {ci_cf[1]:.4f}]")
    print(f"  CATE std:      {cate_vals.std():.4f}  (heterogeneity signal)")
    results.append({"estimator": "CausalForest", "outcome": "P(concession>0)",
                    "ate": ate_cf, "ci_lo": float(ci_cf[0]), "ci_hi": float(ci_cf[1])})

    cate_df = cs[["case_number"] + available_conf].copy()
    cate_df["cate"]    = cate_vals
    cate_df["cate_lo"] = cate_lo.flatten()
    cate_df["cate_hi"] = cate_hi.flatten()
    cate_df["D"]       = D
    cate_df["Y_bin"]   = Y_bin
    cate_df.to_csv(OUT_DIR / "cate_surface.csv", index=False)
    print(f"  CATE surface saved → {OUT_DIR}/cate_surface.csv")

    # Top-5 features driving heterogeneity
    fi = pd.Series(cf.feature_importances_, index=available_conf).sort_values(ascending=False)
    print("  Top CATE drivers:")
    for feat, imp in fi.head(5).items():
        print(f"    {feat:<40} {imp:.4f}")

except ImportError as e:
    print(f"  SKIP (missing dep): {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MSM / IPW — longitudinal check
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/3] MSM / Stabilised IPW (longitudinal)...")
msm_rows = []
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import statsmodels.api as sm

    panel = df_raw.copy()
    panel[TREATMENT] = fraction_01(panel[TREATMENT])
    panel[OUTCOME_CONT]   = df_raw[OUTCOME_CONT]
    panel[OUTCOME_BINARY] = df_raw[OUTCOME_BINARY]

    # Time-varying confounders available in panel
    tv_conf = [c for c in [
        "cumulative_petition_pct_lag1",
        "cumulative_commission_hearings_lag1",
        "cumulative_council_hearings_lag1",
        "period_seq",
    ] if c in panel.columns]

    # Static baseline confounders (first period)
    static_conf = [c for c in available_conf if c in panel.columns]

    # Merge static confounders per case
    static_vals = (panel.groupby("case_number")[static_conf].first().reset_index())
    panel = panel.merge(static_vals, on="case_number", suffixes=("", "_base"))
    base_cols = [c + "_base" for c in static_conf if c + "_base" in panel.columns]

    all_conf_cols = tv_conf + base_cols
    all_conf_cols = [c for c in all_conf_cols if c in panel.columns]

    sub = panel.dropna(subset=[TREATMENT] + all_conf_cols).copy()
    sub[all_conf_cols] = sub[all_conf_cols].fillna(0)

    # ── Propensity model: P(D_t > 0 | L_t) ───────────────────────────────────
    treated_t  = (sub[TREATMENT] > 0).astype(int)
    scaler = StandardScaler()
    Xp = scaler.fit_transform(sub[all_conf_cols].values)

    prop_model = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    prop_model.fit(Xp, treated_t)
    p_treated  = prop_model.predict_proba(Xp)[:, 1].clip(0.01, 0.99)

    # Marginal model: overall P(D_t > 0)
    p_marginal = treated_t.mean()

    # Stabilised weights
    sw = np.where(treated_t == 1,
                  p_marginal       / p_treated,
                  (1 - p_marginal) / (1 - p_treated))
    sw = np.clip(sw, 0.1, 10.0)   # truncate at 10× for stability
    sub["sw"] = sw

    print(f"  IPW weights: mean={sw.mean():.3f}  std={sw.std():.3f}  max={sw.max():.2f}")

    # ── WLS outcome models ─────────────────────────────────────────────────────
    for outcome_col, label in [
        (OUTCOME_BINARY, "P(concession>0)"),
        (OUTCOME_CONT,   "E[concession_size]"),
        (OUTCOME_RESOLVED, "P(resolved)"),
    ]:
        if outcome_col not in sub.columns:
            continue
        y_msm   = sub[outcome_col].fillna(0).values
        d_msm   = sub[TREATMENT].values
        w_msm   = sub["sw"].values

        Xm = sm.add_constant(d_msm)
        try:
            wls = sm.WLS(y_msm, Xm, weights=w_msm).fit(
                cov_type="HC3"
            )
            coef  = float(wls.params[1])
            ci    = wls.conf_int(alpha=0.05)
            lo    = float(ci[1][0])
            hi    = float(ci[1][1])
            pval  = float(wls.pvalues[1])
            print(f"  MSM {label:<30} coef={coef:+.4f}  95% CI [{lo:.4f}, {hi:.4f}]  p={pval:.4f}")
            results.append({"estimator": "MSM-IPW", "outcome": label,
                            "ate": coef, "ci_lo": lo, "ci_hi": hi})
            msm_rows.append({"outcome": label, "coef": coef, "ci_lo": lo,
                             "ci_hi": hi, "pval": pval, "n_obs": len(y_msm)})
        except Exception as ex:
            print(f"  MSM {label}: failed — {ex}")

    pd.DataFrame(msm_rows).to_csv(OUT_DIR / "msm_results.csv", index=False)
    print(f"  MSM results saved → {OUT_DIR}/msm_results.csv")

except ImportError as e:
    print(f"  SKIP (missing dep): {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Save combined ATE table + summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── RESULTS ──────────────────────────────────────────────────")
ate_df = pd.DataFrame(results)
if not ate_df.empty:
    print(ate_df.to_string(index=False, float_format="{:+.4f}".format))
    ate_df.to_csv(OUT_DIR / "ate_table.csv", index=False)

elapsed = (time.time() - t0) / 60
summary_lines = [
    "CAUSAL BASELINES SUMMARY",
    "=" * 60,
    f"Dataset: {len(cs)} cases  |  Confounders: {len(available_conf)}",
    f"Concession rate: {Y_bin.mean():.1%}  |  Treated (D>0): {(D>0).mean():.1%}",
    f"Elapsed: {elapsed:.1f} min",
    "",
    "ATE TABLE",
    ate_df.to_string(index=False) if not ate_df.empty else "(no results)",
]
summary_path = OUT_DIR / "baselines_summary.txt"
summary_path.write_text("\n".join(summary_lines))

print(f"\nAll outputs → {OUT_DIR}")
print(f"Elapsed: {elapsed:.1f} min")
print("DONE")
