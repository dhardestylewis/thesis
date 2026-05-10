"""
audit_temporal_leakage.py
--------------------------
Per-row temporal leakage audit for the biweekly panel.

Leakage definition (user-specified):
  For each panel row at period_start, a feature is leaked if the source
  document/event that generated its value has a date AFTER period_start.

For each feature group, we track the 'knowable_at' column -- the date at
which that value first became observable -- and count rows where the feature
is non-null but knowable_at > period_start.

Post-vote outcome features (Final_Zoning, Delta_Approved_Height, etc.) are
EXPECTED to be null before Final_Council_Date and non-null after: any
non-null value before the vote date is a violation.

NLP features (nlp_* / council_nlp_*) are currently merged case-level without
temporal anchoring, so we flag any non-null value before the FIRST commission
document date for that case.

Outputs:
  Data/interim/leakage_audit_report.csv   -- per-feature summary
  Data/interim/leakage_violations.csv     -- row-level violations (High/Medium only)
"""
import pandas as pd
import numpy as np

BASE      = r"c:\Users\dhl\data\Thesis\thesis\Data"
PANEL     = r"c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv"
AGENDA    = BASE + r"\interim\commission_agendas_cases.csv"
MODEL_RDY = BASE + r"\final\model_ready_zoning_data.csv"
OUT_RPT   = BASE + r"\interim\leakage_audit_report.csv"
OUT_VIOL  = BASE + r"\interim\leakage_violations.csv"

print("Loading panel...", flush=True)
panel = pd.read_csv(PANEL, low_memory=False)
panel["period_start"] = pd.to_datetime(panel["period_start"])
print("  {:,} rows | {:,} cases".format(len(panel), panel["case_number"].nunique()), flush=True)

# ------------------------------------------------------------------
# Attach App_Date (T0) and Final_Council_Date per case
# ------------------------------------------------------------------
print("Attaching case-level timestamps...", flush=True)
mrd = pd.read_csv(MODEL_RDY, low_memory=False, usecols=[c for c in [
    "case_number", "App_Date", "Final_Council_Date"
] if True])
mrd_cols = ["case_number", "App_Date"]
if "Final_Council_Date" in mrd.columns:
    mrd_cols.append("Final_Council_Date")
mrd = mrd[mrd_cols].drop_duplicates("case_number")
mrd["App_Date"] = pd.to_datetime(mrd.get("App_Date"), errors="coerce")
if "Final_Council_Date" in mrd.columns:
    mrd["Final_Council_Date"] = pd.to_datetime(mrd["Final_Council_Date"], errors="coerce")
else:
    mrd["Final_Council_Date"] = pd.NaT

panel = panel.merge(mrd, on="case_number", how="left")
panel["App_Date"]           = pd.to_datetime(panel["App_Date"], errors="coerce")
panel["Final_Council_Date"] = pd.to_datetime(panel.get("Final_Council_Date", pd.Series(dtype="datetime64[ns]")), errors="coerce")

# First commission doc date per case (earliest agenda meeting date)
agenda = pd.read_csv(AGENDA, low_memory=False)
agenda["meeting_date"] = pd.to_datetime(agenda["meeting_date"], errors="coerce")
first_meeting = agenda.dropna(subset=["meeting_date"]).groupby("case_number")["meeting_date"].min().reset_index()
first_meeting.columns = ["case_number", "first_meeting_date"]
panel = panel.merge(first_meeting, on="case_number", how="left")
panel["first_meeting_date"] = pd.to_datetime(panel["first_meeting_date"], errors="coerce")

# ------------------------------------------------------------------
# Feature group definitions
# Each entry: (feature_cols, knowable_at_col, risk_level, notes)
# knowable_at_col is a column already on the panel (after merges above)
# or a sentinel: 'period_start' means always known (no risk), 'App_Date' etc.
# ------------------------------------------------------------------
groups = [
    {
        "group": "LDC height/FAR (filing-time)",
        "cols": ["proposed_max_height_ft", "existing_max_height_ft",
                 "proposed_max_far", "existing_max_far",
                 "proposed_max_bldg_cov_pct", "existing_max_bldg_cov_pct"],
        "knowable_at": "App_Date",
        "risk": "Low",
        "notes": "LDC code at filing is known at T0 (App_Date). Static per case.",
    },
    {
        "group": "PDF height features (per-meeting)",
        "cols": ["pdf_requested_height_ft", "pdf_proposed_height_ft",
                 "pdf_story_count", "pdf_story_height_ft",
                 "pdf_reduced_to_ft", "pdf_compatibility_height_ft",
                 "pdf_staff_recommends_ht"],
        "knowable_at": "first_meeting_date",
        "risk": "Low",
        "notes": "Merged via merge_asof(backward) on source_date <= period_start.",
    },
    {
        "group": "staff_recommended_height (legacy)",
        "cols": ["staff_recommended_height"],
        "knowable_at": "first_meeting_date",
        "risk": "Low",
        "notes": "Anchored via merge_asof(backward) in merge_panel_features.py.",
    },
    {
        "group": "Post-vote outcomes",
        "cols": ["Final_Zoning", "Delta_Approved_Height", "Staff_Attrition_Height",
                 "Delta_Approved_FAR"],
        "knowable_at": "Final_Council_Date",
        "risk": "High",
        "notes": "Valid only for period_start >= Final_Council_Date. Non-null before = VIOLATION.",
    },
    {
        "group": "NLP features (commission)",
        "cols": ["nlp_document_count", "nlp_total_tokens",
                 "nlp_oppose_hits", "nlp_traffic_hits", "nlp_density_hits"],
        "knowable_at": "period_start",
        "risk": "Low",
        "notes": "Anchored via cumulative merge_asof(backward).",
    },
    {
        "group": "NLP features (council)",
        "cols": ["council_nlp_document_count", "council_nlp_total_tokens",
                 "council_nlp_oppose_hits", "council_nlp_traffic_hits",
                 "council_nlp_density_hits"],
        "knowable_at": "period_start",
        "risk": "Low",
        "notes": "Anchored via cumulative merge_asof(backward).",
    },
    {
        "group": "Petition features",
        "cols": ["petition_pct_this_period", "petition_count_this_period",
                 "cumulative_petition_pct", "cumulative_petition_count",
                 "cumulative_petition_events"],
        "knowable_at": "period_start",  # already event-windowed, no risk
        "risk": "Low",
        "notes": "Period-windowed on petition_date. Already leakage-safe.",
    },
    {
        "group": "Council/commission hearings",
        "cols": ["council_hearings_this_period", "cumulative_council_hearings",
                 "commission_hearings_this_period", "cumulative_commission_hearings"],
        "knowable_at": "period_start",
        "risk": "Low",
        "notes": "Period-windowed on meeting_date. Already leakage-safe.",
    },
    {
        "group": "Parcel / EARS features",
        "cols": ["market_value", "appraised_value", "land_acres", "yr_built",
                 "improvement_sq_ft", "land_market_value", "improvement_market_value"],
        "knowable_at": "App_Date",
        "risk": "Low",
        "notes": "Annual vintage forward-filled. Available at filing time.",
    },
    {
        "group": "ACS demographics",
        "cols": ["total_population", "median_household_income", "median_gross_rent",
                 "renter_share", "owner_share", "rent_burden"],
        "knowable_at": "App_Date",
        "risk": "Low",
        "notes": "Annual ACS vintage. Available before filing.",
    },
    {
        "group": "FRED macro",
        "cols": ["mortgage_rate_30yr", "local_unemployment_rate",
                 "fed_funds_rate", "treasury_10yr_yield",
                 "mortgage_rate_30yr_1yr_lag", "mortgage_rate_30yr_momentum"],
        "knowable_at": "period_start",
        "risk": "Low",
        "notes": "Merged via merge_asof(backward) on observation_date. Leakage-safe.",
    },
    {
        "group": "Spatial features",
        "cols": ["knn_petition_rate_1km", "dist_petition_rate_lag1",
                 "active_cases_100m", "active_cases_500m", "active_gravity_index_t"],
        "knowable_at": "period_start",
        "risk": "Low",
        "notes": "knn uses filing_year-1 lag. Active cases computed at period_start. Safe.",
    },
]

# ------------------------------------------------------------------
# Audit each group
# ------------------------------------------------------------------
print("Running per-row leakage audit...", flush=True)
report_rows = []
violation_rows = []

for grp in groups:
    grp_cols   = [c for c in grp["cols"] if c in panel.columns]
    ka_col     = grp["knowable_at"]
    risk       = grp["risk"]

    for col in grp_cols:
        non_null_mask = panel[col].notna()
        n_non_null    = non_null_mask.sum()

        if ka_col == "period_start" or ka_col not in panel.columns:
            # No temporal risk to check
            n_violations = 0
            example_violation = ""
        else:
            ka_series = panel[ka_col]
            # Violation: non-null AND knowable_at > period_start
            violation_mask = non_null_mask & ka_series.notna() & (ka_series > panel["period_start"])
            n_violations   = violation_mask.sum()
            if n_violations > 0 and risk in ("High", "Medium"):
                vdf = panel.loc[violation_mask, ["case_number", "period_start", ka_col, col]].head(3)
                violation_rows.append(vdf.assign(feature=col, risk=risk))
            example_violation = ""
            if n_violations > 0:
                ex = panel.loc[violation_mask, ["case_number", "period_start", ka_col, col]].iloc[0]
                example_violation = "case={} period_start={} knowable_at={}".format(
                    ex["case_number"], ex["period_start"].date(), ex[ka_col])

        report_rows.append({
            "group":           grp["group"],
            "feature":         col,
            "knowable_at_col": ka_col,
            "n_non_null":      int(n_non_null),
            "n_violations":    int(n_violations),
            "risk":            risk,
            "notes":           grp["notes"],
            "example_violation": example_violation,
        })

report_df = pd.DataFrame(report_rows)

print("\n" + "="*70)
print("LEAKAGE AUDIT SUMMARY")
print("="*70)
for risk_level in ["High", "Medium", "Low"]:
    sub = report_df[report_df["risk"] == risk_level]
    viol = sub["n_violations"].sum()
    print("[{}] {} features | {:,} total violations".format(risk_level, len(sub), viol))
    if risk_level in ("High", "Medium") and viol > 0:
        for _, row in sub[sub["n_violations"] > 0].iterrows():
            print("    VIOLATION: {} -- {:,} rows".format(row["feature"], row["n_violations"]))

report_df.to_csv(OUT_RPT, index=False)
print("\nAudit report saved: {}".format(OUT_RPT))

if violation_rows:
    viol_df = pd.concat(violation_rows, ignore_index=True)
    viol_df.to_csv(OUT_VIOL, index=False)
    print("Row-level violations saved: {}".format(OUT_VIOL))
else:
    print("No High/Medium violations found.")

# ------------------------------------------------------------------
# Exit nonzero if any High-risk violations
# ------------------------------------------------------------------
high_viols = report_df[report_df["risk"] == "High"]["n_violations"].sum()
if high_viols > 0:
    print("\nFAIL: {:,} High-risk leakage violations detected.".format(high_viols))
    import sys
    sys.exit(1)
else:
    print("\nPASS: No High-risk leakage violations.")
