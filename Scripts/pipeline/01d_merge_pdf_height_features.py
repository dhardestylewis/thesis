"""
merge_pdf_height_features.py
------------------------------
Merges pdf_height_features.csv into biweekly_panel.csv using merge_asof.

Leakage contract:
  For each panel row at period_start, only PDF-extracted values whose
  source_date <= period_start are visible. This is enforced by
  merge_asof(direction='backward').

Also merges Final_Zoning / Delta_Approved_Height / Staff_Attrition_Height
from model_ready_zoning_data.csv anchored to Final_Council_Date, so those
columns are only non-null on periods at or after the council vote.
"""
import pandas as pd
import numpy as np
import os

BASE      = r"c:\Users\dhl\data\Thesis\thesis\Data"
PANEL_IN  = r"c:\Users\dhl\data\Thesis\thesis\Data\Panel\biweekly_panel.csv"
PDF_HT    = BASE + r"\interim\pdf_height_features.csv"
MODEL_RDY = BASE + r"\final\model_ready_zoning_data.csv"
PANEL_OUT = PANEL_IN  # overwrite in place

print("Loading panel...", flush=True)
panel = pd.read_csv(PANEL_IN, low_memory=False)
panel["period_start"] = pd.to_datetime(panel["period_start"])
n_start = len(panel)
print("  {:,} rows | {:,} cases".format(n_start, panel["case_number"].nunique()), flush=True)

# ------------------------------------------------------------------
# 1. PDF height features (leakage-safe via merge_asof backward)
# ------------------------------------------------------------------
print("\nMerging PDF height features...", flush=True)
pdf_ht = pd.read_csv(PDF_HT, low_memory=False)
pdf_ht["source_date"] = pd.to_datetime(pdf_ht["source_date"], errors="coerce")
pdf_ht = pdf_ht.dropna(subset=["source_date"]).sort_values(["case_number", "source_date"])

PDF_COLS = [
    "pdf_requested_zoning", "pdf_requested_height_ft", "pdf_requested_max_far", "pdf_proposed_height_ft",
    "pdf_story_count", "pdf_story_height_ft", "pdf_reduced_to_ft",
    "pdf_compatibility_height_ft", "pdf_staff_recommends_ht",
]
# Drop any already-present pdf columns so we can re-merge cleanly
panel = panel.drop(columns=[c for c in PDF_COLS if c in panel.columns], errors="ignore")

# merge_asof requires each side sorted by the on-key only (not by the by-key first)
panel_sorted  = panel.sort_values("period_start")
pdf_ht_sorted = pdf_ht.sort_values("source_date")

merged = pd.merge_asof(
    panel_sorted,
    pdf_ht_sorted[["case_number", "source_date"] + [c for c in PDF_COLS if c in pdf_ht.columns]],
    left_on="period_start",
    right_on="source_date",
    by="case_number",
    direction="backward",
).drop(columns=["source_date"], errors="ignore")


assert len(merged) == n_start, "Row count changed after PDF height merge!"
matched_pdf = merged["pdf_requested_height_ft"].notna().sum()
print("  pdf_requested_height_ft non-null rows: {:,}".format(matched_pdf), flush=True)
for col in PDF_COLS:
    if col in merged.columns:
        n = merged[col].notna().sum()
        print("  {:<42}: {:,} non-null".format(col, n), flush=True)

# ------------------------------------------------------------------
# 2. Post-vote outcome features anchored to Final_Council_Date
#    These are only valid on periods at or after the vote date.
#    Leakage contract: non-null iff period_start >= Final_Council_Date
# ------------------------------------------------------------------
print("\nMerging post-vote outcome features (Final_Zoning, Delta_Approved_Height, Staff_Attrition_Height)...", flush=True)
mrd = pd.read_csv(MODEL_RDY, low_memory=False)

OUTCOME_COLS = [c for c in ["Final_Zoning", "Delta_Approved_Height", "Staff_Attrition_Height",
                              "Delta_Approved_FAR"] if c in mrd.columns]
if OUTCOME_COLS:
    mrd["Final_Council_Date"] = pd.to_datetime(mrd.get("final_date")).fillna(pd.to_datetime(mrd.get("approval_date"))).dt.tz_localize(None)
    outcome_df = mrd[["case_number", "Final_Council_Date"] + OUTCOME_COLS].drop_duplicates("case_number")

    # Drop columns that may already exist
    merged = merged.drop(columns=[c for c in OUTCOME_COLS if c in merged.columns], errors="ignore")
    merged = merged.drop(columns=["Final_Council_Date"], errors="ignore")

    # Merge on case_number to get the vote date and outcome values
    merged = merged.merge(outcome_df, on="case_number", how="left")

    # Zero out outcome values on periods BEFORE the vote
    for col in OUTCOME_COLS:
        if col in merged.columns:
            before_vote = merged["Final_Council_Date"].notna() & (merged["period_start"] < merged["Final_Council_Date"])
            merged.loc[before_vote, col] = np.nan

    # Summary
    for col in OUTCOME_COLS:
        if col in merged.columns:
            n = merged[col].notna().sum()
            print("  {:<42}: {:,} non-null (post-vote only)".format(col, n), flush=True)
else:
    print("  No outcome columns found in model_ready_zoning_data.csv", flush=True)

# ------------------------------------------------------------------
# 2b. Recalculate net_height_change now that pdf_requested_height_ft exists
# ------------------------------------------------------------------
if "pdf_requested_height_ft" in merged.columns:
    initial_req = merged.groupby("case_number")["pdf_requested_height_ft"].transform("max")
    current_constraint = merged[["pdf_requested_height_ft", "pdf_staff_recommends_ht"]].min(axis=1) if "pdf_staff_recommends_ht" in merged.columns else merged["pdf_requested_height_ft"]
    current_constraint = current_constraint.fillna(initial_req)
    final_ht = merged["pdf_reduced_to_ft"].fillna(current_constraint).fillna(0) if "pdf_reduced_to_ft" in merged.columns else current_constraint.fillna(0)
    merged["net_height_change"] = (initial_req - final_ht).clip(lower=0).fillna(0)
    print("  Recalculated net_height_change -> {:,} cases have non-zero reductions".format((merged["net_height_change"] > 0).sum()))

# ------------------------------------------------------------------
# 3. Save
# ------------------------------------------------------------------
assert len(merged) == n_start, "Final row count mismatch!"
merged.to_csv(PANEL_OUT, index=False)
print("\nSaved: {}".format(PANEL_OUT))
print("Shape: {:,} rows x {} cols".format(len(merged), len(merged.columns)))
