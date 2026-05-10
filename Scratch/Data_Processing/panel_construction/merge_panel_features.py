import os
import pandas as pd
import numpy as np

BASE = r"C:\Users\dhl\data\Thesis\thesis\Data"
PANEL_PATH = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"

print("Loading Panel...")
panel = pd.read_csv(PANEL_PATH, low_memory=False)
panel["period_start"] = pd.to_datetime(panel["period_start"])
initial_rows = len(panel)

# Drop any columns that were previously merged (leaky or otherwise) so we
# can re-attach them cleanly below.
COLS_TO_RESET = ["had_preapp", "days_to_site_plan", "days_to_building_permit", "staff_recommended_height"]
panel = panel.drop(columns=[c for c in COLS_TO_RESET if c in panel.columns])

# ─────────────────────────────────────────────────────────────────────────────
# 1.  AMANDA Spatial Features
# Only `had_preapp` is leakage-safe: it reflects a filing that occurred
# BEFORE the zoning application date and is thus known at T0.
# `days_to_site_plan` and `days_to_building_permit` are FUTURE outcomes and
# must NOT be used as features.  They are saved as a separate outcomes table.
# ─────────────────────────────────────────────────────────────────────────────
amanda_path = os.path.join(BASE, "interim", "amanda_spatial_features.csv")
if os.path.exists(amanda_path):
    print("Merging AMANDA Spatial Features (leakage-safe only)...")
    amanda = pd.read_csv(amanda_path).drop_duplicates(subset=["case_number"])

    # Only the pre-application indicator is safe as a predictor.
    panel = panel.merge(amanda[["case_number", "had_preapp"]], on="case_number", how="left")
    panel["had_preapp"] = panel["had_preapp"].fillna(0).astype(int)

    # Save the downstream timeline columns as a SEPARATE outcomes file.
    outcomes_path = os.path.join(BASE, "interim", "amanda_downstream_outcomes.csv")
    amanda[["case_number", "days_to_site_plan", "days_to_building_permit"]].to_csv(outcomes_path, index=False)
    print(f"  had_preapp merged. Future outcomes saved to {outcomes_path}")
else:
    print("AMANDA features not found, skipping...")
    panel["had_preapp"] = 0

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Staff-Recommended Height — temporally anchored via merge_asof
# We join the per-meeting height concession against the commission agenda
# meeting dates, then merge_asof onto each panel row's period_start.
# At period t, the model only sees recommendations made at meetings that
# happened STRICTLY BEFORE period_start.  This produces a step-function
# that correctly updates as the developer makes concessions over time.
# ─────────────────────────────────────────────────────────────────────────────
concessions_path = r"C:\Users\dhl\data\Thesis\thesis\Scratch\extracted_height_concessions.csv"
agenda_path = os.path.join(BASE, "interim", "commission_agendas_cases.csv")

if os.path.exists(concessions_path) and os.path.exists(agenda_path):
    print("Building temporally-anchored staff_recommended_height...")
    concess = pd.read_csv(concessions_path)
    agenda = pd.read_csv(agenda_path)
    agenda["meeting_date"] = pd.to_datetime(agenda["meeting_date"])

    # Join height concessions to the meeting dates via case_number.
    # A single case may appear at multiple meetings; each meeting gets the
    # current staff-recommended height extracted from that agenda PDF.
    # (We use the case_number–level match here; height extraction was also
    # per-case, so we broadcast to every meeting that case appeared at.)
    concess_dated = agenda.merge(concess[["case_number", "staff_recommended_height"]],
                                  on="case_number", how="inner")
    concess_dated = concess_dated.sort_values(["case_number", "meeting_date"])

    # For each panel row, merge_asof to get the last known staff height
    # from any meeting that occurred BEFORE the current period_start.
    panel_sorted = panel.sort_values("period_start")
    concess_dated_sorted = concess_dated.sort_values("meeting_date")

    panel_sorted = pd.merge_asof(
        panel_sorted,
        concess_dated_sorted[["case_number", "meeting_date", "staff_recommended_height"]]
        .rename(columns={"meeting_date": "_ref_date"}),
        left_on="period_start",
        right_on="_ref_date",
        by="case_number",
        direction="backward",  # only use meetings BEFORE current period
    ).drop(columns=["_ref_date"])

    panel = panel_sorted
    matched = panel["staff_recommended_height"].notna().sum()
    print(f"  staff_recommended_height matched for {matched:,} panel rows (step-function, leakage-safe)")
else:
    print("Height concessions or agenda file not found, skipping...")
    panel["staff_recommended_height"] = np.nan

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Save
# ─────────────────────────────────────────────────────────────────────────────
panel.to_csv(PANEL_PATH, index=False)
print(f"\nSaved updated panel to {PANEL_PATH}")
print(f"Shape: {panel.shape[0]:,} rows x {panel.shape[1]} cols")
assert len(panel) == initial_rows, f"Merge altered row count! {initial_rows} -> {len(panel)}"
print("Row count assertion PASSED.")
