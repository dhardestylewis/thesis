import pandas as pd
import numpy as np
import os

with open(r'c:\Users\dhl\data\Thesis\thesis\Scratch\Data_Processing\build_biweekly_panel.py', 'r', encoding='utf-8') as f:
    src = f.read()

# 1. EARS fix
src = src.replace(
    'unique_cy = panel[["case_number","parcel_id_10","ears_account_number","year"]].drop_duplicates()',
    'panel["vintage_year_ears"] = panel["year"] - 1\nunique_cy = panel[["case_number","parcel_id_10","ears_account_number","vintage_year_ears"]].drop_duplicates()'
)
src = src.replace(
    'for yr, grp in unique_cy.groupby("year"):',
    'for yr, grp in unique_cy.groupby("vintage_year_ears"):'
)
src = src.replace(
    'chunks.append(m[["case_number","year"]+feats])',
    'chunks.append(m[["case_number","vintage_year_ears"]+feats])'
)
src = src.replace(
    'pj = pd.concat(chunks).drop_duplicates(["case_number","year"])',
    'pj = pd.concat(chunks).drop_duplicates(["case_number","vintage_year_ears"])'
)
src = src.replace(
    'panel = panel.merge(pj, on=["case_number","year"], how="left")',
    'panel = panel.merge(pj, on=["case_number","vintage_year_ears"], how="left")'
)

# 2. ACS fix
src = src.replace(
    'panel = panel.merge(acs_sub, left_on=["census_tract","year"],\n                        right_on=["census_tract","acs_year"], how="left")',
    'panel["vintage_year_acs"] = panel["year"] - 1\n    panel = panel.merge(acs_sub, left_on=["census_tract","vintage_year_acs"],\n                        right_on=["census_tract","acs_year"], how="left")'
)

# 3. Petition fix
import re
pet_pattern = r'# ── 8\. Petition events ────────────────────────────────────────────────────.*?# Spatial petition lag'
replacement = """# ── 8. Petition events ────────────────────────────────────────────────────
print("Step 7: Petition events...")
pet = pd.read_csv(PETITION_CSV, low_memory=False)
pet["date"] = pd.to_datetime(pet["date"], format="mixed", errors="coerce")
pet = pet.dropna(subset=["date", "area_pct"])

panel_periods = panel[["case_number", "period_start", "period_end"]].copy()
pet_merged = pet.merge(panel_periods, on="case_number", how="inner")
pet_matched = pet_merged[(pet_merged["date"] >= pet_merged["period_start"]) & (pet_merged["date"] <= pet_merged["period_end"])]

pet_period_agg = pet_matched.groupby(["case_number", "period_start"]).agg(
    petition_count_this_period=("signed", "sum"),
    petition_pct_this_period=("area_pct", "sum")
).reset_index()

panel = panel.merge(pet_period_agg, on=["case_number", "period_start"], how="left")
panel["petition_event"] = panel["petition_pct_this_period"].notna().astype(int)
print(f"  Petition events: {panel['petition_event'].sum()} | pct non-null: {panel['petition_pct_this_period'].notna().sum()}")

pet_case_agg = pet.groupby("case_number")["area_pct"].sum().reset_index().rename(columns={"area_pct": "label_petition_total_pct"})
panel = panel.merge(pet_case_agg, on="case_number", how="left")
panel["label_valid_protest"] = (panel["label_petition_total_pct"] >= 20).astype(int)

pet_first = pet.groupby("case_number")["date"].min().reset_index()
pet_first["petition_year"] = pet_first["date"].dt.year
pet_first["petition_quarter"] = pet_first["date"].dt.quarter
panel = panel.merge(pet_first[["case_number", "petition_year", "petition_quarter"]], on="case_number", how="left")

# Spatial petition lag"""
src = re.sub(pet_pattern, replacement, src, flags=re.DOTALL)

# 4. Remove leaky static outcome fields from OUT_COLS
src = src.replace(
    '"proposed_max_height_ft","proposed_max_far","proposed_max_bldg_cov_pct",',
    '# "proposed_max_height_ft","proposed_max_far","proposed_max_bldg_cov_pct", # REMOVED: 2026 snapshot leakage'
)
src = src.replace(
    '"existing_max_height_ft","existing_max_far","existing_max_bldg_cov_pct",',
    '# "existing_max_height_ft","existing_max_far","existing_max_bldg_cov_pct", # REMOVED: 2026 snapshot leakage'
)
src = src.replace(
    '"label_petition_total_pct","label_valid_protest","petition_year","petition_quarter",',
    '# "label_petition_total_pct","label_valid_protest","petition_year","petition_quarter", # REMOVED: static/future labels, excluded from features'
)

with open(r'c:\Users\dhl\data\Thesis\thesis\Scratch\Data_Processing\build_biweekly_panel.py', 'w', encoding='utf-8') as f:
    f.write(src)
print("Patch applied.")
