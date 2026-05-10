"""
build_biweekly_panel.py
One row per (case_number x 2-week hearing slot).
Features: thesis panel (EARS/LDB, ACS, council, petition) +
          Properlytic paradigms (macro lags, velocity, KNN spatial lag,
          cyclical encoding, ratio engineering, target encoding).
"""
import os
import numpy as np
import pandas as pd

BASE      = r"c:\Users\dhl\data\Thesis\thesis\Data"
OUT_DIR   = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
CUTOFF    = pd.Timestamp("2026-05-02")
MAX_DAYS  = 3650   # 10-yr cap

ZONING_CSV    = os.path.join(BASE, "final", "model_ready_zoning_data.csv")
ENRICHED_CSV  = os.path.join(BASE, "Zoning_Cases","Processed_Data","CSV","enriched_zoning_data_causal.csv")
PETITION_CSV  = os.path.join(BASE, "Protest_Petitions","petition_signers_backfilled.csv")
COUNCIL_CSV   = os.path.join(BASE, "interim", "council_agendas_cases.csv")
LDB16_CSV     = os.path.join(BASE, "CoA_Open_Data","LDB_2016_4nsn-uea6.csv")
LDB21_CSV     = os.path.join(BASE, "CoA_Open_Data","LDB_2021_kk8y-6cmt.csv")
CROSSWALK_CSV = os.path.join(BASE, "Panel","Reference","id_crosswalk.csv")
ACS_CSV       = os.path.join(BASE, "Panel","census","acs_tract_timeseries.csv")
GEOID_CSV     = os.path.join(BASE, "Panel","geo","case_geoid_lookup.csv")
EARS_DIR      = os.path.join(BASE, "Panel","Intermediate")

def safe_pid(x):
    try: return str(int(float(x))).zfill(10)
    except: return None

# ── 1. Zoning cases ────────────────────────────────────────────────────────
print("Step 1: Zoning cases...")
z = pd.read_csv(ZONING_CSV, low_memory=False)
for c in ["App_Date","Final_Council_Date","final_date","approval_date"]:
    if c in z.columns: z[c] = pd.to_datetime(z[c], errors="coerce")
z = z[z["case_number"].notna() & z["App_Date"].notna()].copy()
z["T0"]     = z["App_Date"]
z["T_vote"] = z["Final_Council_Date"].fillna(z.get("final_date")).fillna(z.get("approval_date"))
z["censored"] = z["T_vote"].isna().astype(int)
# Right-censor: unresolved cases get 2yr observation window, not full CUTOFF
CENSOR_WINDOW = 730  # 2 years max for unresolved cases in skeleton
if "label_real_days_in_pipeline" in z.columns:
    z["label_real_days_in_pipeline"] = z["label_real_days_in_pipeline"].fillna(
        (CUTOFF - z["T0"]).dt.days)
else:
    z["label_real_days_in_pipeline"] = np.where(
        z["T_vote"].notna(), (z["T_vote"]-z["T0"]).dt.days,
        (CUTOFF - z["T0"]).dt.days)
# Skeleton end: resolved cases use vote date; unresolved use min(CENSOR_WINDOW, MAX_DAYS)
z["T_end"] = np.where(
    z["T_vote"].notna(),
    np.minimum(z["T_vote"], z["T0"] + pd.Timedelta(days=MAX_DAYS)),
    z["T0"] + pd.Timedelta(days=CENSOR_WINDOW))
z["T_end"] = pd.to_datetime(z["T_end"])
z["parcel_id_10"] = z["parcel_id_10"].map(safe_pid) if "parcel_id_10" in z.columns else np.nan
z["filing_year"]  = z["T0"].dt.year

n_cens = z["censored"].sum()
print(f"  {len(z)} cases | {n_cens} right-censored")

# ── 2. Skeleton ────────────────────────────────────────────────────────────
print("Step 2: Building skeleton...")
rows = []
for _, r in z.iterrows():
    periods = pd.date_range(r["T0"], r["T_end"], freq="2W")
    if len(periods) == 0: periods = pd.DatetimeIndex([r["T0"]])
    for i, ps in enumerate(periods):
        pe = ps + pd.Timedelta(days=13)
        rows.append({
            "case_number": r["case_number"],
            "period_start": ps, "period_end": pe,
            "period_seq": i+1,
            "year": ps.year, "quarter": ps.quarter,
            "bw_sin": np.sin(2*np.pi*(ps.isocalendar()[1]/52)),
            "bw_cos": np.cos(2*np.pi*(ps.isocalendar()[1]/52)),
            "filing_event": int(i==0),
            "vote_event": int(
                pd.notna(r["T_vote"]) and ps <= r["T_vote"] <= pe),
            "resolved": int(
                pd.notna(r["T_vote"]) and ps <= r["T_vote"]),
        })
panel = pd.DataFrame(rows)
print(f"  {len(panel):,} rows | {panel['case_number'].nunique():,} cases | "
      f"{len(panel)/panel['case_number'].nunique():.1f} periods/case")

# ── 3. Case metadata ───────────────────────────────────────────────────────
CASE_COLS = [c for c in [
    "case_number","parcel_id_10","latitude","longitude","shape_area",
    "council_district","T0","T_vote","censored","filing_year",
    "label_real_days_in_pipeline","Remand_Count","Council_Appearances",
    "Aggregate_Sentiment","label_valid_petition_pct",
    "Delta_Approved_Height","Delta_Approved_FAR","Staff_Attrition_Height",
    # zoning flags from thesis
    "is_pud","is_tod","is_npa","owner_initiated","hb24_eligible",
] if c in z.columns]
panel = panel.merge(z[CASE_COLS].drop_duplicates("case_number"),
                    on="case_number", how="left")

# ── 4. Parcel data (annual, forward-filled) ────────────────────────────────
print("Step 3-4: Parcel data...")
EARS_COLS = ["account_number","total_market_value","appraised_value",
             "land_market_value","improvement_market_value",
             "land_acres","year_built","improvement_sq_ft",
             "exemption_flag_hs","homesite_flag","land_use_code",
             "deed_acreage","most_recent_sale_date"]
ears = {}
for yr in range(2019, 2026):
    fp = os.path.join(EARS_DIR, f"ears_{yr}_clean.csv")
    if not os.path.exists(fp): continue
    avail = pd.read_csv(fp, nrows=0).columns.tolist()
    cols  = [c for c in EARS_COLS if c in avail]
    df = pd.read_csv(fp, low_memory=False, usecols=cols)
    df = df.rename(columns={"total_market_value":"market_value",
                             "year_built":"yr_built",
                             "deed_acreage":"land_acres_deed"})
    ears[yr] = df

LDB16 = pd.read_csv(LDB16_CSV, low_memory=False,
    usecols=["PID_10","MARKET_VAL","APPRAISED_VAL","LAND_ACRES","YR_BUILT"]).rename(
    columns={"PID_10":"parcel_id_10","MARKET_VAL":"market_value",
             "APPRAISED_VAL":"appraised_value","LAND_ACRES":"land_acres","YR_BUILT":"yr_built"})
LDB16["parcel_id_10"] = LDB16["parcel_id_10"].map(safe_pid)
LDB16["data_year"] = 2016

LDB21 = pd.read_csv(LDB21_CSV, low_memory=False,
    usecols=["PID_10","MARKET_VAL","ASSESSED_V","LAND_ACRES","YR_BUILT"]).rename(
    columns={"PID_10":"parcel_id_10","MARKET_VAL":"market_value",
             "ASSESSED_V":"appraised_value","LAND_ACRES":"land_acres","YR_BUILT":"yr_built"})
LDB21["parcel_id_10"] = LDB21["parcel_id_10"].map(safe_pid)
LDB21["data_year"] = 2021

xwalk = pd.read_csv(CROSSWALK_CSV, low_memory=False)
xwalk["parcel_id_10"] = xwalk["parcel_id_10"].astype(str).str.zfill(10)
panel = panel.merge(xwalk[["parcel_id_10","ears_account_number"]].drop_duplicates(),
                    on="parcel_id_10", how="left")

PARCEL_FEATS = ["market_value","appraised_value","land_acres","yr_built",
                "land_market_value","improvement_market_value",
                "improvement_sq_ft","exemption_flag_hs","homesite_flag","land_use_code"]

unique_cy = panel[["case_number","parcel_id_10","ears_account_number","year"]].drop_duplicates()
chunks = []
for yr, grp in unique_cy.groupby("year"):
    if yr >= 2019 and yr in ears:
        ref = ears[yr]; ref_type = "ears"
    elif yr >= 2019 and ears:
        ref = ears[min(ears.keys(), key=lambda k: abs(k-yr))]; ref_type = "ears"
    elif yr >= 2017:
        ref = LDB21; ref_type = "ldb"
    else:
        ref = LDB16; ref_type = "ldb"

    feats = [f for f in PARCEL_FEATS if f in ref.columns]
    if ref_type == "ears":
        sub = grp.dropna(subset=["ears_account_number"]).copy()
        sub["ears_account_number"] = sub["ears_account_number"].astype(str)
        ref["account_number"]      = ref["account_number"].astype(str)
        m = sub.merge(ref[["account_number"]+feats],
                      left_on="ears_account_number",right_on="account_number",how="left")
    else:
        sub = grp.dropna(subset=["parcel_id_10"])
        m = sub.merge(ref[["parcel_id_10"]+feats], on="parcel_id_10", how="left")
    chunks.append(m[["case_number","year"]+feats])

if chunks:
    pj = pd.concat(chunks).drop_duplicates(["case_number","year"])
    panel = panel.merge(pj, on=["case_number","year"], how="left")
    print(f"  Parcel joined: {pj['case_number'].nunique():,} cases")

# ── 5. Engineered parcel ratios (Properlytic paradigm) ────────────────────
# Coerce EARS numeric cols that may arrive as strings
for _nc in ["market_value","appraised_value","land_acres","yr_built",
             "land_market_value","improvement_market_value","improvement_sq_ft",
             "exemption_flag_hs","homesite_flag"]:
    if _nc in panel.columns:
        panel[_nc] = pd.to_numeric(panel[_nc], errors="coerce")
if "yr_built" in panel.columns:
    panel["building_age"] = panel["year"] - panel["yr_built"]
if all(c in panel.columns for c in ["improvement_sq_ft","land_acres"]):
    panel["land_to_building_ratio"] = (panel["land_acres"]*43560) / panel["improvement_sq_ft"].replace(0,np.nan)
if all(c in panel.columns for c in ["improvement_market_value","market_value"]):
    panel["improvement_ratio"] = panel["improvement_market_value"] / panel["market_value"].replace(0,np.nan)

# ── 6. ACS demographics (annual, forward-filled) ──────────────────────────
print("Step 5: ACS demographics...")
try:
    acs = pd.read_csv(ACS_CSV, low_memory=False)
    geoid = pd.read_csv(GEOID_CSV, low_memory=False)
    geoid = geoid.rename(columns={"case_id":"case_number", "geoid_tract":"census_tract"})
    geoid["census_tract"] = geoid["census_tract"].astype(str)
    panel = panel.merge(geoid[["case_number","census_tract"]].drop_duplicates("case_number"),
                        on="case_number", how="left")

    ACS_FEATS = ["geoid_tract","vintage","total_population","median_household_income",
                 "median_gross_rent","owner_occupied_units","renter_occupied_units",
                 "race_white","race_black","race_hispanic","race_asian",
                 "median_age","poverty_count","median_home_value","total_housing_units"]
    acs_sub = acs[[c for c in ACS_FEATS if c in acs.columns]].copy()
    acs_sub = acs_sub.rename(columns={"geoid_tract":"census_tract","vintage":"acs_year"})
    acs_sub["census_tract"] = acs_sub["census_tract"].astype(str)
    panel = panel.merge(acs_sub, left_on=["census_tract","year"],
                        right_on=["census_tract","acs_year"], how="left")
    if all(c in panel.columns for c in ["renter_occupied_units","owner_occupied_units"]):
        tot = panel["renter_occupied_units"]+panel["owner_occupied_units"]
        panel["renter_share"] = panel["renter_occupied_units"] / tot.replace(0,np.nan)
        panel["owner_share"]  = panel["owner_occupied_units"]  / tot.replace(0,np.nan)
    if all(c in panel.columns for c in ["median_gross_rent","median_household_income"]):
        panel["rent_burden"] = (panel["median_gross_rent"]*12) / panel["median_household_income"].replace(0,np.nan)
    if all(c in panel.columns for c in ["market_value","median_household_income"]):
        panel["affordability_proxy"] = panel["market_value"] / panel["median_household_income"].replace(0,np.nan)
    print(f"  ACS joined for {panel['census_tract'].notna().sum()} rows")
except Exception as e:
    print(f"  ACS skipped: {e}")

# ── 7. FRED macro (annual) ─────────────────────────────────────────────────
# ── 7. FRED macro (high-frequency) ─────────────────────────────────────────
print("Step 6: FRED macro...")
FRED_PATH = os.path.join(BASE, "Panel", "macro", "fred_timeseries.csv")
if os.path.exists(FRED_PATH):
    fred = pd.read_csv(FRED_PATH)
    fred["observation_date"] = pd.to_datetime(fred["observation_date"])
    fred = fred.sort_values("observation_date")
    panel = panel.sort_values("period_start")
    
    # Forward-fill merge macro onto the exact period_start
    panel = pd.merge_asof(panel, fred, left_on="period_start", right_on="observation_date", direction="backward")
    
    fred_cols = [c for c in fred.columns if c != "observation_date"]
    
    # 1-year momentum (Velocity)
    # We create a 1-year shifted version of fred and merge it again
    fred_lag = fred.copy()
    fred_lag["observation_date"] = fred_lag["observation_date"] + pd.DateOffset(years=1)
    fred_lag = fred_lag.rename(columns={c: f"{c}_1yr_lag" for c in fred_cols})
    fred_lag = fred_lag.dropna(subset=[c for c in fred_lag.columns if "_1yr_lag" in c], how="all")
    fred_lag = fred_lag.sort_values("observation_date")
    
    panel = pd.merge_asof(panel, fred_lag, left_on="period_start", right_on="observation_date", direction="backward")
    
    for c in fred_cols:
        if f"{c}_1yr_lag" in panel.columns:
            panel[f"{c}_momentum"] = panel[c] - panel[f"{c}_1yr_lag"]
            
    print(f"  FRED macro joined via merge_asof: {fred_cols}")
else:
    print("  FRED not found, skipping")

# ── 8. Petition events ────────────────────────────────────────────────────
print("Step 7: Petition events...")
pet = pd.read_csv(PETITION_CSV, low_memory=False)
pet["date"] = pd.to_datetime(pet["date"], format="mixed", errors="coerce")
pet_agg = pet.groupby("case_number").agg(
    petition_date         =("date","min"),
    petition_signer_count =("signed","sum"),
    label_petition_total_pct    =("area_pct","sum"),
).reset_index()

# Inject Recovered Petitions
REC_PATH = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\recovered_petitions.csv"
rec_df = pd.read_csv(REC_PATH)
rec_agg = rec_df.groupby("case_number").agg(
    petition_signer_count=("signed", "sum"),
    label_petition_total_pct=("area_pct", "sum")
).reset_index()

# Merge recovered cases into pet_agg
for _, row in rec_agg.iterrows():
    if row["case_number"] not in pet_agg["case_number"].values:
        new_row = pd.DataFrame([{
            "case_number": row["case_number"],
            "petition_date": pd.NaT,
            "petition_signer_count": row["petition_signer_count"],
            "label_petition_total_pct": row["label_petition_total_pct"]
        }])
        pet_agg = pd.concat([pet_agg, new_row], ignore_index=True)

# Impute missing or completely out-of-bounds petition_date using T_end - 1 day
pet_agg = pet_agg.merge(z[["case_number", "T0", "T_end"]].drop_duplicates(), on="case_number", how="left")

# Condition 1: NaT
cond_missing = pet_agg["petition_date"].isna()
# Condition 2: Way outside the window (OCR garbage)
cond_oob = (pet_agg["petition_date"] < pet_agg["T0"]) | (pet_agg["petition_date"] > pet_agg["T_end"])

pet_agg.loc[cond_missing | cond_oob, "petition_date"] = pet_agg.loc[cond_missing | cond_oob, "T_end"] - pd.Timedelta(days=1)

pet_agg["label_valid_protest"]    = (pet_agg["label_petition_total_pct"] >= 20).astype(int)
pet_agg["petition_year"]    = pet_agg["petition_date"].dt.year
pet_agg["petition_quarter"] = pet_agg["petition_date"].dt.quarter
panel = panel.merge(pet_agg, on="case_number", how="left")
panel["label_valid_protest"] = panel["label_valid_protest"].fillna(0).astype(int)
panel["petition_event"] = (
    panel["petition_date"].notna() &
    (panel["petition_date"] >= panel["period_start"]) &
    (panel["petition_date"] <= panel["period_end"])
).astype(int)
panel["petition_pct_this_period"]   = np.where(panel["petition_event"]==1, panel["label_petition_total_pct"],  np.nan)
panel["petition_count_this_period"] = np.where(panel["petition_event"]==1, panel["petition_signer_count"], np.nan)
print(f"  Petition events: {panel['petition_event'].sum()} | pct non-null: {panel['petition_pct_this_period'].notna().sum()}")

# Spatial petition lag: historical petition rate by council district
# (expanding window, lag-1 to prevent leakage — Properlytic target-encoding paradigm)
if "council_district" in panel.columns:
    case_pet = z[["case_number","filing_year","council_district"]].drop_duplicates("case_number")
    case_pet = case_pet.merge(pet_agg[["case_number","label_valid_protest"]],
                               on="case_number", how="left")
    case_pet["label_valid_protest"] = case_pet["label_valid_protest"].fillna(0)
    dist_yr = (case_pet.groupby(["council_district","filing_year"])["label_valid_protest"]
               .mean().reset_index(name="dist_petition_rate"))
    dist_yr.sort_values(["council_district","filing_year"], inplace=True)
    dist_yr["dist_petition_rate_lag1"] = (
        dist_yr.groupby("council_district")["dist_petition_rate"]
        .transform(lambda x: x.shift(1).expanding().mean()))
    panel = panel.merge(
        dist_yr[["council_district","filing_year","dist_petition_rate_lag1"]],
        left_on=["council_district","year"], right_on=["council_district","filing_year"],
        how="left")

# ── 9. Council hearing appearances ────────────────────────────────────────
print("Step 8: Council hearings...")
try:
    cdf = pd.read_csv(COUNCIL_CSV, low_memory=False)
    # Meeting_Date format: "October 1, 2009  Austin City Council..."
    # Extract just the date portion before double-space or em-dash
    date_str = cdf["Meeting_Date"].str.extract(r'^([A-Za-z]+ \d+,?\s+\d{4})')[0].str.strip()
    cdf["meeting_date"] = pd.to_datetime(date_str, format='mixed', errors="coerce")
    cdf = cdf.rename(columns={"Case_Number":"case_number"})[["case_number","meeting_date"]].dropna()
    print(f"  Council rows loaded: {len(cdf)}, date parse rate: {cdf['meeting_date'].notna().mean()*100:.0f}%")
    merged_c = panel[["case_number","period_start","period_end"]].merge(cdf, on="case_number", how="left")
    merged_c["in_period"] = (
        (merged_c["meeting_date"] >= merged_c["period_start"]) &
        (merged_c["meeting_date"] <= merged_c["period_end"]))
    counts = (merged_c[merged_c["in_period"]]
              .groupby(["case_number","period_start"]).size()
              .reset_index(name="council_hearings_this_period"))
    panel = panel.merge(counts, on=["case_number","period_start"], how="left")
    panel["council_hearings_this_period"] = panel["council_hearings_this_period"].fillna(0).astype(int)
    
    # Engineer Cumulative History to resolve sparsity
    panel["cumulative_council_hearings"] = panel.groupby("case_number")["council_hearings_this_period"].transform(lambda x: x.cumsum().shift(1).fillna(0))
    # Shift petition cumulatives by 1 to avoid contemporaneous leakage:
    # at period t, these reflect history up to t-1, not including the current period's event.
    panel["cumulative_petition_events"]  = panel.groupby("case_number")["petition_event"].transform(lambda x: x.cumsum().shift(1).fillna(0))
    panel["cumulative_petition_count"]   = panel.groupby("case_number")["petition_count_this_period"].transform(lambda x: x.fillna(0).cumsum().shift(1).fillna(0))
    panel["cumulative_petition_pct"]     = panel.groupby("case_number")["petition_pct_this_period"].transform(lambda x: x.fillna(0).cumsum().shift(1).fillna(0))
    
    print(f"  Hearings matched: {panel['council_hearings_this_period'].sum():.0f} total")
except Exception as e:
    print(f"  Council hearings skipped: {e}")
    panel["council_hearings_this_period"] = 0
    panel["cumulative_council_hearings"] = 0
    panel["cumulative_petition_events"] = panel.groupby("case_number")["petition_event"].transform(lambda x: x.cumsum().shift(1).fillna(0))
    panel["cumulative_petition_count"] = panel.groupby("case_number")["petition_count_this_period"].transform(lambda x: x.fillna(0).cumsum().shift(1).fillna(0))
    panel["cumulative_petition_pct"] = panel.groupby("case_number")["petition_pct_this_period"].transform(lambda x: x.fillna(0).cumsum().shift(1).fillna(0))

# ── 8b. Commission hearings ──────────────────────────────────────────────
print("Step 8b: Commission hearings...")
try:
    COMMISSION_CSV = os.path.join(BASE, "interim", "commission_agendas_cases.csv")
    comm = pd.read_csv(COMMISSION_CSV)
    comm["meeting_date"] = pd.to_datetime(comm["meeting_date"])
    
    merged_comm = panel[["case_number","period_start","period_end"]].merge(comm, on="case_number", how="left")
    merged_comm["in_period"] = (
        (merged_comm["meeting_date"] >= merged_comm["period_start"]) &
        (merged_comm["meeting_date"] <= merged_comm["period_end"]))
    
    counts_comm = (merged_comm[merged_comm["in_period"]]
              .groupby(["case_number","period_start"]).size()
              .reset_index(name="commission_hearings_this_period"))
    
    panel = panel.merge(counts_comm, on=["case_number","period_start"], how="left")
    panel["commission_hearings_this_period"] = panel["commission_hearings_this_period"].fillna(0).astype(int)
    
    panel["cumulative_commission_hearings"] = panel.groupby("case_number")["commission_hearings_this_period"].transform(lambda x: x.cumsum().shift(1).fillna(0))
    
    print(f"  Commission hearings matched: {panel['commission_hearings_this_period'].sum():.0f} total")
except Exception as e:
    print(f"  Commission hearings skipped: {e}")
    panel["commission_hearings_this_period"] = 0
    panel["cumulative_commission_hearings"] = 0

# ── 10. Dimensional features ────────────────────────────────
print("Step 9: Dimensional features...")
try:
    enr = pd.read_csv(ENRICHED_CSV, low_memory=False, usecols=[
        "case_number","proposed_max_height_ft","proposed_max_far",
        "existing_max_height_ft","existing_max_far",
        "proposed_max_bldg_cov_pct","existing_max_bldg_cov_pct",
    ]).drop_duplicates("case_number")
    panel = panel.merge(enr, on="case_number", how="left")
    print(f"  Enriched: {enr['case_number'].nunique()} cases")
except Exception as e:
    print(f"  Enriched skipped: {e}")
    for c in ["proposed_max_height_ft","proposed_max_far","existing_max_height_ft","existing_max_far","proposed_max_bldg_cov_pct","existing_max_bldg_cov_pct"]:
        panel[c] = np.nan

# ── 11. Spatial KNN petition lag ──────────────────────────────────────────
print("Step 10: KNN spatial petition lag...")
try:
    from sklearn.neighbors import BallTree
    case_coords = z[["case_number","T0"]].copy()
    case_coords["filing_year"] = case_coords["T0"].dt.year
    coords_full = (z[["case_number","latitude","longitude","filing_year"]]
                    .merge(pet_agg[["case_number","label_valid_protest"]], on="case_number", how="left"))
    coords_full["label_valid_protest"] = coords_full["label_valid_protest"].fillna(0)
    coords_full = coords_full.dropna(subset=["latitude","longitude"])
    knn_out = []
    for yr in sorted(coords_full["filing_year"].dropna().unique()):
        curr = coords_full[coords_full["filing_year"]==yr].copy()
        prev = coords_full[coords_full["filing_year"]<yr].dropna(subset=["latitude","longitude"])
        if len(prev) > 0 and len(curr) > 0:
            tree = BallTree(np.radians(prev[["latitude","longitude"]].values), metric="haversine")
            k = min(10, len(prev))
            _, idx = tree.query(np.radians(curr[["latitude","longitude"]].values), k=k)
            curr["knn_petition_rate_1km"] = prev["label_valid_protest"].values[idx].mean(axis=1)
        else:
            curr["knn_petition_rate_1km"] = np.nan
        knn_out.append(curr[["case_number","knn_petition_rate_1km"]])
    if knn_out:
        knn_df = pd.concat(knn_out).drop_duplicates("case_number")
        panel = panel.merge(knn_df, on="case_number", how="left")
        print(f"  KNN lag: {panel['knn_petition_rate_1km'].notna().sum():,} rows matched")
except Exception as e:
    print(f"  KNN skipped: {e}")
    panel["knn_petition_rate_1km"] = np.nan

# ── 12. Output ────────────────────────────────────────────────────────────
print("\nStep 11: Writing output...")
OUT_COLS = [
    # Index
    "case_number","period_start","period_seq","year","quarter",
    "bw_sin","bw_cos",          # cyclical encoding of bi-weekly slot
    # Events
    "filing_event","petition_event","vote_event","resolved","censored",
    # Dimensional features
    "proposed_max_height_ft","proposed_max_far","proposed_max_bldg_cov_pct",
    "existing_max_height_ft","existing_max_far","existing_max_bldg_cov_pct",
    # Petition detail
    "petition_pct_this_period","petition_count_this_period",
    "label_petition_total_pct","label_valid_protest","petition_year","petition_quarter",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct",
    # Council/Commission activity
    "council_hearings_this_period", "cumulative_council_hearings",
    "commission_hearings_this_period", "cumulative_commission_hearings",
    # Parcel (annual forward-fill)
    "market_value","appraised_value","land_acres","yr_built",
    "land_market_value","improvement_market_value","improvement_sq_ft",
    "exemption_flag_hs","homesite_flag","land_use_code",
    # Parcel ratios
    "building_age","land_to_building_ratio","improvement_ratio",
    # ACS (annual forward-fill)
    "total_population","median_household_income","median_gross_rent",
    "renter_share","owner_share","rent_burden","affordability_proxy",
    "race_white","race_black","race_hispanic","median_age",
    # FRED macro
    "mortgage_rate_30yr","local_unemployment_rate",
    "fed_funds_rate","treasury_10yr_yield",
    "mortgage_rate_30yr_1yr_lag","local_unemployment_rate_1yr_lag",
    "fed_funds_rate_1yr_lag","treasury_10yr_yield_1yr_lag",
    "mortgage_rate_30yr_momentum","local_unemployment_rate_momentum",
    "fed_funds_rate_momentum","treasury_10yr_yield_momentum",
    # Spatial
    "latitude","longitude","shape_area","council_district","census_tract",
    "knn_petition_rate_1km","dist_petition_rate_lag1",
    # Case flags
    "filing_year","is_pud","is_tod","is_npa","owner_initiated","hb24_eligible",
    # Outcomes (constant per case)
    "label_real_days_in_pipeline","Remand_Count",
    "Delta_Approved_Height","Delta_Approved_FAR","Staff_Attrition_Height",
    "Aggregate_Sentiment","label_valid_petition_pct",
]
out = panel[[c for c in OUT_COLS if c in panel.columns]].copy()
out_path = os.path.join(OUT_DIR, "biweekly_panel.csv")
out.to_csv(out_path, index=False)
sz = os.path.getsize(out_path)/1e6
print(f"\nSaved: {out_path}")
print(f"Shape: {out.shape[0]:,} rows x {out.shape[1]} cols | {sz:.1f} MB")
print(f"Cases: {out['case_number'].nunique():,}")
print("\nMissingness (top 20%):")
miss = (out.isnull().mean()*100).round(1).sort_values(ascending=False)
print(miss[miss>0].head(20).to_string())
