import os
import gc
import numpy as np
import pandas as pd

BASE      = r"c:\Users\dhl\data\Thesis\thesis\Data"
OUT_DIR   = r"c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\04_T_Learner_ML"
OUT_FILE  = os.path.join(OUT_DIR, "annualized_all_parcel_panel.parquet")

ZONING_CSV    = os.path.join(BASE, "final", "model_ready_zoning_data.csv")
SPATIAL_CSV   = os.path.join(BASE, "Panel", "spatial_allocation_panel.csv")
LDB16_CSV     = os.path.join(BASE, "CoA_Open_Data", "LDB_2016_4nsn-uea6.csv")
LDB21_CSV     = os.path.join(BASE, "CoA_Open_Data", "LDB_2021_kk8y-6cmt.csv")
CROSSWALK_CSV = os.path.join(BASE, "Panel", "Reference", "id_crosswalk.csv")
EARS_DIR      = os.path.join(BASE, "Panel", "Intermediate")
FRED_PATH     = os.path.join(BASE, "Panel", "macro", "fred_timeseries.csv")

def safe_pid(x):
    try: return str(int(float(x))).zfill(10)
    except: return None

def main():
    # ── 1. Load the Universal Spine ──────────────────────────────────────────
    print("1. Loading Universal Parcel Spine...")
    # This dataset has ~400,000 parcels. We will use it as the definitive universe.
    base_panel = pd.read_csv(SPATIAL_CSV, low_memory=False)
    
    # standardize the ID to merge cleanly
    base_panel["parcel_id_10"] = base_panel["standardized_tcad_id"].astype(str).str.zfill(10)
    
    # ── 2. Create the 19-Year Cartesian Product ──────────────────────────────
    print("2. Expanding to 19-Year Longitudinal Skeleton...")
    years = pd.DataFrame({"year": range(2007, 2026)})
    # Cross join creates 19 rows per parcel
    spine = base_panel.merge(years, how="cross")
    print(f"   Spine shape: {spine.shape} rows.")

    # ── 3. Hydrate Temporal Target Variables ─────────────────────────────────
    print("3. Hydrating Target Variables (Filing Events)...")
    z = pd.read_csv(ZONING_CSV, low_memory=False)
    z["application_start_date"] = pd.to_datetime(z["application_start_date"], errors="coerce")
    z = z[z["application_start_date"].notna()].copy()
    z["filing_year"] = z["application_start_date"].dt.year
    z["parcel_id_10"] = z["parcel_id_10"].map(safe_pid)
    
    # Create an event flag per parcel per year
    filings = z.groupby(["parcel_id_10", "filing_year"]).agg(
        case_count=("application_start_date", "size"),
        Valid_Petition_Pct=("Valid_Petition_Pct", "max")
    ).reset_index()
    filings["is_filed_this_year"] = 1

    spine = spine.merge(filings, left_on=["parcel_id_10", "year"], right_on=["parcel_id_10", "filing_year"], how="left")
    spine["is_filed_this_year"] = spine["is_filed_this_year"].fillna(0).astype(int)
    spine["Valid_Petition_Pct"] = spine["Valid_Petition_Pct"].fillna(0.0)
    spine = spine.drop(columns=["filing_year", "case_count"])

    # ── 4. Hydrate Macro-Economics ───────────────────────────────────────────
    print("4. Hydrating FRED Macro Data...")
    if os.path.exists(FRED_PATH):
        fred = pd.read_csv(FRED_PATH)
        fred["observation_date"] = pd.to_datetime(fred["observation_date"])
        fred["year"] = fred["observation_date"].dt.year
        # Group by year and take the mean to annualize
        fred_annual = fred.groupby("year").mean(numeric_only=True).reset_index()
        
        # Calculate 1-yr momentum lags manually
        fred_cols = [c for c in fred_annual.columns if c != "year"]
        for c in fred_cols:
            fred_annual[f"{c}_1yr_lag"] = fred_annual[c].shift(1)
            fred_annual[f"{c}_momentum"] = fred_annual[c] - fred_annual[f"{c}_1yr_lag"]
            
        spine = spine.merge(fred_annual, on="year", how="left")
    else:
        print("   [!] FRED macro file not found. Skipping.")

    # ── 5. Hydrate Temporal Property Value Records (EARS/LDB) ────────────────
    print("5. Forward-filling property data (LDB 2016 -> EARS 2019-2025)...")
    
    LDB16 = pd.read_csv(LDB16_CSV, low_memory=False, usecols=["PID_10","MARKET_VAL","APPRAISED_VAL","LAND_ACRES","YR_BUILT"]).rename(
        columns={"PID_10":"parcel_id_10", "MARKET_VAL":"market_value", "APPRAISED_VAL":"appraised_value", "LAND_ACRES":"land_acres", "YR_BUILT":"yr_built"})
    LDB16["parcel_id_10"] = LDB16["parcel_id_10"].map(safe_pid)
    
    LDB21 = pd.read_csv(LDB21_CSV, low_memory=False, usecols=["PID_10","MARKET_VAL","ASSESSED_V","LAND_ACRES","YR_BUILT"]).rename(
        columns={"PID_10":"parcel_id_10", "MARKET_VAL":"market_value", "ASSESSED_V":"appraised_value", "LAND_ACRES":"land_acres", "YR_BUILT":"yr_built"})
    LDB21["parcel_id_10"] = LDB21["parcel_id_10"].map(safe_pid)

    # Load Crosswalk
    xwalk = pd.read_csv(CROSSWALK_CSV, low_memory=False)
    xwalk["parcel_id_10"] = xwalk["parcel_id_10"].astype(str).str.zfill(10)
    spine = spine.merge(xwalk[["parcel_id_10", "ears_account_number"]].drop_duplicates(), on="parcel_id_10", how="left")

    EARS_COLS = ["account_number","total_market_value","appraised_value","land_acres","year_built"]
    ears = {}
    for yr in range(2019, 2026):
        fp = os.path.join(EARS_DIR, f"ears_{yr}_clean.csv")
        if not os.path.exists(fp): continue
        df = pd.read_csv(fp, low_memory=False, usecols=lambda c: c in EARS_COLS)
        df = df.rename(columns={"total_market_value":"market_value", "year_built":"yr_built"})
        ears[yr] = df

    # We will build a parallel dataframe for the temporal property chunks
    chunks = []
    # Free up memory
    unique_pairs = spine[["parcel_id_10", "ears_account_number", "year"]].drop_duplicates()
    
    for yr, grp in unique_pairs.groupby("year"):
        if yr <= 2018:
            # User Rule: LDB_2016 for earliest dataset (2016, 2017, 2018)
            ref = LDB16; ref_type = "ldb"
        elif yr in ears:
            # Use EARS
            ref = ears[yr]; ref_type = "ears"
        elif yr == 2021:
            # Fallback for 2021 if EARS is missing
            ref = LDB21; ref_type = "ldb"
        else:
            # If EARS missing, forward fill by using the closest previous available year
            avail_years = [k for k in ears.keys() if k < yr]
            if avail_years:
                ref = ears[max(avail_years)]
                ref_type = "ears"
            else:
                ref = LDB16; ref_type = "ldb"

        if ref_type == "ears":
            sub = grp.dropna(subset=["ears_account_number"]).copy()
            sub["ears_account_number"] = sub["ears_account_number"].astype(str)
            ref["account_number"] = ref["account_number"].astype(str)
            m = sub.merge(ref, left_on="ears_account_number", right_on="account_number", how="left")
            m = m.drop(columns=["account_number"])
        else:
            sub = grp.dropna(subset=["parcel_id_10"]).copy()
            m = sub.merge(ref, on="parcel_id_10", how="left")
            
        chunks.append(m)
        
    prop_panel = pd.concat(chunks)
    
    print("6. Merging Property Values back to Spine...")
    # Drop duplicates just in case
    prop_panel = prop_panel.drop_duplicates(["parcel_id_10", "year"])
    
    # Merge back onto spine
    spine = spine.merge(prop_panel.drop(columns=["ears_account_number"]), on=["parcel_id_10", "year"], how="left")

    # Clean up and force numeric types
    for col in ["market_value", "appraised_value", "land_acres", "yr_built"]:
        spine[col] = pd.to_numeric(spine[col], errors="coerce")
        
    spine["building_age"] = spine["year"] - spine["yr_built"]

    # ── 6. Export to Parquet ─────────────────────────────────────────────────
    print(f"7. Exporting to Parquet... Final Shape: {spine.shape}")
    
    # Ensure all column names are strings and clean for parquet export
    spine.columns = spine.columns.astype(str)
    spine.to_parquet(OUT_FILE, index=False)
    
    print(f"SUCCESS: Annualized panel saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
