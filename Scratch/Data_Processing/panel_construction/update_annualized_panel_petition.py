import pandas as pd
import numpy as np

PARQUET_PATH = r"c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\04_T_Learner_ML\annualized_all_parcel_panel.parquet"
ZONING_CSV = r"c:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"

def safe_pid(x):
    try: return str(int(float(x))).zfill(10)
    except: return None

print("1. Loading Annualized Panel...")
df = pd.read_parquet(PARQUET_PATH)

print("2. Loading Zoning Cases for Petition Data...")
z = pd.read_csv(ZONING_CSV, low_memory=False)
z["App_Date"] = pd.to_datetime(z["App_Date"], errors="coerce")
z = z[z["App_Date"].notna()].copy()
z["filing_year"] = z["App_Date"].dt.year
z["parcel_id_10"] = z["parcel_id_10"].map(safe_pid)

# Group by parcel_id_10 and filing_year, taking the MAX Valid_Petition_Pct if multiple cases exist
petitions = z.groupby(["parcel_id_10", "filing_year"])["Valid_Petition_Pct"].max().reset_index()

print("3. Merging Petition Severity onto Panel...")
# If it already exists, drop it to avoid _x, _y
if "Valid_Petition_Pct" in df.columns:
    df = df.drop(columns=["Valid_Petition_Pct"])

df = df.merge(petitions, left_on=["parcel_id_10", "year"], right_on=["parcel_id_10", "filing_year"], how="left")
df["Valid_Petition_Pct"] = df["Valid_Petition_Pct"].fillna(0.0)
df = df.drop(columns=["filing_year"], errors="ignore")

print(f"4. Saving Updated Panel... (Max Petition: {df['Valid_Petition_Pct'].max()})")
df.to_parquet(PARQUET_PATH, index=False)
print("Done!")
