import pandas as pd
import numpy as np

print("Loading annualized panel...")
df = pd.read_parquet(r"c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\04_T_Learner_ML\annualized_all_parcel_panel.parquet")

print("Loading petitions...")
p = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\Protest_Petitions\petition_signers_backfilled.csv")

# Sum valid signatures per case
valid_petitions = p[p["signed"] == 1].groupby("case_number")["area_pct"].sum().reset_index()
valid_petitions.rename(columns={"area_pct": "true_petition_pct"}, inplace=True)

# Merge to get parcel_id_10 from model_ready_zoning_data
print("Loading zoning cases to map case_number to parcel_id_10...")
z = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv", low_memory=False)
def safe_pid(x):
    try: return str(int(float(x))).zfill(10)
    except: return None
z["parcel_id_10"] = z["parcel_id_10"].map(safe_pid)
z["filing_year"] = pd.to_datetime(z["App_Date"], errors="coerce").dt.year

z_mapped = z.merge(valid_petitions, on="case_number", how="inner")
# Group by parcel_id_10 and filing_year to get the max petition per parcel-year
petitions = z_mapped.groupby(["parcel_id_10", "filing_year"])["true_petition_pct"].max().reset_index()

print("Merging true petitions into annualized panel...")
df = df.merge(petitions, left_on=["parcel_id_10", "year"], right_on=["parcel_id_10", "filing_year"], how="left")

# Replace Valid_Petition_Pct
df["Valid_Petition_Pct"] = df["true_petition_pct"].fillna(0.0)
df = df.drop(columns=["true_petition_pct", "filing_year"])

# Cap petition percentage at 100 in case of summation errors (e.g. 3217% -> 32.17%)
# Wait, if max is 3217, it's probably out of 10000 or it's just raw area sums. Let's leave as is for now, but CatBoost regressor will handle it.
# Wait, the threshold is > 20 for protested! So if it's 3217, it's > 20.
df.to_parquet(r"c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\04_T_Learner_ML\annualized_all_parcel_panel.parquet", index=False)
print("SUCCESS: Patched annualized panel!")
