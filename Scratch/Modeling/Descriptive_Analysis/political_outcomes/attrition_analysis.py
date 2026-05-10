import pandas as pd
import numpy as np

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"
PET_INTENSITY = rf"{OUT_DIR}\petition_intensity.csv"

master = pd.read_csv(MASTER_PATH, low_memory=False)
pet = pd.read_csv(PET_INTENSITY)

# We want to see case status and attrition
# Join any_protest to master
df = master.drop_duplicates("case_number").copy()
pet_status = pet[["case_number", "petition_n_parcels"]].copy()
pet_status["any_protest"] = (pet_status["petition_n_parcels"] > 0).astype(int)

df = df.merge(pet_status[["case_number", "any_protest"]], on="case_number", how="left")
df["any_protest"] = df["any_protest"].fillna(0)

# Create a clean completion status
# "Approved (Scraped)" or "Approved (Unscraped)" -> Approved
# "Unresolved (*)" -> Pending/Dead
df["is_approved"] = df["Derived_Status"].str.contains("Approved").astype(float)
df["is_unresolved"] = df["Derived_Status"].str.contains("Unresolved").astype(float)

# Let's also look at how this changes by vintage (filing year) to separate "recently pending" from "dead"
df["application_start_date"] = pd.to_datetime(df["application_start_date"], errors="coerce")
df["filing_year"] = df["application_start_date"].dt.year

print("1. OVERALL APPROVAL / ATTRITION RATES (All Years)")
comp = df.groupby("any_protest").agg(
    total_cases=("case_number", "count"),
    pct_approved=("is_approved", "mean"),
    pct_unresolved=("is_unresolved", "mean")
).round(3)
comp.index = ["No Protest", "Any Protest"]
print(comp.to_string())

print("\n2. ATTRITION RATES FOR 'MATURE' CASES (Filed 2016-2022)")
# Cases filed 2023+ might genuinely be pending. Cases from 2016-2022 that are "Unresolved" are functionally dead/withdrawn.
mature = df[df["filing_year"] <= 2022]
comp_mature = mature.groupby("any_protest").agg(
    total_cases=("case_number", "count"),
    pct_approved=("is_approved", "mean"),
    pct_dead=("is_unresolved", "mean")  # Unresolved here means dead
).round(3)
comp_mature.index = ["No Protest", "Any Protest"]
print(comp_mature.to_string())

# Also check outcomes (downgrades) among ONLY the approved cases
print("\n3. DOWNGRADE RATES AMONG *APPROVED* MATURE CASES")
mature_approved = mature[mature["is_approved"] == 1].copy()

OVERLAY_STRIP = __import__("re").compile(
    r"(-NP|-CO|-H|-V|-CURE|-NCCD|-MU|-L|-SH|-DB90|-DB110|-ETOD|-PDA|-IA|-UC|-CU|-ICG|-W|-LEED|-SR|-PO|-DT|-NO|-OLD)"
)
INTENSITY = {
    "W":1,"RR":1,"AG":1,"DR":1,"SF-1":2,"SF-2":2,"SF-3":2,
    "SF-4A":3,"SF-4B":3,"SF-5":3,"SF-6":3,"TF":3,
    "MF-1":4,"MF-2":4,"MF-3":5,"MF-4":5,"MF-5":6,"MF-6":6,
    "LO":5,"GO":6,"NO":5,"LR":6,"GR":7,"CS":7,"CS-1":7,"CR":7,"CH":8,
    "LI":8,"MI":9,"HI":9,"CBD":9,"DMU":8,"TOD":7,"MU":7,"PUD":7,"P":6,
}
def base_zone(z):
    if not isinstance(z, str): return None
    return OVERLAY_STRIP.sub("", z.strip().upper()).strip("-")
def intensity(z):
    return INTENSITY.get(base_zone(z), np.nan)

mature_approved["req_int"] = mature_approved["Requested_Zoning"].apply(intensity)
mature_approved["fin_int"] = mature_approved["Final_Zoning"].apply(intensity)
mature_approved["zoning_changed"] = (mature_approved["Requested_Zoning"].str.strip() != mature_approved["Final_Zoning"].str.strip()).astype(float)
mature_approved["downgrade"] = ((mature_approved["fin_int"] < mature_approved["req_int"]) & mature_approved["zoning_changed"].astype(bool)).astype(float)

comp_down = mature_approved.dropna(subset=["downgrade"]).groupby("any_protest").agg(
    n_approved_with_zoning_data=("case_number", "count"),
    downgrade_rate=("downgrade", "mean")
).round(3)
comp_down.index = ["No Protest", "Any Protest"]
print(comp_down.to_string())
