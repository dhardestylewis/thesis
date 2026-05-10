"""
Protest Dosage Response Analysis
Analyzes how the intensity of protest (percentage signed, number of signing parcels)
correlates with outcomes (downgrades, hearings, time to resolution).
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
PANEL_PATH  = rf"{OUT_DIR}\biweekly_panel.csv"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"
PET_INTENSITY = rf"{OUT_DIR}\petition_intensity.csv"

# Load data
pet = pd.read_csv(PET_INTENSITY)
master = pd.read_csv(MASTER_PATH, low_memory=False)
panel = pd.read_csv(PANEL_PATH, low_memory=False)

# Rebuild outcomes
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

outcomes = master[["case_number","Requested_Zoning","Final_Zoning"]].drop_duplicates("case_number").copy()
outcomes["req_intensity"] = outcomes["Requested_Zoning"].apply(intensity)
outcomes["fin_intensity"] = outcomes["Final_Zoning"].apply(intensity)
outcomes["zoning_changed"] = (outcomes["Requested_Zoning"].str.strip() != outcomes["Final_Zoning"].str.strip()).astype(float)
outcomes["downgrade"] = ((outcomes["fin_intensity"] < outcomes["req_intensity"]) & outcomes["zoning_changed"].astype(bool)).astype(float)

# Council hearings (total)
hearings = panel.groupby("case_number")["council_hearings_this_period"].sum().reset_index()

# Merge
df = pet.merge(outcomes, on="case_number", how="inner").merge(hearings, on="case_number", how="inner")
# Ensure we only look at protested cases to see the marginal dosage effect, or all cases
# If all cases, non-protested have petition_area_pct_raw = 0
all_cases = master[["case_number"]].drop_duplicates().merge(outcomes, on="case_number", how="left").merge(hearings, on="case_number", how="left")
all_cases = all_cases.merge(pet[["case_number", "petition_area_pct_raw", "petition_n_parcels"]], on="case_number", how="left")
all_cases["petition_area_pct_raw"] = all_cases["petition_area_pct_raw"].fillna(0)
all_cases["petition_n_parcels"] = all_cases["petition_n_parcels"].fillna(0)
all_cases["any_protest"] = (all_cases["petition_n_parcels"] > 0).astype(int)

print("\n" + "="*60)
print("1. ANY PROTEST VS NO PROTEST (Outcomes)")
print("="*60)
comp = all_cases.groupby("any_protest").agg(
    count=("case_number", "count"),
    avg_council_hearings=("council_hearings_this_period", "mean"),
    zoning_changed_rate=("zoning_changed", "mean"),
    downgrade_rate=("downgrade", "mean"),
).round(3)
comp.index = ["No Protest", "Any Protest"]
print(comp.to_string())

print("\n" + "="*60)
print("2. DOSAGE RESPONSE: DO MORE SIGNATURES = MORE DOWNGRADES?")
print("="*60)
print("Among cases that got AT LEAST SOME protest (n={}), does the percentage of neighbors".format(len(df)))
print("signing correlate with a higher chance of a zoning downgrade or more hearings?")

df_clean = df.dropna(subset=["downgrade", "petition_area_pct_raw"])

# OLS for Council Hearings ~ Protest Pct
X_hearings = sm.add_constant(df["petition_area_pct_raw"])
y_hearings = df["council_hearings_this_period"]
model_hearings = sm.OLS(y_hearings, X_hearings, missing='drop').fit()
print("\n--- OLS: Total Council Hearings ~ Protest Percentage ---")
print(f"Coefficient for Protest Pct: {model_hearings.params['petition_area_pct_raw']:.4f} (p-value: {model_hearings.pvalues['petition_area_pct_raw']:.3f})")
if model_hearings.pvalues['petition_area_pct_raw'] < 0.05:
    print("Result: SIGNIFICANT. Higher protest intensity leads to more council hearings.")
else:
    print("Result: NOT significant. Intensity of protest doesn't increase hearings beyond the baseline delay.")

# Logistic for Downgrade ~ Protest Pct
X_down = sm.add_constant(df_clean["petition_area_pct_raw"])
y_down = df_clean["downgrade"]
model_down = sm.Logit(y_down, X_down).fit(disp=0)
print("\n--- Logistic: Probability of Downgrade ~ Protest Percentage ---")
print(f"Coefficient for Protest Pct: {model_down.params['petition_area_pct_raw']:.4f} (p-value: {model_down.pvalues['petition_area_pct_raw']:.3f})")
if model_down.pvalues['petition_area_pct_raw'] < 0.05:
    print("Result: SIGNIFICANT. Higher protest intensity increases probability of a zoning concession.")
else:
    print("Result: NOT significant. Intensity of protest does not increase the probability of a concession.")

print("\n" + "="*60)
print("3. QUARTILE ANALYSIS: Intensity Tiers")
print("="*60)
df["protest_quartile"] = pd.qcut(df["petition_area_pct_raw"], 4, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"])
q_comp = df.groupby("protest_quartile").agg(
    n_cases=("case_number", "count"),
    median_pct_signed=("petition_area_pct_raw", "median"),
    avg_council_hearings=("council_hearings_this_period", "mean"),
    downgrade_rate=("downgrade", "mean")
).round(3)
print(q_comp.to_string())
