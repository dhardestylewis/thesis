"""
Placebo Threshold Test for Regression Discontinuity
Loops through fake cutoffs (5% to 35%) to prove that the discontinuous jump
in bureaucratic delay and downgrades only occurs at the legal 20% mark.
"""
import pandas as pd
import numpy as np
from scipy import stats

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"
PET_INTENSITY = rf"{OUT_DIR}\petition_intensity_corrected.csv"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"

pet = pd.read_csv(PET_INTENSITY)
master = pd.read_csv(MASTER_PATH, low_memory=False)
panel = pd.read_csv(PANEL_PATH, low_memory=False)

# Outcomes
OVERLAY_STRIP = __import__("re").compile(r"(-NP|-CO|-H|-V|-CURE|-NCCD|-MU|-L|-SH|-DB90|-DB110|-ETOD|-PDA|-IA|-UC|-CU|-ICG|-W|-LEED|-SR|-PO|-DT|-NO|-OLD)")
INTENSITY = {"W":1,"RR":1,"AG":1,"DR":1,"SF-1":2,"SF-2":2,"SF-3":2,"SF-4A":3,"SF-4B":3,"SF-5":3,"SF-6":3,"TF":3,"MF-1":4,"MF-2":4,"MF-3":5,"MF-4":5,"MF-5":6,"MF-6":6,"LO":5,"GO":6,"NO":5,"LR":6,"GR":7,"CS":7,"CS-1":7,"CR":7,"CH":8,"LI":8,"MI":9,"HI":9,"CBD":9,"DMU":8,"TOD":7,"MU":7,"PUD":7,"P":6}
def get_int(z): return INTENSITY.get(OVERLAY_STRIP.sub("", str(z).strip().upper()).strip("-"), np.nan)

master["case_number"] = master["case_number"].str.strip()
master["req_int"] = master["Requested_Zoning"].apply(get_int)
master["fin_int"] = master["Final_Zoning"].apply(get_int)
master["z_changed"] = master["Requested_Zoning"].str.strip() != master["Final_Zoning"].str.strip()
master["t_downgrade"] = ((master["fin_int"] < master["req_int"]) & master["z_changed"]).astype(float)
concessions = master[["case_number", "t_downgrade"]].drop_duplicates("case_number")

hearings = panel.groupby("case_number").agg(
    t_total_council_hearings=("council_hearings_this_period", "sum")
).reset_index()

df = pet[["case_number", "true_petition_pct"]].copy()
df["case_number"] = df["case_number"].str.strip()
df = df.merge(concessions, on="case_number", how="inner").merge(hearings, on="case_number", how="inner")

def run_rd_cutoff(cutoff, t_col):
    df["run_var"] = df["true_petition_pct"] - cutoff
    rd_df = df[(df["run_var"] >= -15) & (df["run_var"] <= 15)].copy() # 15% bandwidth
    
    if len(rd_df[rd_df["run_var"] >= 0]) >= 5 and len(rd_df[rd_df["run_var"] < 0]) >= 5:
        left = rd_df[rd_df["run_var"] < 0][t_col].mean()
        right = rd_df[rd_df["run_var"] >= 0][t_col].mean()
        ate = right - left
        pooled_se = np.sqrt(rd_df[rd_df["run_var"] < 0][t_col].var()/max(len(rd_df[rd_df["run_var"] < 0]),1) + rd_df[rd_df["run_var"] >= 0][t_col].var()/max(len(rd_df[rd_df["run_var"] >= 0]),1))
        z_stat = ate / (pooled_se + 1e-9)
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        return ate, p_val
    return np.nan, np.nan

cutoffs = [5, 10, 15, 20, 25, 30]
targets = {"t_total_council_hearings": "Council Hearings", "t_downgrade": "Zoning Downgrade"}

print("="*60)
print("PLACEBO THRESHOLD TEST: Does the jump only happen at 20%?")
print("="*60)

for t_col, t_name in targets.items():
    print(f"\nTarget: {t_name}")
    print(f"{'Cutoff':<10} | {'Discontinuous Jump (ATE)':<25} | {'P-Value':<10}")
    print("-" * 50)
    for c in cutoffs:
        ate, p = run_rd_cutoff(c, t_col)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        if c == 20:
            print(f"{str(c)+'% (LEGAL)':<10} | {ate:>+10.3f}                       | {p:.3f} ({sig})")
        else:
            print(f"{str(c)+'%':<10} | {ate:>+10.3f}                       | {p:.3f} ({sig})")
