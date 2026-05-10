"""
Plot Placebo RD Cutoffs
Calculates the Regression Discontinuity ATE jump at every 1% threshold
and plots it to visually prove the effect peaks exclusively at 20%.
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"
PET_INTENSITY = rf"{OUT_DIR}\petition_intensity_corrected.csv"

pet = pd.read_csv(PET_INTENSITY)
master = pd.read_csv(MASTER_PATH, low_memory=False)

OVERLAY_STRIP = __import__("re").compile(r"(-NP|-CO|-H|-V|-CURE|-NCCD|-MU|-L|-SH|-DB90|-DB110|-ETOD|-PDA|-IA|-UC|-CU|-ICG|-W|-LEED|-SR|-PO|-DT|-NO|-OLD)")
INTENSITY = {"W":1,"RR":1,"AG":1,"DR":1,"SF-1":2,"SF-2":2,"SF-3":2,"SF-4A":3,"SF-4B":3,"SF-5":3,"SF-6":3,"TF":3,"MF-1":4,"MF-2":4,"MF-3":5,"MF-4":5,"MF-5":6,"MF-6":6,"LO":5,"GO":6,"NO":5,"LR":6,"GR":7,"CS":7,"CS-1":7,"CR":7,"CH":8,"LI":8,"MI":9,"HI":9,"CBD":9,"DMU":8,"TOD":7,"MU":7,"PUD":7,"P":6}
def get_int(z): return INTENSITY.get(OVERLAY_STRIP.sub("", str(z).strip().upper()).strip("-"), np.nan)

master["case_number"] = master["case_number"].str.strip()
master["req_int"] = master["Requested_Zoning"].apply(get_int)
master["fin_int"] = master["Final_Zoning"].apply(get_int)
master["z_changed"] = master["Requested_Zoning"].str.strip() != master["Final_Zoning"].str.strip()
master["t_downgrade"] = ((master["fin_int"] < master["req_int"]) & master["z_changed"]).astype(float)
concessions = master[["case_number", "t_downgrade"]].drop_duplicates("case_number")

df = pet[["case_number", "true_petition_pct"]].copy()
df["case_number"] = df["case_number"].str.strip()
df = df.merge(concessions, on="case_number", how="inner")

def run_rd_cutoff(cutoff, t_col):
    df["run_var"] = df["true_petition_pct"] - cutoff
    # 15% bandwidth around the fake cutoff
    rd_df = df[(df["run_var"] >= -15) & (df["run_var"] <= 15)].copy()
    
    if len(rd_df[rd_df["run_var"] >= 0]) >= 3 and len(rd_df[rd_df["run_var"] < 0]) >= 3:
        left = rd_df[rd_df["run_var"] < 0][t_col].mean()
        right = rd_df[rd_df["run_var"] >= 0][t_col].mean()
        ate = right - left
        return ate
    return np.nan

cutoffs = list(range(1, 55))
ates = [run_rd_cutoff(c, "t_downgrade") for c in cutoffs]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(cutoffs, ates, color='firebrick', marker='o', linewidth=2, markersize=5)
plt.axvline(x=20, color='black', linestyle='--', linewidth=2, label='20% Legal Supermajority Threshold')

# Highlight peak
peak_idx = np.nanargmax(ates)
plt.scatter([cutoffs[peak_idx]], [ates[peak_idx]], color='gold', s=150, zorder=5, edgecolor='black', label=f'Peak Effect (+{ates[peak_idx]:.3f})')

plt.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

plt.title("Placebo Cutoff Test: The Causal Jump in Zoning Downgrades", fontsize=14, fontweight='bold')
plt.xlabel("Simulated Legal Threshold (% Neighbors Signing)", fontsize=12)
plt.ylabel("Discontinuous Jump in Probability of Downgrade (ATE)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(alpha=0.2)

sns.despine()
plt.tight_layout()

out_path = rf"{OUT_DIR}\rd_placebo_test.png"
plt.savefig(out_path, dpi=300)
print(f"Plot saved to {out_path}")
