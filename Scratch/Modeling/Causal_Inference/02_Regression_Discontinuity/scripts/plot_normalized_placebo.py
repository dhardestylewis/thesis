"""
Plot Multi-Target Normalized Threshold Test
Calculates the Regression Discontinuity ATE jump at every 1% threshold for 7 targets,
but NORMALIZES the outcomes to Z-scores (Standard Deviations) so all effects
are plotted on the exact same scale and their magnitudes are directly comparable.
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
PET_INTENSITY = rf"{OUT_DIR}\exact_geometric_petition_intensity.csv"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"
CASE_MASTER = r"C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\Build\case_master.csv"
VOTES_TRANSCRIPT = r"C:\Users\dhl\data\Thesis\thesis\Data\interim\zoning_cases_with_council_votes.csv"

# Load Data
pet = pd.read_csv(PET_INTENSITY)
master = pd.read_csv(MASTER_PATH, low_memory=False)
panel = pd.read_csv(PANEL_PATH, low_memory=False)
cm = pd.read_csv(CASE_MASTER, low_memory=False)
vt = pd.read_csv(VOTES_TRANSCRIPT, low_memory=False)

pet["case_number"] = pet["case_number"].str.strip()
master["case_number"] = master["case_number"].str.strip()
cm["CASE_NUMBER"] = cm["CASE_NUMBER"].str.strip()
vt["Case_Number"] = vt["Case_Number"].str.strip()

# 1. Outcomes
def clean_status(s):
    if pd.isna(s): return "Unknown"
    s = s.lower()
    if "withdrawn" in s: return "Withdrawn"
    if "denied" in s: return "Denied"
    if "closed" in s or "void" in s or "expired" in s: return "Passive_Death"
    return "Pending"

cm_status = cm[["CASE_NUMBER", "DETAILED_STATUS"]].drop_duplicates("CASE_NUMBER").copy()
cm_status["status_cat"] = cm_status["DETAILED_STATUS"].apply(clean_status)
cm_status["t_withdrawal"] = (cm_status["status_cat"] == "Withdrawn").astype(float)
cm_status["t_denial"] = (cm_status["status_cat"] == "Denied").astype(float)
cm_status["t_passive_death"] = (cm_status["status_cat"] == "Passive_Death").astype(float)

vt_df = []
import re
vote_pattern = re.compile(r'\b(\d{1,2})-(\d{1,2})\s*vote\b', re.IGNORECASE)
for _, row in vt.iterrows():
    matches = vote_pattern.findall(str(row["Vote_Transcript"]))
    if matches:
        for m in matches:
            yes, no = int(m[0]), int(m[1])
            if 3 <= (yes + no) <= 11:
                vt_df.append({"Case_Number": row["Case_Number"], "no_votes": no})
vt_agg = pd.DataFrame(vt_df)
if not vt_agg.empty:
    vt_agg = vt_agg.groupby("Case_Number").agg(t_max_nay_votes=("no_votes", "max")).reset_index()
else:
    vt_agg = pd.DataFrame(columns=["Case_Number", "t_max_nay_votes"])

hearings = panel.groupby("case_number").agg(
    t_total_council_hearings=("council_hearings_this_period", "sum")
).reset_index()

OVERLAY_STRIP = __import__("re").compile(r"(-NP|-CO|-H|-V|-CURE|-NCCD|-MU|-L|-SH|-DB90|-DB110|-ETOD|-PDA|-IA|-UC|-CU|-ICG|-W|-LEED|-SR|-PO|-DT|-NO|-OLD)")
INTENSITY = {"W":1,"RR":1,"AG":1,"DR":1,"SF-1":2,"SF-2":2,"SF-3":2,"SF-4A":3,"SF-4B":3,"SF-5":3,"SF-6":3,"TF":3,"MF-1":4,"MF-2":4,"MF-3":5,"MF-4":5,"MF-5":6,"MF-6":6,"LO":5,"GO":6,"NO":5,"LR":6,"GR":7,"CS":7,"CS-1":7,"CR":7,"CH":8,"LI":8,"MI":9,"HI":9,"CBD":9,"DMU":8,"TOD":7,"MU":7,"PUD":7,"P":6}
def get_int(z): return INTENSITY.get(OVERLAY_STRIP.sub("", str(z).strip().upper()).strip("-"), np.nan)
master["req_int"] = master["Requested_Zoning"].apply(get_int)
master["fin_int"] = master["Final_Zoning"].apply(get_int)
master["z_changed"] = master["Requested_Zoning"].str.strip() != master["Final_Zoning"].str.strip()
master["t_downgrade"] = ((master["fin_int"] < master["req_int"]) & master["z_changed"]).astype(float)

master["req_co"] = master["Requested_Zoning"].str.contains("-CO", na=False)
master["fin_co"] = master["Final_Zoning"].str.contains("-CO", na=False)
master["t_co_added"] = (~master["req_co"] & master["fin_co"] & master["z_changed"]).astype(float)

concessions = master[["case_number", "t_downgrade", "t_co_added"]].drop_duplicates("case_number")

df = pet[["case_number", "label_exact_geometric_petition_pct"]].copy()
df = df.merge(cm_status.rename(columns={"CASE_NUMBER":"case_number"}), on="case_number", how="inner")
df = df.merge(vt_agg.rename(columns={"Case_Number":"case_number"}), on="case_number", how="left")
df["t_max_nay_votes"] = df["t_max_nay_votes"].fillna(0)
df = df.merge(hearings, on="case_number", how="inner")
df = df.merge(concessions, on="case_number", how="inner")

targets = {
    "t_withdrawal": ("Withdrawal", "prob"),
    "t_denial": ("Denial", "prob"),
    "t_passive_death": ("Passive Death", "prob"),
    "t_downgrade": ("Zoning Downgrade", "prob"),
    "t_co_added": ("Conditional Overlay Added", "prob"),
    "t_total_council_hearings": ("Council Hearings", "count_h"),
    "t_max_nay_votes": ("Nay Votes", "count_v")
}

# Normalize all targets to Z-scores (Standard Deviations) so they are directly comparable
for t in targets.keys():
    df[f"{t}_z"] = (df[t] - df[t].mean()) / df[t].std()

# 2. Compute RD with Normalized Data
def run_rd_cutoff(cutoff, t_col):
    df["run_var"] = df["label_exact_geometric_petition_pct"] - cutoff
    rd_df = df[(df["run_var"] >= -15) & (df["run_var"] <= 15)].copy()
    if len(rd_df[rd_df["run_var"] >= 0]) >= 2 and len(rd_df[rd_df["run_var"] < 0]) >= 2:
        left = rd_df[rd_df["run_var"] < 0][t_col].mean()
        right = rd_df[rd_df["run_var"] >= 0][t_col].mean()
        return right - left
    return np.nan

cutoffs = list(range(1, 101))

fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
axes = axes.flatten()

for i, (t_col, (t_name, t_type)) in enumerate(targets.items()):
    # Run the RD on the Z-scored column
    ates = [run_rd_cutoff(c, f"{t_col}_z") for c in cutoffs]
    ax = axes[i]
    
    # Mask out extreme noise above 70%
    ax.plot(cutoffs[:50], ates[:50], color='indigo', marker='o', linewidth=2, markersize=4)
    ax.plot(cutoffs[50:], ates[50:], color='indigo', marker='o', linewidth=1, markersize=2, alpha=0.3)

    ax.axvline(x=20, color='black', linestyle='--', linewidth=1.5, label='20% Threshold')
    
    # Highlight peak magnitude
    valid_ates = [a for a in ates[:50] if not np.isnan(a)]
    if valid_ates:
        if "Death" in t_name or "Conditional" in t_name: # Negative peaks
            peak_val = np.nanmin(ates[:50])
        else:
            peak_val = np.nanmax(ates[:50])
            
        peak_idx = ates.index(peak_val)
        ax.scatter([cutoffs[peak_idx]], [peak_val], color='gold', s=150, zorder=5, edgecolor='black')
        
        # Add text label
        offset = (np.nanmax(ates[:50]) - np.nanmin(ates[:50])) * 0.05
        if np.isnan(offset) or offset == 0: offset = 0.05
        y_text = peak_val + offset if peak_val > 0 else peak_val - offset
        ax.text(cutoffs[peak_idx], y_text, f"{peak_val:+.2f} SD", ha='center', va='center', fontweight='bold', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_title(t_name, fontsize=12, fontweight='bold')
    
    if i >= 4: ax.set_xlabel("Evaluated Cutoff %")
    if i % 4 == 0: ax.set_ylabel("ATE Jump (Standard Deviations)")
    ax.grid(alpha=0.2)
    if i==0: ax.legend(fontsize=9, loc='upper left')

axes[7].set_visible(False)
plt.suptitle("Normalized Threshold Robustness Test (Effect Magnitude in Standard Deviations)", fontsize=16, fontweight='bold')
sns.despine()
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = rf"{OUT_DIR}\multi_target_normalized_robustness.png"
plt.savefig(out_path, dpi=300)
print(f"Plot saved to {out_path}")
