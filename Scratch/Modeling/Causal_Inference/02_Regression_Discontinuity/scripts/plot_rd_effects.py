"""
Plot Regression Discontinuity
Generates visual proof of the discontinuous jump at the 20% threshold,
controlling for any continuous underlying trends.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"
PET_INTENSITY = rf"{OUT_DIR}\petition_intensity_corrected.csv"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"

# Load Data
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

# Merge
df = pet[["case_number", "true_petition_pct"]].copy()
df["case_number"] = df["case_number"].str.strip()
df = df.merge(concessions, on="case_number", how="inner").merge(hearings, on="case_number", how="inner")

# Isolate the bandwidth around the threshold (0% to 40%)
df = df[(df["true_petition_pct"] > 0) & (df["true_petition_pct"] <= 50)].copy()

# Plotting Function
def plot_rd(ax, x, y, title, ylabel):
    ax.scatter(x, y, alpha=0.3, color='gray', s=15, label='Individual Cases')
    
    # Left side (< 20%)
    left_mask = x < 20
    x_left = x[left_mask]
    y_left = y[left_mask]
    if len(x_left) > 3:
        # Fit polynomial
        poly_left = np.poly1d(np.polyfit(x_left, y_left, 1))
        x_plot_left = np.linspace(x_left.min(), 19.99, 100)
        ax.plot(x_plot_left, poly_left(x_plot_left), color='blue', linewidth=2.5, label='Linear Fit (<20%)')
        # Scatter binned means for cleaner visual
        bins = np.linspace(0, 20, 10)
        bin_means, bin_edges, _ = stats.binned_statistic(x_left, y_left, 'mean', bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.scatter(bin_centers, bin_means, color='blue', s=40, zorder=5)

    # Right side (>= 20%)
    right_mask = x >= 20
    x_right = x[right_mask]
    y_right = y[right_mask]
    if len(x_right) > 3:
        # Fit polynomial
        poly_right = np.poly1d(np.polyfit(x_right, y_right, 1))
        x_plot_right = np.linspace(20.0, 50, 100)
        ax.plot(x_plot_right, poly_right(x_plot_right), color='red', linewidth=2.5, label='Linear Fit (>=20%)')
        # Scatter binned means
        bins = np.linspace(20, 50, 15)
        bin_means, bin_edges, _ = stats.binned_statistic(x_right, y_right, 'mean', bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.scatter(bin_centers, bin_means, color='red', s=40, zorder=5)

    ax.axvline(x=20, color='black', linestyle='--', linewidth=1.5, label='20% Legal Threshold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("True Petition Percentage (%)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.legend()

import scipy.stats as stats
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

plot_rd(ax1, df["true_petition_pct"], df["t_total_council_hearings"], 
        "RD Plot: Bureaucratic Delay", "Total Council Hearings")
        
plot_rd(ax2, df["true_petition_pct"], df["t_downgrade"], 
        "RD Plot: Zoning Concessions", "Probability of Downgrade")

plt.tight_layout()
plt.savefig(rf"{OUT_DIR}\rd_visual_proof.png", dpi=300)
print(f"Plot saved to {OUT_DIR}\\rd_visual_proof.png")
