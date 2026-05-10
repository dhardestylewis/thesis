import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"
GEOM_PATH = rf"{OUT_DIR}\exact_geometric_petition_intensity.csv"
OUT_PLOT = rf"{OUT_DIR}\rd_placebo_thresholds.png"

def plot_placebo_panel(ax, df, cutoff, is_true=False):
    # Colors
    c_control = "#2A3F54"
    c_treated = "#E74C3C" if is_true else "#7F8C8D" # Gray out the fake treated sides
    palette = {0: c_control, 1: c_treated}
    
    # Binarize for the specific cutoff
    df = df.copy()
    df["threshold_crossed"] = (df["label_exact_geometric_petition_pct"] >= cutoff).astype(int)
    
    # Scatter 0%
    zero_pct = df[df["label_exact_geometric_petition_pct"] == 0]
    jittered_zero = np.random.uniform(-1, 1, size=len(zero_pct))
    sns.scatterplot(x=jittered_zero, y=zero_pct["council_approved"], 
                    color=c_control, alpha=0.1, s=15, ax=ax, edgecolor="none", zorder=1)
    
    # Scatter >0%
    active_petitions = df[df["label_exact_geometric_petition_pct"] > 0]
    sns.scatterplot(data=active_petitions, x="label_exact_geometric_petition_pct", y="council_approved",
                    hue="threshold_crossed", palette=palette, alpha=0.5, s=25,
                    edgecolor="w", linewidth=0.5, ax=ax, legend=False, zorder=2)
    
    # RD Fits
    control = df[df["label_exact_geometric_petition_pct"] < cutoff]
    treated = df[df["label_exact_geometric_petition_pct"] >= cutoff]
    
    sns.regplot(data=control, x="label_exact_geometric_petition_pct", y="council_approved",
                scatter=False, color=c_control, order=2, ax=ax, 
                line_kws={"linewidth": 2, "zorder": 3})
                
    if len(treated) > 5:
        sns.regplot(data=treated, x="label_exact_geometric_petition_pct", y="council_approved",
                    scatter=False, color=c_treated, order=1, ax=ax, 
                    line_kws={"linewidth": 2, "zorder": 3})
    
    ax.axvline(cutoff, color="black" if is_true else "gray", 
               linestyle="-" if is_true else "--", linewidth=2 if is_true else 1.5, zorder=0)
               
    title = f"TRUE THRESHOLD: 20%" if is_true else f"Fake Placebo Threshold: {cutoff}%"
    ax.set_title(title, fontsize=11, weight='bold' if is_true else 'normal', color='black' if is_true else 'gray')
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Approval Prob" if ax.get_subplotspec().is_first_col() else "")
    ax.set_xlabel("Petition %" if ax.get_subplotspec().is_last_row() else "")

def main():
    print("Loading saturated universe data...")
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    geom_df = pd.read_csv(GEOM_PATH)
    
    master["case_number"] = master["case_number"].str.strip()
    geom_df["case_number"] = geom_df["case_number"].str.strip()
    
    # Merge outcome and binarize
    df = geom_df.merge(master[["case_number", "Derived_Status"]], on="case_number", how="inner")
    df["council_approved"] = df["Derived_Status"].apply(lambda x: 1 if pd.notna(x) and "Approved" in str(x) else 0)
    
    # Setup 2x3 Grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=200)
    cutoffs = [10, 15, 20, 25, 30, 40]
    
    for i, cutoff in enumerate(cutoffs):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        print(f"Plotting cutoff: {cutoff}%")
        is_true = (cutoff == 20)
        plot_placebo_panel(ax, df, cutoff, is_true)
    
    plt.suptitle("Regression Discontinuity Placebo Threshold Falsification Tests", fontsize=18, weight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUT_PLOT)
    print(f"Simulation complete! Saved to {OUT_PLOT}")

if __name__ == "__main__":
    main()
