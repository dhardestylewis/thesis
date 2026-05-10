import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"
GEOM_PATH = rf"{OUT_DIR}\exact_geometric_petition_intensity.csv"
OUT_PLOT = rf"{OUT_DIR}\rd_ablation_grid.png"

def plot_rd_panel(ax, df, title):
    # Plotting logic mirrored from the placebo plot
    palette = {0: "#2A3F54", 1: "#E74C3C"}
    
    # 0% petitions
    zero_pct = df[df["label_exact_geometric_petition_pct"] == 0]
    jittered_zero = np.random.uniform(-1, 1, size=len(zero_pct))
    sns.scatterplot(x=jittered_zero, y=zero_pct["council_approved"], 
                    color=palette[0], alpha=0.1, s=15, ax=ax, edgecolor="none", zorder=1)
    
    # >0% petitions
    active_petitions = df[df["label_exact_geometric_petition_pct"] > 0]
    sns.scatterplot(data=active_petitions, x="label_exact_geometric_petition_pct", y="council_approved",
                    hue="threshold_crossed", palette=palette, alpha=0.5, s=25,
                    edgecolor="w", linewidth=0.5, ax=ax, legend=False, zorder=2)
    
    # RD Fits
    control = df[df["label_exact_geometric_petition_pct"] < 20]
    treated = df[df["label_exact_geometric_petition_pct"] >= 20]
    
    sns.regplot(data=control, x="label_exact_geometric_petition_pct", y="council_approved",
                scatter=False, color="#2A3F54", order=2, ax=ax, 
                line_kws={"linewidth": 2, "zorder": 3})
                
    if len(treated) > 5:
        sns.regplot(data=treated, x="label_exact_geometric_petition_pct", y="council_approved",
                    scatter=False, color="#E74C3C", order=1, ax=ax, 
                    line_kws={"linewidth": 2, "zorder": 3})
    
    ax.axvline(20, color="black", linestyle="--", linewidth=1.5, zorder=0)
    ax.set_title(f"{title} (N={len(df):,})", fontsize=10, weight='bold')
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Approval Prob" if ax.get_subplotspec().is_first_col() else "")
    ax.set_xlabel("Petition %" if ax.get_subplotspec().is_last_row() else "")

def main():
    print("Loading data for ablation simulation...")
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    geom_df = pd.read_csv(GEOM_PATH)
    
    master["case_number"] = master["case_number"].str.strip()
    geom_df["case_number"] = geom_df["case_number"].str.strip()
    
    # Merge outcome
    df = geom_df.merge(master[["case_number", "Derived_Status"]], on="case_number", how="inner")
    
    # Binarize outcome
    df["council_approved"] = df["Derived_Status"].apply(lambda x: 1 if pd.notna(x) and "Approved" in str(x) else 0)
    
    df["threshold_crossed"] = (df["label_exact_geometric_petition_pct"] >= 20).astype(int)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    
    print("1. Plotting Baseline...")
    plot_rd_panel(axes[0, 0], df, "Baseline Model (Full Data)")
    
    print("2. Plotting 30% Global Ablation...")
    df_30 = df.sample(frac=0.70, random_state=42)
    plot_rd_panel(axes[0, 1], df_30, "Random Ablation (Dropped 30%)")
    
    print("3. Plotting 60% Global Ablation...")
    df_60 = df.sample(frac=0.40, random_state=42)
    plot_rd_panel(axes[1, 0], df_60, "Random Ablation (Dropped 60%)")
    
    print("4. Plotting Targeted Donut Hole Ablation...")
    # Drop cases exactly at the threshold (between 15% and 25%)
    # Keep control (0%), far left (<15%), and far right (>25%)
    donut_mask = (df["label_exact_geometric_petition_pct"] < 15) | (df["label_exact_geometric_petition_pct"] > 25)
    df_donut = df[donut_mask]
    plot_rd_panel(axes[1, 1], df_donut, "Donut Hole Ablation (Dropped 15%-25%)")
    
    plt.suptitle("Regression Discontinuity Structural Stability Analysis", fontsize=16, weight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT_PLOT)
    print(f"Simulation complete! Saved to {OUT_PLOT}")

if __name__ == "__main__":
    main()
