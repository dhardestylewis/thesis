import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"
GEOM_PATH = rf"{OUT_DIR}\exact_geometric_petition_intensity.csv"
OUT_PLOT = rf"{OUT_DIR}\rd_unified_ate_placebos.png"

def main():
    print("Loading data for Unified Placebo ATE calculation...")
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    geom_df = pd.read_csv(GEOM_PATH)
    
    master["case_number"] = master["case_number"].str.strip()
    geom_df["case_number"] = geom_df["case_number"].str.strip()
    
    df = geom_df.merge(master[["case_number", "Derived_Status"]], on="case_number", how="inner")
    df["council_approved"] = df["Derived_Status"].apply(lambda x: 1 if pd.notna(x) and "Approved" in str(x) else 0)
    
    # Filter for cases around the thresholds to emulate local linear regression bandwidth
    # Bandwidth of +/- 15 around the threshold
    bandwidth = 15
    
    cutoffs = range(5, 96)
    ate_estimates = []
    ate_ci_lower_95 = []
    ate_ci_upper_95 = []
    ate_ci_lower_90 = []
    ate_ci_upper_90 = []
    
    print("Calculating ATE jump for every integer cutoff from 5% to 95%...")
    for cutoff in cutoffs:
        # Localize data within the bandwidth around the current fake cutoff
        local_df = df[(df["label_exact_geometric_petition_pct"] >= cutoff - bandwidth) & 
                      (df["label_exact_geometric_petition_pct"] <= cutoff + bandwidth)].copy()
        
        # We need enough data on both sides to run the regression
        if len(local_df[local_df["label_exact_geometric_petition_pct"] >= cutoff]) < 3 or \
           len(local_df[local_df["label_exact_geometric_petition_pct"] < cutoff]) < 3:
            ate_estimates.append(np.nan)
            ate_ci_lower_95.append(np.nan)
            ate_ci_upper_95.append(np.nan)
            ate_ci_lower_90.append(np.nan)
            ate_ci_upper_90.append(np.nan)
            continue
            
        local_df["centered_pct"] = local_df["label_exact_geometric_petition_pct"] - cutoff
        local_df["threshold_crossed"] = (local_df["label_exact_geometric_petition_pct"] >= cutoff).astype(int)
        
        # Local Linear Regression with interaction term
        model = smf.ols("council_approved ~ centered_pct * threshold_crossed", data=local_df).fit()
        
        # The coefficient on 'threshold_crossed' is the exact size of the jump at X=cutoff
        jump = model.params["threshold_crossed"]
        conf_95 = model.conf_int(alpha=0.05).loc["threshold_crossed"]
        conf_90 = model.conf_int(alpha=0.10).loc["threshold_crossed"]
        
        ate_estimates.append(jump)
        ate_ci_lower_95.append(conf_95[0])
        ate_ci_upper_95.append(conf_95[1])
        ate_ci_lower_90.append(conf_90[0])
        ate_ci_upper_90.append(conf_90[1])
        
    print("Plotting Unified Placebo Curve...")
    plt.figure(figsize=(12, 6), dpi=200)
    
    plt.axhline(0, color="gray", linestyle="--", linewidth=1.5, zorder=0)
    plt.axvline(20, color="#E74C3C", linestyle="-", linewidth=2, alpha=0.5, zorder=1)
    
    # 95% Confidence Band (Lighter, Wider)
    plt.fill_between(cutoffs, ate_ci_lower_95, ate_ci_upper_95, color="#3498DB", alpha=0.15, zorder=2, label="95% CI")
    # 90% Confidence Band (Darker, Tighter)
    plt.fill_between(cutoffs, ate_ci_lower_90, ate_ci_upper_90, color="#3498DB", alpha=0.35, zorder=3, label="90% CI")
    
    plt.plot(cutoffs, ate_estimates, color="#2A3F54", linewidth=2.5, marker="o", markersize=3, zorder=4)
    
    # Dynamic Y limits to prevent extreme explosion from sparse tails
    valid_lower = [v for v in ate_ci_lower_95 if not np.isnan(v)]
    if valid_lower:
        plt.ylim(max(-2, min(valid_lower) - 0.2), 1)
        
    plt.text(20.5, plt.ylim()[0] + 0.1, 
             "True Legal Threshold (20%)", color="#E74C3C", weight="bold", rotation=90, verticalalignment="bottom", zorder=5)
    
    plt.title("Treatment Effect Stability Across Placebo Thresholds", fontsize=14, weight="bold")
    plt.xlabel("Assumed Legal Threshold (Petition %)", fontsize=12)
    plt.ylabel("Estimated Size of ATE Jump\n(Drop in Approval Probability)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(OUT_PLOT)
    print(f"Artifact saved to {OUT_PLOT}")

if __name__ == "__main__":
    main()
