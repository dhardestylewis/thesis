import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"
GEOM_PATH = rf"{OUT_DIR}\exact_geometric_petition_intensity.csv"
OUT_PLOT = rf"{OUT_DIR}\rd_zero_truncated_placebos.png"

def calculate_ate(df, target_col):
    bandwidth = 15
    cutoffs = range(5, 96)
    
    results = {"cutoff": [], "ate": [], "ci_lower_95": [], "ci_upper_95": [], "ci_lower_90": [], "ci_upper_90": []}
    
    for cutoff in cutoffs:
        local_df = df[(df["label_exact_geometric_petition_pct"] >= cutoff - bandwidth) & 
                      (df["label_exact_geometric_petition_pct"] <= cutoff + bandwidth)].copy()
        
        if len(local_df[local_df["label_exact_geometric_petition_pct"] >= cutoff]) < 3 or \
           len(local_df[local_df["label_exact_geometric_petition_pct"] < cutoff]) < 3:
            results["cutoff"].append(cutoff)
            results["ate"].append(np.nan)
            results["ci_lower_95"].append(np.nan)
            results["ci_upper_95"].append(np.nan)
            results["ci_lower_90"].append(np.nan)
            results["ci_upper_90"].append(np.nan)
            continue
            
        local_df["centered_pct"] = local_df["label_exact_geometric_petition_pct"] - cutoff
        local_df["threshold_crossed"] = (local_df["label_exact_geometric_petition_pct"] >= cutoff).astype(int)
        
        try:
            model = smf.ols(f"{target_col} ~ centered_pct * threshold_crossed", data=local_df).fit()
            jump = model.params["threshold_crossed"]
            conf_95 = model.conf_int(alpha=0.05).loc["threshold_crossed"]
            conf_90 = model.conf_int(alpha=0.10).loc["threshold_crossed"]
            
            results["cutoff"].append(cutoff)
            results["ate"].append(jump)
            results["ci_lower_95"].append(conf_95[0])
            results["ci_upper_95"].append(conf_95[1])
            results["ci_lower_90"].append(conf_90[0])
            results["ci_upper_90"].append(conf_90[1])
        except:
            results["cutoff"].append(cutoff)
            results["ate"].append(np.nan)
            results["ci_lower_95"].append(np.nan)
            results["ci_upper_95"].append(np.nan)
            results["ci_lower_90"].append(np.nan)
            results["ci_upper_90"].append(np.nan)
            
    return pd.DataFrame(results)

def plot_target_panel(ax, results_df, title, color_theme, y_label=False):
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, zorder=0)
    ax.axvline(20, color="#E74C3C", linestyle="-", linewidth=2, alpha=0.5, zorder=1)
    
    cutoffs = results_df["cutoff"]
    
    # 95% CI
    ax.fill_between(cutoffs, results_df["ci_lower_95"], results_df["ci_upper_95"], 
                    color=color_theme, alpha=0.15, zorder=2)
    # 90% CI
    ax.fill_between(cutoffs, results_df["ci_lower_90"], results_df["ci_upper_90"], 
                    color=color_theme, alpha=0.35, zorder=3)
    
    # Point Estimate Line
    ax.plot(cutoffs, results_df["ate"], color="#2A3F54", linewidth=2.5, marker="o", markersize=3, zorder=4)
    
    # Dynamic Y limits
    valid_lower = results_df["ci_lower_95"].dropna()
    if not valid_lower.empty:
        min_y = max(-2, valid_lower.min() - 0.1)
        max_y = min(2, results_df["ci_upper_95"].max() + 0.1)
        ax.set_ylim(min_y, max_y)
        
    ax.text(20.5, ax.get_ylim()[0] + 0.1, 
             "Legal Threshold (20%)", color="#E74C3C", weight="bold", rotation=90, verticalalignment="bottom", zorder=5)
             
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Assumed Legal Threshold (Petition %)", fontsize=10)
    if y_label:
        ax.set_ylabel("ATE Jump Magnitude at Cutoff", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

def main():
    print("Loading data...")
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    geom_df = pd.read_csv(GEOM_PATH)
    
    master["case_number"] = master["case_number"].str.strip()
    geom_df["case_number"] = geom_df["case_number"].str.strip()
    
    CASE_MASTER = r"C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\Build\case_master.csv"
    cm = pd.read_csv(CASE_MASTER, low_memory=False)
    cm["CASE_NUMBER"] = cm["CASE_NUMBER"].str.strip()
    
    def clean_status(s):
        if pd.isna(s): return "Unknown"
        s = s.lower()
        if "withdrawn" in s: return "Withdrawn"
        if "denied" in s: return "Denied"
        return "Pending"
        
    cm_status = cm[["CASE_NUMBER", "DETAILED_STATUS"]].drop_duplicates("CASE_NUMBER").copy()
    cm_status["status_cat"] = cm_status["DETAILED_STATUS"].apply(clean_status)
    
    df = geom_df.merge(master[["case_number", "Derived_Status"]], on="case_number", how="inner")
    df = df.merge(cm_status, left_on="case_number", right_on="CASE_NUMBER", how="left")
    
    # Engineer the 3 binary targets
    df["target_approved"] = df["Derived_Status"].apply(lambda x: 1 if pd.notna(x) and "Approved" in str(x) else 0)
    df["target_withdrawn"] = (df["status_cat"] == "Withdrawn").astype(float)
    df["target_denied"] = (df["status_cat"] == "Denied").astype(float)
    
    # ==========================================
    # ZERO TRUNCATION: Remove the 0% Inflation!
    # ==========================================
    df = df[df["label_exact_geometric_petition_pct"] > 0]
    print(f"Zero-Truncated dataset active. N = {len(df)} contested cases.")
    
    print("Calculating ATE for Approval...")
    res_app = calculate_ate(df, "target_approved")
    print("Calculating ATE for Withdrawal...")
    res_with = calculate_ate(df, "target_withdrawn")
    print("Calculating ATE for Denial...")
    res_den = calculate_ate(df, "target_denied")
    
    print("Plotting Multi-Target Zero-Truncated Grid...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)
    
    plot_target_panel(axes[0], res_app, "Approval Probability (Zero-Truncated)", "#3498DB", y_label=True)
    plot_target_panel(axes[1], res_with, "Withdrawal Probability (Zero-Truncated)", "#F39C12")
    plot_target_panel(axes[2], res_den, "Denial Probability (Zero-Truncated)", "#E74C3C")
    
    plt.suptitle("Multi-Target Treatment Effect Stability (Zero-Inflated Cases Excluded)", fontsize=16, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_PLOT)
    print(f"Artifact saved to {OUT_PLOT}")

if __name__ == "__main__":
    main()
