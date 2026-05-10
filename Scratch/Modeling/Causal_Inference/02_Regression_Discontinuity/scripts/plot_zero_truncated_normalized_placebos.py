import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import re
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"
GEOM_PATH = rf"{OUT_DIR}\exact_geometric_petition_intensity.csv"
CASE_MASTER = r"C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\Build\case_master.csv"
VOTE_RECORD = r"C:\Users\dhl\data\Thesis\thesis\Data\interim\zoning_cases_with_council_votes.csv"
OUT_PLOT = rf"{OUT_DIR}\rd_zero_truncated_normalized_placebos.png"

def calculate_ate(df, target_col):
    bandwidth = 15
    cutoffs = range(5, 96)
    
    results = {"cutoff": [], "ate": [], "ci_lower_95": [], "ci_upper_95": [], "ci_lower_90": [], "ci_upper_90": []}
    
    for cutoff in cutoffs:
        local_df = df[(df["label_exact_geometric_petition_pct"] >= cutoff - bandwidth) & 
                      (df["label_exact_geometric_petition_pct"] <= cutoff + bandwidth)].copy()
        
        # Require minimum valid cases on both sides
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

def plot_target_panel(ax, results_df, title, color_theme, is_left_edge=False):
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, zorder=0)
    ax.axvline(20, color="#E74C3C", linestyle="-", linewidth=2, alpha=0.5, zorder=1)
    
    cutoffs = results_df["cutoff"]
    
    # Fill CI bands
    ax.fill_between(cutoffs, results_df["ci_lower_95"], results_df["ci_upper_95"], 
                    color=color_theme, alpha=0.15, zorder=2)
    ax.fill_between(cutoffs, results_df["ci_lower_90"], results_df["ci_upper_90"], 
                    color=color_theme, alpha=0.35, zorder=3)
    
    # Plot solid ATE line
    ax.plot(cutoffs, results_df["ate"], color="#2A3F54", linewidth=2.5, marker="o", markersize=2, zorder=4)
    
    # Fixed Normalized Limits so all panels are identical scale
    ax.set_ylim(-3, 3)
    
    ax.text(20.5, -2.8, "Legal Threshold (20%)", color="#E74C3C", weight="bold", 
            rotation=90, verticalalignment="bottom", zorder=5, fontsize=8)
             
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("Assumed Legal Threshold (Petition %)", fontsize=9)
    if is_left_edge:
        ax.set_ylabel("Normalized ATE Jump (Z-Score)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

def main():
    print("Loading data layers...")
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    geom_df = pd.read_csv(GEOM_PATH)
    cm = pd.read_csv(CASE_MASTER, low_memory=False)
    vt = pd.read_csv(VOTE_RECORD, low_memory=False)
    
    master["case_number"] = master["case_number"].str.strip()
    geom_df["case_number"] = geom_df["case_number"].str.strip()
    cm["CASE_NUMBER"] = cm["CASE_NUMBER"].str.strip()
    
    # 1. Base Merge
    df = geom_df.merge(master, on="case_number", how="inner")
    
    # 2. Extract Detailed Status
    def clean_status(s):
        if pd.isna(s): return "Unknown"
        s = s.lower()
        if "withdrawn" in s: return "Withdrawn"
        if "denied" in s: return "Denied"
        if "closed" in s or "void" in s or "expired" in s: return "Passive_Death"
        return "Pending"
        
    cm_status = cm[["CASE_NUMBER", "DETAILED_STATUS"]].drop_duplicates("CASE_NUMBER").copy()
    cm_status["status_cat"] = cm_status["DETAILED_STATUS"].apply(clean_status)
    df = df.merge(cm_status, left_on="case_number", right_on="CASE_NUMBER", how="left")
    
    # 3. Extract Max Nay Votes
    vt_df = []
    vote_pattern = re.compile(r'\b(\d{1,2})-(\d{1,2})\s*vote\b', re.IGNORECASE)
    for _, row in vt.iterrows():
        matches = vote_pattern.findall(str(row["Vote_Transcript"]))
        if matches:
            for m in matches:
                yes, no = int(m[0]), int(m[1])
                if 3 <= (yes + no) <= 11:
                    vt_df.append({"case_number": row["Case_Number"], "no_votes": no})
    
    vt_agg = pd.DataFrame(vt_df)
    if not vt_agg.empty:
        vt_agg = vt_agg.groupby("case_number").agg(t_max_nay_votes=("no_votes", "max")).reset_index()
        vt_agg["case_number"] = vt_agg["case_number"].str.strip()
        df = df.merge(vt_agg, on="case_number", how="left")
    else:
        df["t_max_nay_votes"] = 0
    df["t_max_nay_votes"].fillna(0, inplace=True)
    
    PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    
    # 4. Target Engineering
    df["t_approval"] = df["Derived_Status"].apply(lambda x: 1 if pd.notna(x) and "Approved" in str(x) else 0)
    df["t_withdrawal"] = (df["status_cat"] == "Withdrawn").astype(float)
    df["t_denial"] = (df["status_cat"] == "Denied").astype(float)
    df["t_passive_death"] = (df["status_cat"] == "Passive_Death").astype(float)
    
    hearings = panel.groupby("case_number").agg(t_council_appearances=("council_hearings_this_period", "sum")).reset_index()
    hearings["case_number"] = hearings["case_number"].str.strip()
    df = df.merge(hearings, on="case_number", how="left")
    df["t_council_appearances"].fillna(0, inplace=True)
    
    # Days in Pipeline
    df["t_days_in_pipeline"] = pd.to_numeric(df["Days_in_Pipeline"], errors='coerce').fillna(0)
    
    # ==========================================
    # ZERO TRUNCATION
    # ==========================================
    pre_trunc = len(df)
    df = df[df["label_exact_geometric_petition_pct"] > 0].copy()
    print(f"Zero-Truncated dataset active. Dropped {pre_trunc - len(df)} uncontested cases. N = {len(df)} contested cases remain.")
    
    # ==========================================
    # Z-SCORE NORMALIZATION
    # ==========================================
    targets = [
        "t_approval", "t_withdrawal", "t_denial", "t_passive_death", 
        "t_council_appearances", "t_max_nay_votes", "t_days_in_pipeline"
    ]
    
    for t in targets:
        mean_val = df[t].mean()
        std_val = df[t].std()
        # Avoid div by 0
        if std_val > 0:
            df[f"z_{t}"] = (df[t] - mean_val) / std_val
        else:
            df[f"z_{t}"] = 0.0
            
    print("Executing Local Linear Regression Discontinuity Sweeps...")
    results = {}
    titles = [
        "1. Final Approval Probability",
        "2. Formal Withdrawal Probability",
        "3. Outright Denial Probability",
        "4. Passive Death (Expired) Probability",
        "5. Total Council Appearances",
        "6. Max Dissenting Nay Votes",
        "7. Total Days in Pipeline"
    ]
    
    for t, title in zip(targets, titles):
        print(f"  -> Sweeping {title}...")
        results[t] = calculate_ate(df, f"z_{t}")
        
    print("Rendering Master Grid...")
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), dpi=200)
    axes = axes.flatten()
    
    colors = ["#27AE60", "#F39C12", "#E74C3C", "#95A5A6", "#8E44AD", "#D35400", "#2980B9"]
    
    for i, t in enumerate(targets):
        is_left = (i == 0 or i == 4)
        plot_target_panel(axes[i], results[t], titles[i], colors[i], is_left_edge=is_left)
        
    # Hide the 8th empty subplot
    axes[7].set_visible(False)
    
    plt.suptitle("Multi-Target Causal Falsification Matrix (Zero-Truncated & Normalized)", fontsize=22, weight="bold", y=0.97)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Footer caption
    fig.text(0.5, 0.01, 
             "Methodology: Local Linear Regression (BW=15). 0% Intensity cases explicitly dropped to eliminate zero-inflation bandwidth artifact. Shaded bands represent 90% and 95% Confidence Intervals.", 
             ha="center", fontsize=11, fontstyle="italic")
             
    plt.savefig(OUT_PLOT)
    print(f"Master artifact saved to {OUT_PLOT}")

if __name__ == "__main__":
    main()
