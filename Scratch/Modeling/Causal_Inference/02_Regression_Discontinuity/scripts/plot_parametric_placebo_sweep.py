import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"
GEOM_PATH = rf"{OUT_DIR}\exact_geometric_petition_intensity.csv"
CASE_MASTER = r"C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\Build\case_master.csv"
OUT_PLOT = rf"{OUT_DIR}\rd_parametric_placebo_sweep.png"

def _extract_results(model, cutoff, results_dict):
    try:
        jump = model.params["threshold_crossed"]
        conf_95 = model.conf_int(alpha=0.05).loc["threshold_crossed"]
        
        results_dict["cutoff"].append(cutoff)
        results_dict["ate"].append(jump)
        results_dict["ci_lower_95"].append(conf_95[0])
        results_dict["ci_upper_95"].append(conf_95[1])
    except:
        results_dict["cutoff"].append(cutoff)
        results_dict["ate"].append(np.nan)
        results_dict["ci_lower_95"].append(np.nan)
        results_dict["ci_upper_95"].append(np.nan)

def calculate_ate_linear(df, target_col):
    bandwidth = 15
    cutoffs = range(5, 96)
    results = {"cutoff": [], "ate": [], "ci_lower_95": [], "ci_upper_95": []}
    
    for cutoff in cutoffs:
        local_df = df[(df["label_exact_geometric_petition_pct"] >= cutoff - bandwidth) & 
                      (df["label_exact_geometric_petition_pct"] <= cutoff + bandwidth)].copy()
        
        if len(local_df[local_df["label_exact_geometric_petition_pct"] >= cutoff]) < 3 or \
           len(local_df[local_df["label_exact_geometric_petition_pct"] < cutoff]) < 3:
            _extract_results(None, cutoff, results)
            continue
            
        local_df["centered_pct"] = local_df["label_exact_geometric_petition_pct"] - cutoff
        local_df["threshold_crossed"] = (local_df["label_exact_geometric_petition_pct"] >= cutoff).astype(int)
        
        try:
            model = smf.ols(f"{target_col} ~ centered_pct * threshold_crossed", data=local_df).fit()
            _extract_results(model, cutoff, results)
        except:
            _extract_results(None, cutoff, results)
            
    return pd.DataFrame(results)

def calculate_ate_quadratic(df, target_col):
    cutoffs = range(5, 96)
    results = {"cutoff": [], "ate": [], "ci_lower_95": [], "ci_upper_95": []}
    
    for cutoff in cutoffs:
        local_df = df.copy()
        local_df["centered_pct"] = local_df["label_exact_geometric_petition_pct"] - cutoff
        local_df["centered_pct_sq"] = local_df["centered_pct"] ** 2
        local_df["threshold_crossed"] = (local_df["label_exact_geometric_petition_pct"] >= cutoff).astype(int)
        
        try:
            formula = f"{target_col} ~ (centered_pct + centered_pct_sq) * threshold_crossed"
            model = smf.ols(formula, data=local_df).fit()
            _extract_results(model, cutoff, results)
        except:
            _extract_results(None, cutoff, results)
            
    return pd.DataFrame(results)

def calculate_ate_cubic(df, target_col):
    cutoffs = range(5, 96)
    results = {"cutoff": [], "ate": [], "ci_lower_95": [], "ci_upper_95": []}
    
    for cutoff in cutoffs:
        local_df = df.copy()
        local_df["centered_pct"] = local_df["label_exact_geometric_petition_pct"] - cutoff
        local_df["centered_pct_sq"] = local_df["centered_pct"] ** 2
        local_df["centered_pct_cu"] = local_df["centered_pct"] ** 3
        local_df["threshold_crossed"] = (local_df["label_exact_geometric_petition_pct"] >= cutoff).astype(int)
        
        try:
            formula = f"{target_col} ~ (centered_pct + centered_pct_sq + centered_pct_cu) * threshold_crossed"
            model = smf.ols(formula, data=local_df).fit()
            _extract_results(model, cutoff, results)
        except:
            _extract_results(None, cutoff, results)
            
    return pd.DataFrame(results)

def plot_sweep_panel(ax, results_df, title, color_theme, is_left_edge=False, is_bottom_edge=False):
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, zorder=0)
    ax.axvline(20, color="#E74C3C", linestyle="-", linewidth=2, alpha=0.5, zorder=1)
    
    cutoffs = results_df["cutoff"]
    
    ax.fill_between(cutoffs, results_df["ci_lower_95"], results_df["ci_upper_95"], 
                    color=color_theme, alpha=0.25, zorder=2)
    ax.plot(cutoffs, results_df["ate"], color="#2A3F54", linewidth=2, marker="o", markersize=2, zorder=4)
    
    valid_lower = results_df["ci_lower_95"].dropna()
    if not valid_lower.empty:
        # Cap the limits to prevent massive explosion warping the visual
        min_y = max(-1.5, valid_lower.min() - 0.1)
        max_y = min(1.5, results_df["ci_upper_95"].max() + 0.1)
        ax.set_ylim(min_y, max_y)
    else:
        ax.set_ylim(-1, 1)
        
    ax.set_title(title, fontsize=11, weight="bold")
    
    if is_bottom_edge:
        ax.set_xlabel("Assumed Threshold (%)", fontsize=10)
    if is_left_edge:
        ax.set_ylabel("ATE Jump", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

def main():
    print("Loading data layers...")
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    geom_df = pd.read_csv(GEOM_PATH)
    cm = pd.read_csv(CASE_MASTER, low_memory=False)
    
    master["case_number"] = master["case_number"].str.strip()
    geom_df["case_number"] = geom_df["case_number"].str.strip()
    cm["CASE_NUMBER"] = cm["CASE_NUMBER"].str.strip()
    
    df = geom_df.merge(master[["case_number", "Derived_Status"]], on="case_number", how="inner")
    
    def clean_status(s):
        if pd.isna(s): return "Unknown"
        s = s.lower()
        if "withdrawn" in s: return "Withdrawn"
        if "denied" in s: return "Denied"
        return "Pending"
        
    cm_status = cm[["CASE_NUMBER", "DETAILED_STATUS"]].drop_duplicates("CASE_NUMBER").copy()
    cm_status["status_cat"] = cm_status["DETAILED_STATUS"].apply(clean_status)
    df = df.merge(cm_status, left_on="case_number", right_on="CASE_NUMBER", how="left")
    
    df["t_approval"] = df["Derived_Status"].apply(lambda x: 1 if pd.notna(x) and "Approved" in str(x) else 0)
    df["t_withdrawal"] = (df["status_cat"] == "Withdrawn").astype(float)
    df["t_denial"] = (df["status_cat"] == "Denied").astype(float)
    
    # ZERO TRUNCATION
    df = df[df["label_exact_geometric_petition_pct"] > 0].copy()
    print(f"Zero-Truncated dataset active. N = {len(df)} contested cases.")
    
    targets = ["t_approval", "t_withdrawal", "t_denial"]
    target_names = ["Approval", "Withdrawal", "Denial"]
    colors = ["#3498DB", "#F39C12", "#E74C3C"]
    
    print("Executing Falsification Sweeps...")
    all_results = {}
    for t in targets:
        print(f"  -> Sweeping {t} (Linear)")
        res_lin = calculate_ate_linear(df, t)
        print(f"  -> Sweeping {t} (Quadratic)")
        res_quad = calculate_ate_quadratic(df, t)
        print(f"  -> Sweeping {t} (Cubic)")
        res_cub = calculate_ate_cubic(df, t)
        
        all_results[t] = {"Linear": res_lin, "Quadratic": res_quad, "Cubic": res_cub}
    
    print("Rendering 3x3 Master Grid...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), dpi=200)
    
    models = ["Linear", "Quadratic", "Cubic"]
    for row, t in enumerate(targets):
        for col, m in enumerate(models):
            ax = axes[row, col]
            is_left = (col == 0)
            is_bottom = (row == 2)
            title = f"{target_names[row]} | {m} Model"
            
            res_df = all_results[t][m]
            plot_sweep_panel(ax, res_df, title, colors[row], is_left_edge=is_left, is_bottom_edge=is_bottom)
            
    plt.suptitle("Parametric Falsification Sweep Matrix\n(Linear vs. Quadratic vs. Cubic Models | Zero-Truncated Data)", 
                 fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_PLOT)
    print(f"Master artifact saved to {OUT_PLOT}")

if __name__ == "__main__":
    main()
