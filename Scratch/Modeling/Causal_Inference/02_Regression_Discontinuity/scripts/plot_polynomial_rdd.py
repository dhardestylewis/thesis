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
OUT_PLOT = rf"{OUT_DIR}\rd_polynomial_robustness.png"

def fit_and_plot_cubic(ax, df, target_col, title, color_theme):
    # Global domain (excluding exactly 0, and capping at 100)
    cutoff = 20
    local_df = df.copy()
    local_df["centered_pct"] = local_df["label_exact_geometric_petition_pct"] - cutoff
    local_df["centered_pct_sq"] = local_df["centered_pct"] ** 2
    local_df["centered_pct_cu"] = local_df["centered_pct"] ** 3
    local_df["threshold_crossed"] = (local_df["label_exact_geometric_petition_pct"] >= cutoff).astype(int)
    
    # Fit Global Cubic Model
    formula = f"{target_col} ~ (centered_pct + centered_pct_sq + centered_pct_cu) * threshold_crossed"
    model = smf.ols(formula, data=local_df).fit()
    
    # Calculate binned scatter points for visual clarity
    bins = np.linspace(0, 100, 20)
    local_df['bin'] = pd.cut(local_df["label_exact_geometric_petition_pct"], bins=bins)
    bin_means = local_df.groupby('bin')[target_col].mean()
    bin_centers = local_df.groupby('bin')["label_exact_geometric_petition_pct"].mean()
    
    # Scatter the empirical bins
    ax.scatter(bin_centers, bin_means, color="gray", alpha=0.6, s=50, edgecolors='white', zorder=2, label="Empirical Binned Mean")
    
    # Generate continuous lines for the polynomial fit
    left_x = np.linspace(0, 19.99, 100)
    right_x = np.linspace(20, 100, 100)
    
    def predict_cubic(x_array, thresh_val):
        c = x_array - cutoff
        temp = pd.DataFrame({
            "centered_pct": c,
            "centered_pct_sq": c**2,
            "centered_pct_cu": c**3,
            "threshold_crossed": thresh_val
        })
        return model.predict(temp)
        
    left_y = predict_cubic(left_x, 0)
    right_y = predict_cubic(right_x, 1)
    
    # Plot polynomial curves
    ax.plot(left_x, left_y, color=color_theme, linewidth=3, zorder=3, label="Global Cubic Fit")
    ax.plot(right_x, right_y, color=color_theme, linewidth=3, zorder=3)
    
    # Formatting
    ax.axvline(20, color="#E74C3C", linestyle="--", linewidth=2, zorder=1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, 100)
    
    jump = model.params["threshold_crossed"]
    pval = model.pvalues["threshold_crossed"]
    
    # Display the regression jump and p-value
    text_str = f"Cubic ATE Jump at 20%: {jump:+.3f}\nP-Value: {pval:.3f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray')
    ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', horizontalalignment='right', bbox=props, zorder=5)
            
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Petition Intensity (%)", fontsize=10)
    ax.set_ylabel("Probability", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)

def main():
    print("Loading data...")
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    geom_df = pd.read_csv(GEOM_PATH)
    cm = pd.read_csv(CASE_MASTER, low_memory=False)
    
    master["case_number"] = master["case_number"].str.strip()
    geom_df["case_number"] = geom_df["case_number"].str.strip()
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
    
    df["t_approval"] = df["Derived_Status"].apply(lambda x: 1 if pd.notna(x) and "Approved" in str(x) else 0)
    df["t_withdrawal"] = (df["status_cat"] == "Withdrawn").astype(float)
    df["t_denial"] = (df["status_cat"] == "Denied").astype(float)
    
    # Zero-Truncation
    df = df[df["label_exact_geometric_petition_pct"] > 0].copy()
    print(f"Zero-Truncated dataset active. N = {len(df)} contested cases.")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=200)
    
    print("Fitting Global Cubic RDD for Approval...")
    fit_and_plot_cubic(axes[0], df, "t_approval", "Approval Probability (Global Cubic RDD)", "#3498DB")
    print("Fitting Global Cubic RDD for Withdrawal...")
    fit_and_plot_cubic(axes[1], df, "t_withdrawal", "Withdrawal Probability (Global Cubic RDD)", "#F39C12")
    print("Fitting Global Cubic RDD for Denial...")
    fit_and_plot_cubic(axes[2], df, "t_denial", "Denial Probability (Global Cubic RDD)", "#E74C3C")
    
    plt.suptitle("Global Parametric RDD Robustness Check (Zero-Truncated, Cubic Fit)", fontsize=16, weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, bbox_inches="tight")
    print(f"Artifact saved to {OUT_PLOT}")

if __name__ == "__main__":
    main()
