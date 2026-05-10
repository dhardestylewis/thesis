import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
import re
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"
GEOM_PATH = rf"{OUT_DIR}\exact_geometric_petition_intensity.csv"
CASE_MASTER = r"C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\Build\case_master.csv"
VOTE_RECORD = r"C:\Users\dhl\data\Thesis\thesis\Data\interim\zoning_cases_with_council_votes.csv"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"
OUT_PLOT = rf"{OUT_DIR}\causal_ml_ablation_sweep.png"

def load_fully_hydrated_data():
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    geom_df = pd.read_csv(GEOM_PATH)
    cm = pd.read_csv(CASE_MASTER, low_memory=False)
    vt = pd.read_csv(VOTE_RECORD, low_memory=False)
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    
    master["case_number"] = master["case_number"].str.strip()
    geom_df["case_number"] = geom_df["case_number"].str.strip()
    cm["CASE_NUMBER"] = cm["CASE_NUMBER"].str.strip()
    
    df = geom_df.merge(master, on="case_number", how="inner")
    
    def clean_status(s):
        if pd.isna(s): return "Unknown"
        s = s.lower()
        if "withdrawn" in s: return "Withdrawn"
        if "denied" in s: return "Denied"
        return "Pending"
    
    cm_status = cm[["CASE_NUMBER", "DETAILED_STATUS"]].drop_duplicates("CASE_NUMBER").copy()
    cm_status["status_cat"] = cm_status["DETAILED_STATUS"].apply(clean_status)
    df = df.merge(cm_status, left_on="case_number", right_on="CASE_NUMBER", how="left")
    
    df["t_denial"] = (df["status_cat"] == "Denied").astype(int)
    df["t_approval"] = df["Derived_Status"].apply(lambda x: 1 if pd.notna(x) and "Approved" in str(x) else 0)
    df["t_withdrawal"] = (df["status_cat"] == "Withdrawn").astype(int)
    
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
    
    panel["case_number"] = panel["case_number"].str.strip()
    hearings = panel.groupby("case_number").agg(t_council_appearances=("council_hearings_this_period", "sum")).reset_index()
    df = df.merge(hearings, on="case_number", how="left")
    df["t_council_appearances"].fillna(0, inplace=True)
    df["t_days_in_pipeline"] = pd.to_numeric(df["Days_in_Pipeline"], errors='coerce').fillna(0)

    panel_sorted = panel.sort_values(["case_number", "period_seq"])
    terminal_panel = panel_sorted.drop_duplicates(subset=["case_number"], keep="last").copy()
    
    pytorch_features = [
        "dist_petition_rate_lag1", "knn_petition_rate_1km",
        "local_unemployment_rate", "bw_sin", "bw_cos",
        "market_value", "appraised_value", "yr_built", "building_age",
        "improvement_ratio", "total_population", "median_household_income",
        "renter_share", "rent_burden", "race_white", "race_hispanic", "median_age",
        "mortgage_rate_30yr", "fed_funds_rate", "treasury_10yr_yield",
        "mortgage_rate_30yr_momentum", "local_unemployment_rate_momentum"
    ]
    terminal_panel = terminal_panel[["case_number"] + pytorch_features]
    df = df.merge(terminal_panel, on="case_number", how="left")

    SPATIAL_PATH = rf"{OUT_DIR}\spatial_attribution_2024.csv"
    spatial_df = pd.read_csv(SPATIAL_PATH, low_memory=False)
    spatial_df["case_number"] = spatial_df["case_number"].str.strip()
    spatial_df = spatial_df.drop_duplicates(subset=["case_number"])
    spatial_df = spatial_df.rename(columns={
        "pct_Architectural": "archetype_pct_Architectural",
        "pct_Bureaucratic": "archetype_pct_Bureaucratic",
        "pct_Economic": "archetype_pct_Economic",
        "pct_Spatial Gravity": "archetype_pct_Spatial_Gravity"
    })
    df = df.merge(spatial_df[["case_number", "archetype_pct_Architectural", "archetype_pct_Bureaucratic", "archetype_pct_Economic", "archetype_pct_Spatial_Gravity"]], on="case_number", how="left")

    covariates = [
        "gross_site_area_acres", "latitude", "longitude", 
        "archetype_pct_Architectural", "archetype_pct_Bureaucratic", "archetype_pct_Economic", "archetype_pct_Spatial_Gravity"
    ] + pytorch_features
    categorical_features = ["council_district", "land_use", "general_land_use", "Staff_Recommendation"]
    
    for c in covariates:
        df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0)
    for cat in categorical_features:
        df[cat] = df.get(cat, "Unknown").fillna("Unknown").astype(str)
        
    return df, covariates + categorical_features, categorical_features

def main():
    print("1. Assembling Master Feature Space...")
    df, features, categorical_features = load_fully_hydrated_data()
    df = df[df["label_exact_geometric_petition_pct"] > 0].copy()
    print(f"Dataset active. N = {len(df)} contested cases.")
    print(f"Total Model Features: {len(features)}")
    
    X = df[features]
    targets = ["t_approval", "t_withdrawal", "t_denial"]
    target_names = ["Approval", "Withdrawal", "Denial"]
    colors = ["#3498DB", "#F39C12", "#E74C3C"]
    
    thresholds = range(5, 96, 5)
    seeds = [42, 100, 2024, 7, 99]
    
    print("\n2. Executing Monte Carlo Causal ML Sweep...")
    
    all_results = {t: {"threshold": [], "mean_cate": [], "mean_lower": [], "mean_upper": [], 
                       "peak_cate": [], "peak_lower": [], "peak_upper": []} for t in targets}
    
    for t_idx, t in enumerate(targets):
        y = df[t]
        print(f"--> Sweeping target: {t}")
        
        for thresh in thresholds:
            T = (df["label_exact_geometric_petition_pct"] >= thresh).astype(int)
            if T.sum() < 5 or (len(T) - T.sum()) < 5:
                continue
                
            idx_0, idx_1 = (T == 0), (T == 1)
            X_0, y_0 = X[idx_0], y[idx_0]
            X_1, y_1 = X[idx_1], y[idx_1]
            pool_all = Pool(X, cat_features=categorical_features)
            
            mean_cates = []
            peak_cates = []
            
            for seed in seeds:
                if y_0.nunique() > 1:
                    model_0 = CatBoostClassifier(iterations=20, learning_rate=0.05, depth=4, verbose=0, random_seed=seed)
                    model_0.fit(Pool(X_0, y_0, cat_features=categorical_features))
                    prob_0 = model_0.predict_proba(pool_all)[:, 1]
                else:
                    prob_0 = np.full(len(X), y_0.mean())
                    
                if y_1.nunique() > 1:
                    model_1 = CatBoostClassifier(iterations=20, learning_rate=0.05, depth=4, verbose=0, random_seed=seed, auto_class_weights='Balanced')
                    model_1.fit(Pool(X_1, y_1, cat_features=categorical_features))
                    prob_1 = model_1.predict_proba(pool_all)[:, 1]
                else:
                    prob_1 = np.full(len(X), y_1.mean())
                
                cate = prob_1 - prob_0
                mean_cates.append(cate.mean())
                peak_cates.append(cate.max() if t == "t_denial" else cate.min())
            
            all_results[t]["threshold"].append(thresh)
            all_results[t]["mean_cate"].append(np.mean(mean_cates))
            all_results[t]["mean_lower"].append(np.percentile(mean_cates, 2.5))
            all_results[t]["mean_upper"].append(np.percentile(mean_cates, 97.5))
            
            all_results[t]["peak_cate"].append(np.mean(peak_cates))
            all_results[t]["peak_lower"].append(np.percentile(peak_cates, 2.5))
            all_results[t]["peak_upper"].append(np.percentile(peak_cates, 97.5))
    
    print("\n3. Rendering Ablation Grid...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=200)
    
    for i, t in enumerate(targets):
        ax = axes[i]
        res = all_results[t]
        
        ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, zorder=0)
        ax.axvline(20, color="#E74C3C", linestyle="-", linewidth=2, alpha=0.5, zorder=1)
        
        # Mean CATE with Bands
        ax.plot(res["threshold"], res["mean_cate"], color="#2A3F54", linewidth=3, label="Average Effect (Mean CATE)", zorder=3)
        ax.fill_between(res["threshold"], res["mean_lower"], res["mean_upper"], color="#2A3F54", alpha=0.2, zorder=2, label="95% CI (Mean)")
        
        # Peak CATE with Bands
        ax.plot(res["threshold"], res["peak_cate"], color=colors[i], linewidth=2, linestyle=":", label="Peak Vulnerability (Max CATE)", zorder=4)
        ax.fill_between(res["threshold"], res["peak_lower"], res["peak_upper"], color=colors[i], alpha=0.15, zorder=2, label="95% CI (Peak)")
        
        ax.set_ylim(-1, 1)
        ax.set_title(f"Monte Carlo Sweep: {target_names[i]}", fontsize=12, weight="bold")
        ax.set_xlabel("Assumed Protest Threshold (%)", fontsize=10)
        if i == 0:
            ax.set_ylabel("Treatment Effect (Change in Probability)", fontsize=10)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        
    plt.suptitle("Causal ML Monte Carlo Ablation Sweep (K=5 Bootstraps)\n(Fully Hydrated: Spatial Blight + NLP + PyTorch Lags + Demographics)", fontsize=16, weight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, bbox_inches="tight")
    print(f"Master plot artifact saved to {OUT_PLOT}")
    
    # Save raw data to CSV
    print("4. Exporting Raw Data Matrix...")
    csv_rows = []
    for t in targets:
        res = all_results[t]
        for idx in range(len(res["threshold"])):
            csv_rows.append({
                "Target": t,
                "Threshold": res["threshold"][idx],
                "Mean_CATE": res["mean_cate"][idx],
                "Mean_Lower_95": res["mean_lower"][idx],
                "Mean_Upper_95": res["mean_upper"][idx],
                "Peak_CATE": res["peak_cate"][idx],
                "Peak_Lower_95": res["peak_lower"][idx],
                "Peak_Upper_95": res["peak_upper"][idx],
            })
    out_df = pd.DataFrame(csv_rows)
    OUT_CSV = rf"{OUT_DIR}\causal_ml_ablation_sweep.csv"
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Master data matrix saved to {OUT_CSV}")

if __name__ == "__main__":
    main()
