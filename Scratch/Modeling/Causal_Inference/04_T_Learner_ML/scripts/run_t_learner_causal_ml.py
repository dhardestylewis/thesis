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
OUT_PLOT = rf"{OUT_DIR}\causal_ml_cate_distribution.png"

def main():
    print("1. Assembling Master Feature Space...")
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    geom_df = pd.read_csv(GEOM_PATH)
    cm = pd.read_csv(CASE_MASTER, low_memory=False)
    vt = pd.read_csv(VOTE_RECORD, low_memory=False)
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    
    master["case_number"] = master["case_number"].str.strip()
    geom_df["case_number"] = geom_df["case_number"].str.strip()
    cm["CASE_NUMBER"] = cm["CASE_NUMBER"].str.strip()
    
    # Base merge
    df = geom_df.merge(master, on="case_number", how="inner")
    
    # Target Extraction
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
    
    # NLP & Friction Features
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
    
    # 1. Total Hearings
    hearings = panel.groupby("case_number").agg(t_council_appearances=("council_hearings_this_period", "sum")).reset_index()
    df = df.merge(hearings, on="case_number", how="left")
    df["t_council_appearances"].fillna(0, inplace=True)
    df["t_days_in_pipeline"] = pd.to_numeric(df["Days_in_Pipeline"], errors='coerce').fillna(0)

    # 2. Extract Terminal State of PyTorch LSTM Features
    panel_sorted = panel.sort_values(["case_number", "period_seq"])
    terminal_panel = panel_sorted.drop_duplicates(subset=["case_number"], keep="last").copy()
    
    # The exact Top 15 features identified by GradientSHAP
    pytorch_features = [
        "existing_max_height_ft", "proposed_max_height_ft", 
        "existing_max_far", "proposed_max_far",
        "dist_petition_rate_lag1", "knn_petition_rate_1km",
        "local_unemployment_rate", "bw_sin",
        "cumulative_petition_events", "cumulative_commission_hearings"
    ]
    terminal_panel = terminal_panel[["case_number"] + pytorch_features]
    df = df.merge(terminal_panel, on="case_number", how="left")

    SPATIAL_PATH = rf"{OUT_DIR}\spatial_attribution_2024.csv"
    spatial_df = pd.read_csv(SPATIAL_PATH, low_memory=False)
    spatial_df["case_number"] = spatial_df["case_number"].str.strip()
    
    # Drop duplicates in spatial to avoid cartesian explosions
    spatial_df = spatial_df.drop_duplicates(subset=["case_number"])
    df = df.merge(spatial_df[["case_number", "archetype_pct_Architectural", "archetype_pct_Bureaucratic", "archetype_pct_Economic", "archetype_pct_Spatial_Gravity"]], on="case_number", how="left")

    # Select Covariates for ML
    covariates = [
        "gross_site_area_acres", "latitude", "longitude", 
        "t_days_in_pipeline", "t_council_appearances", "t_max_nay_votes",
        "Aggregate_Sentiment", "Opposition_Volume", "Support_Volume", "Remand_Count",
        "archetype_pct_Architectural", "archetype_pct_Bureaucratic", "archetype_pct_Economic", "archetype_pct_Spatial_Gravity"
    ] + pytorch_features
    categorical_features = ["council_district", "land_use", "general_land_use", "Staff_Recommendation"]
    
    for c in covariates:
        df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0)
        
    for cat in categorical_features:
        df[cat] = df.get(cat, "Unknown").fillna("Unknown").astype(str)
    
    features = covariates + categorical_features
    X = df[features]
    y = df["t_denial"]
    T = (df["label_exact_geometric_petition_pct"] >= 20).astype(int)
    
    print(f"Total Cases: {len(df)}")
    print(f"Treated (>=20%): {T.sum()}")
    print(f"Control (<20%): {len(T) - T.sum()}")
    
    print("\n2. Initializing CatBoost T-Learner (Causal ML)...")
    
    # Train M0 on Control group (T=0)
    print("  -> Training Model M0 (Control Group)...")
    idx_0 = T == 0
    X_0, y_0 = X[idx_0], y[idx_0]
    pool_0 = Pool(X_0, y_0, cat_features=categorical_features)
    model_0 = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, verbose=0, random_seed=42)
    model_0.fit(pool_0)
    
    # Train M1 on Treated group (T=1)
    print("  -> Training Model M1 (Treated Group)...")
    idx_1 = T == 1
    X_1, y_1 = X[idx_1], y[idx_1]
    pool_1 = Pool(X_1, y_1, cat_features=categorical_features)
    
    # Use class weights if Treated group is highly imbalanced
    model_1 = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=4, verbose=0, random_seed=42, auto_class_weights='Balanced')
    model_1.fit(pool_1)
    
    print("\n3. Calculating Conditional Average Treatment Effects (CATE)...")
    # Predict Y(0) and Y(1) for ALL cases
    pool_all = Pool(X, cat_features=categorical_features)
    prob_0 = model_0.predict_proba(pool_all)[:, 1]
    prob_1 = model_1.predict_proba(pool_all)[:, 1]
    
    cate = prob_1 - prob_0
    df["CATE"] = cate
    
    print("Rendering CATE Distribution Plot...")
    plt.figure(figsize=(10, 6), dpi=200)
    
    sns.histplot(df["CATE"], bins=50, color="#8E44AD", kde=True, alpha=0.6)
    plt.axvline(0, color="gray", linestyle="--", linewidth=2, zorder=3)
    plt.axvline(df["CATE"].mean(), color="#E74C3C", linestyle="-", linewidth=2, label=f"Mean CATE (Global ATE): {df['CATE'].mean():+.3f}")
    
    # Highlight potential Heterogeneous pockets
    sig_pos = df[df["CATE"] > 0.15]
    if not sig_pos.empty:
        plt.axvspan(0.15, df["CATE"].max(), color="#E74C3C", alpha=0.1, label=f"Vulnerable Pockets (N={len(sig_pos)})")
        
    plt.title("Heterogeneous Treatment Effects (CATE)\nCatBoost T-Learner Output for 20% Threshold", fontsize=14, weight="bold")
    plt.xlabel("Individual Treatment Effect (Change in Denial Probability)", fontsize=12)
    plt.ylabel("Number of Zoning Cases", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(OUT_PLOT)
    print(f"Master artifact saved to {OUT_PLOT}")
    
    # Log top vulnerable cases
    if not sig_pos.empty:
        print("\nTop 5 Most Vulnerable Cases Discovered by Causal ML:")
        top = sig_pos.sort_values("CATE", ascending=False).head(5)
        for _, row in top.iterrows():
            print(f"Case {row['case_number']} | Dist {row['council_district']} | {row['gross_site_area_acres']:.1f} Acres | CATE: +{row['CATE']:.3f} Denial Risk")

if __name__ == "__main__":
    main()
