import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
import os

warnings.filterwarnings('ignore')

BASE_DIR = r"c:\Users\dhl\data\Thesis\thesis"
SUPP_CSV = os.path.join(BASE_DIR, "Scratch", "Modeling", "Causal_Inference", "04_T_Learner_ML", "model_ready_zoning_supplemented.csv")

def main():
    print(f"Loading Supplemented Annual Panel: {SUPP_CSV}")
    df = pd.read_csv(SUPP_CSV, low_memory=False)
    
    GEOM_PATH = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\exact_geometric_petition_intensity.csv"
    print(f"Loading Geometric Petition Intensity: {GEOM_PATH}")
    geom_df = pd.read_csv(GEOM_PATH)
    
    df["case_number"] = df["case_number"].astype(str).str.strip()
    geom_df["case_number"] = geom_df["case_number"].astype(str).str.strip()
    
    df = df.merge(geom_df[["case_number", "exact_geometric_petition_pct"]], on="case_number", how="inner")
    
    # We only care about cases with a valid target and case_number
    df = df.dropna(subset=["exact_geometric_petition_pct", "case_number"])
    
    # Define Target
    thresh = 20
    df["target"] = (df["exact_geometric_petition_pct"] >= thresh).astype(int)
    
    # Features from the core protest panel
    core_features = [
        "gross_site_area_acres", "council_district", "Initial_Zoning", "Requested_Zoning", 
        "shape_area", "shape_length"
    ]
    
    # Features from the biweekly supplement
    biweekly_features = [
        "total_population", "median_household_income", "renter_share", "rent_burden", 
        "affordability_proxy", "race_white", "race_black", "race_hispanic", "median_age", 
        "mortgage_rate_30yr", "local_unemployment_rate", "fed_funds_rate", 
        "treasury_10yr_yield", "mortgage_rate_30yr_momentum", "local_unemployment_rate_momentum", 
        "fed_funds_rate_momentum", "treasury_10yr_yield_momentum", "knn_petition_rate_1km", 
        "dist_petition_rate_lag1", "active_cases_1km", "active_cases_2km"
    ]
    
    features = core_features + biweekly_features
    categorical_features = ["council_district", "Initial_Zoning", "Requested_Zoning"]
    
    # Clean categoricals
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("Unknown")
            
    # Subselect and drop cases where core features are completely missing
    model_df = df[["case_number", "target"] + features].copy()
    
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(model_df, groups=model_df["case_number"]))
    
    train = model_df.iloc[train_idx]
    test = model_df.iloc[test_idx]
    
    X_train, y_train = train[features], train["target"]
    X_test, y_test = test[features], test["target"]
    
    print(f"\nTraining CatBoost on {len(X_train)} cases with {len(features)} features...")
    
    model = CatBoostClassifier(
        iterations=300, 
        learning_rate=0.05, 
        depth=6, 
        verbose=0, 
        random_seed=42, 
        auto_class_weights='Balanced',
        task_type="GPU"
    )
    
    model.fit(Pool(X_train, y_train, cat_features=categorical_features))
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    pos_rate = y_test.mean() * 100
    print(f"\n=== SUPPLEMENTED MODEL PERFORMANCE ===")
    print(f"Threshold >= {thresh:2d}% | ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f} | Test Positives: {y_test.sum()} ({pos_rate:.1f}%)")
    
    # Feature Importance
    print("\n=== TOP 10 FEATURES ===")
    importances = model.get_feature_importance()
    feat_imps = pd.DataFrame({"Feature": features, "Importance": importances})
    print(feat_imps.sort_values(by="Importance", ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    main()
