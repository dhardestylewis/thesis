import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
import warnings
import sys
warnings.filterwarnings('ignore')

sys.path.append(r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference")
from run_causal_ml_sweep import load_fully_hydrated_data

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"

def run_mc_sweep(df, features, categorical_features, target="t_denial"):
    thresholds = range(5, 96, 5)
    seeds = [42, 100, 2024, 7, 99]
    results = []
    
    cat_feats = [f for f in categorical_features if f in features]
    
    for thresh in thresholds:
        T = (df["label_exact_geometric_petition_pct"] >= thresh).astype(int)
        y = df[target]
        X = df[features]
        
        idx_1 = (T == 1)
        idx_0 = (T == 0)
        
        if len(y[idx_1]) < 5 or y[idx_1].nunique() <= 1:
            continue
            
        if len(y[idx_0]) < 5 or y[idx_0].nunique() <= 1:
            continue
            
        seed_cates = []
        for s in seeds:
            model_1 = CatBoostClassifier(iterations=30, learning_rate=0.05, depth=4, verbose=0, random_seed=s, auto_class_weights='Balanced')
            model_0 = CatBoostClassifier(iterations=30, learning_rate=0.05, depth=4, verbose=0, random_seed=s, auto_class_weights='Balanced')
            
            pool_1 = Pool(X[idx_1], y[idx_1], cat_features=cat_feats)
            pool_0 = Pool(X[idx_0], y[idx_0], cat_features=cat_feats)
            
            model_1.fit(pool_1)
            model_0.fit(pool_0)
            
            prob_1 = model_1.predict_proba(X)[:, 1]
            prob_0 = model_0.predict_proba(X)[:, 1]
            cate = prob_1 - prob_0
            seed_cates.append(cate)
            
        seed_cates = np.array(seed_cates)
        mean_cates = seed_cates.mean(axis=1) # shape: (5,)
        peak_cates = seed_cates.max(axis=1)  # shape: (5,)
        
        results.append({
            "Threshold": thresh,
            "Mean_CATE": mean_cates.mean(),
            "Peak_CATE": peak_cates.mean()
        })
    return pd.DataFrame(results)

def main():
    print("1. Loading Fully Hydrated Data Matrix...")
    df, features, categorical_features = load_fully_hydrated_data()
    df = df[df["label_exact_geometric_petition_pct"] > 0].copy()
    
    print("2. Running Baseline Sweep...")
    base_res = run_mc_sweep(df, features, categorical_features)
    
    print("3. Running Ablation: Excluding Acreage/Scale...")
    scale_feats = ["gross_site_area_acres", "proposed_max_height_ft", "existing_max_height_ft", "proposed_max_far", "existing_max_far"]
    feats_no_scale = [f for f in features if f not in scale_feats]
    noscale_res = run_mc_sweep(df, feats_no_scale, categorical_features)
    
    print("4. Running Ablation: Excluding Spatial Contagion...")
    contagion_feats = ["knn_petition_rate_1km", "dist_petition_rate_lag1", "archetype_pct_Spatial_Gravity", "archetype_pct_Architectural", "archetype_pct_Economic", "archetype_pct_Bureaucratic"]
    feats_no_cont = [f for f in features if f not in contagion_feats]
    nocont_res = run_mc_sweep(df, feats_no_cont, categorical_features)
    
    print("5. Plotting Degradation...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Plot Mean CATE (Boomerang)
    ax = axes[0]
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(20, color='red', linestyle='-', alpha=0.3)
    ax.plot(base_res["Threshold"], base_res["Mean_CATE"], color='black', linewidth=3, label="Baseline (Full Matrix)")
    ax.plot(noscale_res["Threshold"], noscale_res["Mean_CATE"], color='#3498DB', linewidth=2, linestyle='--', label="Ablated (No Acreage/Scale)")
    ax.set_title("Mean CATE (The Boomerang Effect)", weight='bold')
    ax.set_xlabel("Protest Threshold (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot Peak CATE (Extreme Vulnerability)
    ax = axes[1]
    ax.axhline(0, color='gray', linestyle='--')
    ax.plot(base_res["Threshold"], base_res["Peak_CATE"], color='black', linewidth=3, label="Baseline (Full Matrix)")
    ax.plot(nocont_res["Threshold"], nocont_res["Peak_CATE"], color='#E74C3C', linewidth=2, linestyle='--', label="Ablated (No Spatial Contagion)")
    ax.set_title("Peak CATE (Extreme Vulnerability)", weight='bold')
    ax.set_xlabel("Protest Threshold (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle("Causal Feature Exclusion (Leave-Covariate-Out Ablation)\nTarget: Denial Risk", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(rf"{OUT_DIR}\causal_exclusion_degradation.png", bbox_inches='tight')
    print("Done!")

if __name__ == "__main__":
    main()
