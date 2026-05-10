import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
import shap
import warnings
import sys
warnings.filterwarnings('ignore')

sys.path.append(r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference")
from run_causal_ml_sweep import load_fully_hydrated_data

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"

def main():
    print("1. Loading Fully Hydrated Data Matrix...")
    df, features, categorical_features = load_fully_hydrated_data()
    df = df[df["label_exact_geometric_petition_pct"] > 0].copy()
    
    thresholds = range(5, 96, 5)
    
    shap_results = []
    
    print("\n2. Executing Continuous SHAP Ablation Sweep...")
    for thresh in thresholds:
        T = (df["label_exact_geometric_petition_pct"] >= thresh).astype(int)
        y = df["t_denial"]
        X = df[features]
        
        idx_1 = (T == 1)
        X_1 = X[idx_1]
        y_1 = y[idx_1]
        
        if len(y_1) < 5 or y_1.nunique() <= 1:
            print(f"Skipping {thresh}% due to insufficient class diversity.")
            continue
            
        print(f"--> Training Model and Extracting SHAP for {thresh}%")
        model_1 = CatBoostClassifier(iterations=30, learning_rate=0.05, depth=4, verbose=0, random_seed=42, auto_class_weights='Balanced')
        pool_1 = Pool(X_1, y_1, cat_features=categorical_features)
        model_1.fit(pool_1)
        
        explainer = shap.TreeExplainer(model_1)
        shap_values = explainer.shap_values(pool_1)
        
        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Store results
        for i, feat in enumerate(X_1.columns):
            shap_results.append({
                "Threshold": thresh,
                "Feature": feat,
                "Mean_Abs_SHAP": mean_abs_shap[i]
            })
            
    # Export CSV
    results_df = pd.DataFrame(shap_results)
    out_csv = rf"{OUT_DIR}\continuous_shap_evolution.csv"
    results_df.to_csv(out_csv, index=False)
    
    # Identify the most important features overall to plot
    top_features = results_df.groupby("Feature")["Mean_Abs_SHAP"].mean().nlargest(6).index.tolist()
    
    # We want to specifically ensure Acreage and Height and Spatial Lag are included to test our hypothesis
    hyp_features = ["gross_site_area_acres", "proposed_max_height_ft", "knn_petition_rate_1km", "dist_petition_rate_lag1", "t_council_appearances"]
    plot_features = list(set(top_features + hyp_features))
    
    # Plotting
    print("\n3. Rendering Evolution Plot...")
    plt.figure(figsize=(12, 7), dpi=300)
    
    colors = sns.color_palette("tab10", len(plot_features))
    
    for i, feat in enumerate(plot_features):
        feat_data = results_df[results_df["Feature"] == feat]
        # Clean up labels for the legend
        label = feat.replace('_', ' ').title().replace(' Ft', '').replace('1Km', '(1km Radius)')
        if "Knn" in label: label = "Spatial Contagion (KNN Protest Density)"
        if "Dist Petition" in label: label = "Spatial Contagion (Historical Protest Lag)"
        if "Gross Site Area" in label: label = "Gross Site Area (Acreage)"
        if "Proposed Max Height" in label: label = "Proposed Max Height"
        
        linewidth = 3.5 if "Contagion" in label or "Acreage" in label or "Height" in label else 1.5
        linestyle = "-" if "Contagion" in label else "--" if "Acreage" in label or "Height" in label else ":"
        
        plt.plot(feat_data["Threshold"], feat_data["Mean_Abs_SHAP"], color=colors[i], linewidth=linewidth, linestyle=linestyle, label=label)

    plt.axvline(20, color='red', linestyle='-', linewidth=2, alpha=0.5, label='20% Legal Threshold')
    
    plt.title("Continuous SHAP Ablation: The Evolution of Denial Drivers\n(Proof of the Crossover from 'Acreage Boomerang' to 'Spatial Doom')", fontsize=14, weight='bold')
    plt.xlabel("Assumed Protest Threshold (%)", fontsize=12)
    plt.ylabel("Feature Importance (Mean Absolute SHAP Value)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=":", alpha=0.6)
    
    out_plot = rf"{OUT_DIR}\continuous_shap_evolution.png"
    plt.savefig(out_plot, bbox_inches="tight")
    print(f"Artifacts saved to {out_csv} and {out_plot}")

if __name__ == "__main__":
    main()
