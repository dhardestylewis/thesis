import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
import shap
import warnings
import sys
warnings.filterwarnings('ignore')

# Add the path to load the hydration function from the previous script
sys.path.append(r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference")
from run_causal_ml_sweep import load_fully_hydrated_data

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
OUT_PLOT = rf"{OUT_DIR}\causal_shap_summary.png"

def main():
    print("1. Loading Fully Hydrated Data Matrix...")
    df, features, categorical_features = load_fully_hydrated_data()
    df = df[df["label_exact_geometric_petition_pct"] > 0].copy()
    
    thresholds_to_explain = [20, 50, 80]
    
    for thresh in thresholds_to_explain:
        print(f"\n--- Analyzing Threshold: {thresh}% ---")
        T = (df["label_exact_geometric_petition_pct"] >= thresh).astype(int)
        y = df["t_denial"]
        X = df[features]
        
        idx_1 = (T == 1)
        X_1 = X[idx_1]
        y_1 = y[idx_1]
        
        if len(y_1) < 5 or y_1.nunique() <= 1:
            print(f"Skipping {thresh}% due to insufficient class diversity in treated group.")
            continue
            
        print(f"2. Training Model 1 (Treated Group) at {thresh}% Threshold...")
        model_1 = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=4, verbose=0, random_seed=42, auto_class_weights='Balanced')
        pool_1 = Pool(X_1, y_1, cat_features=categorical_features)
        model_1.fit(pool_1)
        
        print(f"3. Executing SHAP TreeExplainer for {thresh}%...")
        explainer = shap.TreeExplainer(model_1)
        shap_values = explainer.shap_values(pool_1)
        
        print(f"4. Rendering SHAP Summary Plot for {thresh}%...")
        plt.figure(figsize=(10, 8), dpi=300)
        
        shap.summary_plot(shap_values, X_1, max_display=15, show=False, plot_type="dot")
        
        plt.title(f"SHAP Feature Attribution: The {thresh}% Threshold (Treated Group)\nWhat drives Denial Risk?", fontsize=14, weight="bold", y=1.05)
        plt.tight_layout()
        out_path = rf"{OUT_DIR}\causal_shap_summary_{thresh}.png"
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Artifact saved to {out_path}")
        plt.close()

if __name__ == "__main__":
    main()
