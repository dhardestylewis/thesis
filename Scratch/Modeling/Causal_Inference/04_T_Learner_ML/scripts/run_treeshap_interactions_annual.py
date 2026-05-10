import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
import shap
import warnings
import sys
warnings.filterwarnings('ignore')

# Add the path to load the hydration function
sys.path.append(r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\04_T_Learner_ML")
from run_causal_ml_sweep import load_fully_hydrated_data

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts"

def main():
    print("1. Loading Fully Hydrated Annualized Data Matrix...")
    df, features, categorical_features = load_fully_hydrated_data()
    
    # We will target label_valid_protest (or use the threshold trick from the sweep)
    # The thesis evaluates protest as >= 20% intensity
    thresh = 20
    df = df.dropna(subset=["exact_geometric_petition_pct"])
    T = (df["exact_geometric_petition_pct"] >= thresh).astype(int)
    y = T
    X = df[features]
    
    print(f"Dataset active. N = {len(df)} cases.")
    print(f"Total Model Features: {len(features)}")
    
    print(f"2. Training CatBoostClassifier on Annualized Data (Target: Protest >= {thresh}%)...")
    # Using task_type="GPU" to accelerate SHAP interaction values and training
    model = CatBoostClassifier(
        iterations=300, 
        learning_rate=0.05, 
        depth=6, 
        verbose=100, 
        random_seed=42, 
        auto_class_weights='Balanced',
        task_type="GPU"
    )
    
    pool = Pool(X, y, cat_features=categorical_features)
    model.fit(pool)
    
    print("3. Executing SHAP TreeExplainer (Interactions)...")
    explainer = shap.TreeExplainer(model)
    
    # Generate SHAP Interaction Values
    # Output shape will be (N, features, features)
    shap_interaction_values = explainer.shap_interaction_values(pool)
    
    print("4. Rendering SHAP Interaction Summary Plot...")
    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_interaction_values, X, max_display=12, show=False)
    
    plt.title(f"SHAP Feature Interactions: Annualized Panel\n(CatBoost predicting Protest >= {thresh}%)", fontsize=14, weight="bold", y=1.05)
    plt.tight_layout()
    out_path_summary = rf"{OUT_DIR}\causal_shap_interaction_summary_annual.png"
    plt.savefig(out_path_summary, bbox_inches="tight")
    print(f"Interaction summary artifact saved to {out_path_summary}")
    plt.close()

if __name__ == "__main__":
    main()
