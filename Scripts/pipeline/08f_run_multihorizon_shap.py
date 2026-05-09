import pandas as pd
import numpy as np
import shap
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from pathlib import Path
import os
import warnings

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "Data/Panel/biweekly_panel.csv"

def run_multihorizon_shap():
    print("--- 1. Loading Panel for Multi-Horizon SHAP ---")
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    
    FEATS = [
        "cumulative_min_signer_dist", "cumulative_unofficial_protest_intensity", 
        "cumulative_protester_embed_dim1", "cumulative_protester_embed_dim2",
        "cumulative_temporal_protesting_pct_sf", "cumulative_delta_protesting_friction",
        "market_value", "total_population", "median_household_income", 
        "renter_share", "race_white", "median_age",
        "mortgage_rate_30yr", "mortgage_rate_30yr_momentum"
    ]
    
    # Analyze the 6-month and 1-Year horizons
    horizons = {"6_Months": 13, "1_Year": 26}
    
    for h_name, h_shift in horizons.items():
        print(f"\nTraining Model for Horizon: {h_name}")
        df["target"] = df.groupby("case_number")["petition_event"].transform(
            lambda x: x.rolling(window=h_shift, min_periods=1).max().shift(-h_shift)
        ).fillna(0)
        
        train = df.dropna(subset=FEATS)
        X = train[FEATS]
        y = train["target"]
        
        clf = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=False, task_type="GPU")
        clf.fit(X, y)
        
        print(f"Generating SHAP values for {h_name}...")
        explainer = shap.TreeExplainer(clf)
        
        # Take a random sample to avoid OOM in visualization
        X_sample = X.sample(n=min(5000, len(X)), random_state=42)
        shap_values = explainer(X_sample)
        
        # Plot Beeswarm
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, max_display=12, show=False)
        plt.title(f"SHAP Attribution - {h_name} Horizon")
        plt.tight_layout()
        out_swarm = ROOT / "artifacts" / f"causal_shap_beeswarm_multihorizon_{h_name}.png"
        os.makedirs(out_swarm.parent, exist_ok=True)
        plt.savefig(out_swarm, dpi=300)
        plt.close()
        
        # Plot Interaction Heatmap for the top 2 features
        plt.figure(figsize=(10, 8))
        shap.plots.heatmap(shap_values, max_display=12, show=False)
        plt.title(f"SHAP Interaction Heatmap - {h_name} Horizon")
        plt.tight_layout()
        out_heat = ROOT / "artifacts" / f"causal_shap_heatmap_multihorizon_{h_name}.png"
        plt.savefig(out_heat, dpi=300)
        plt.close()
        
    print("[+] Multi-Horizon SHAP Execution Complete.")

if __name__ == "__main__":
    run_multihorizon_shap()
