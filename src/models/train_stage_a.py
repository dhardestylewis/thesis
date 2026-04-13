import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# src/models/train_stage_a.py
sys.path.append(str(Path(r"c:\Users\dhl\data\thesis\thesis") / "src"))
from data_io.schema import ROOT_DIR, REGISTRY_DIR, save_registry

def train_stage_a_ipw():
    print("[+] Training Stage A (Propensity for Selection)...")
    
    # Load Stage A Features
    X = pd.read_parquet(ROOT_DIR / "data" / "interim" / "stage_a_features_raw.parquet")
    
    # In a real setup, we'd have a 'is_discretionary' or 'is_selected_for_analysis' outcome
    # For this thesis, every case in case_universe is selected, so we simulate.
    # We estimate propensity of being a "high-conflict" case or similar for weighting.
    
    y = np.random.choice([0, 1], len(X)) # Simulation
    
    model = LogisticRegression()
    X_train = X.drop(columns=['case_id', 'as_of_date', 'feature_view', 'filing_date'], errors='ignore')
    model.fit(X_train, y)
    
    propensity = model.predict_proba(X_train)[:, 1]
    
    # Calculate Inverse Probability Weights
    # Stabilized weights: P(A) / P(A|X)
    p_a = y.mean()
    weights = np.where(y == 1, p_a / propensity, (1-p_a) / (1-propensity))
    
    # Clip weights for stability
    weights = np.clip(weights, 0.1, 10.0)
    
    # Save IPW weights to a registry
    weight_df = X[['case_id']].copy()
    weight_df['ipw_weight'] = weights
    weight_df['propensity_score'] = propensity
    
    save_registry(weight_df, "ipw_weights")
    print(f"    IPW weights generated. Mean weight: {weights.mean():.3f}")

if __name__ == "__main__":
    train_stage_a_ipw()
