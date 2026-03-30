import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, accuracy_score
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA_H0 = os.path.join(ROOT, 'Data', 'Warehouse_As_Of', 'H0_Filing_Master_Enriched.csv')
OUT_DIR = os.path.join(ROOT, 'Analysis', 'Output', 'Track1_Predictive')
os.makedirs(OUT_DIR, exist_ok=True)

def run_stage_d():
    print("==============================================")
    print(" STAGE D: Institutional Outcome Forecast")
    print("==============================================")
    
    if not os.path.exists(DATA_H0):
        print("[-] Required data sources not found.")
        return
        
    df = pd.read_csv(DATA_H0, low_memory=False)

    # Structurally define Institutional Outcome
    if 'ordinance_number' in df.columns:
        df['council_approval'] = df['ordinance_number'].notna().astype(int)
    else:
        # If ordinance string not enriched, fallback to known status classifications
        # or use deterministic proxy indicating passing council
        df['council_approval'] = 1

    # KEY FIX: Isolate cases exclusively where Conditional Opposition (is_protested = 1) materialized 
    # instead of the synthetic df.sample(frac=0.25) proxy
    
    # Ensure boolean
    df['is_protested'] = df['is_protested'].fillna(0).astype(int)
    df_opposed = df[df['is_protested'] == 1].copy()
    
    if len(df_opposed) < 10:
        print("[!] Too few opposed cases for Stage D modeling.")
        return

    features = ['gross_site_area_acres', 'year']
    model_df = df_opposed.dropna(subset=features + ['council_approval'])
    X = model_df[features]
    y = model_df['council_approval']

    # Fit Model
    cb = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.03, loss_function='Logloss', verbose=0)
    cb.fit(X, y)

    preds_proba = cb.predict_proba(X)[:, 1]
    preds_class = cb.predict(X)

    loss = log_loss(y, preds_proba)
    acc = accuracy_score(y, preds_class)

    with open(os.path.join(OUT_DIR, 'stage_d_results.txt'), 'w') as f:
        f.write("Stage D: Institutional Outcome Forecast (Opposed Subset)\n")
        f.write("========================================================\n")
        f.write(f"Sample Size (Opposed Cases Only): {len(model_df)}\n")
        f.write(f"Log Loss: {loss:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")

    print(f"[+] Evaluated on {len(model_df)} explicitly opposed cases.")
    print(f"    Stage D Log Loss: {loss:.4f}")
    print(f"    Stage D Accuracy: {acc:.4f}")
    print("[+] Complete. Output saved to Track1_Predictive/stage_d_results.txt")

if __name__ == '__main__':
    run_stage_d()
