import pandas as pd
import numpy as np
import os
import sys
from sklearn.neighbors import NearestNeighbors

# Import central registry
sys.path.insert(0, os.path.join(r"C:\Users\dhl\data\thesis\thesis", "Analysis", "Scripts"))
from artifact_registry import TraceabilityRegistry as AR
from Utilities_and_Logs.lib_metrics import update_metric

PREDS_FILE = str(AR.STAGE_C_OOF_H0)
DATA_FILE = os.path.join(r"C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")

def compute_gaps(df, pred_col):
    fprs = {}
    fnrs = {}
    
    # We still measure gap across the *discrete* districts to see if 
    # continuous spatial thresholding fixes the administrative disparity.
    for d in df['district'].unique():
        sub = df[df['district'] == d]
        
        positives = sub[sub['y_true'] == 1]
        if len(positives) > 0:
            fnrs[d] = 1.0 - (positives[pred_col].sum() / len(positives))
            
        negatives = sub[sub['y_true'] == 0]
        if len(negatives) > 0:
            fprs[d] = negatives[pred_col].sum() / len(negatives)
            
    fnr_gap = (max(fnrs.values()) - min(fnrs.values())) * 100 if fnrs else 0.0
    fpr_gap = (max(fprs.values()) - min(fprs.values())) * 100 if fprs else 0.0
    return fnr_gap, fpr_gap

def run_experiment():
    print("==================================================")
    print(" FAIRNESS THRESHOLDING EXPERIMENT (PHASE 3) ")
    print(" Continuous Spatial Density Fairness (KNN) ")
    print("==================================================")
    
    if not os.path.exists(PREDS_FILE):
        print(f"[-] Predictions file not found: {PREDS_FILE}")
        return
        
    df = pd.read_csv(PREDS_FILE)
    
    # Baseline
    global_thresh = df['y_true'].mean()
    df['pred_baseline'] = (df['y_prob'] > global_thresh).astype(int)
    fnr0, fpr0 = compute_gaps(df, 'pred_baseline')
    print(f"[0] Baseline Global Threshold (mu_y = {global_thresh:.4f})")
    print(f"    FNR Gap: {fnr0:5.2f}%  |  FPR Gap: {fpr0:5.2f}%\n")
    update_metric("metricSensitivityFprBaseline", f"{fpr0:.2f}\\%")
    update_metric("metricSensitivityFnrBaseline", f"{fnr0:.2f}\\%")
    
    if not os.path.exists(DATA_FILE):
        print(f"[-] Master dataset not found: {DATA_FILE}")
        return
        
    # Load raw coordinates from H0 file and apply exactly the same filter Stage C used
    master = pd.read_csv(DATA_FILE, usecols=['year', 'latitude', 'longitude'], low_memory=False)
    master['year'] = pd.to_numeric(master['year'], errors='coerce')
    master = master.dropna(subset=['year']).sort_values('year').copy()
    
    if len(master) != len(df):
        print(f"[-] Row mismatch! Master has {len(master)} rows, Preds has {len(df)} rows.")
        return
        
    print(f"[+] Deterministic Row-Match successful ({len(df)} properties).")
    
    # Inject spatial coordinates natively onto the prediction array
    df['latitude'] = master['latitude'].values
    df['longitude'] = master['longitude'].values
    
    # Drop rows that have NaNs in the spatial coordinate (since historical cases may lack geocodes)
    initial_len = len(df)
    df = df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)
    print(f"    Usable Spatial Properties: {len(df)}/{initial_len}\n")
    
    if len(df) < 100:
        print("[-] Insufficient spatial matches to compute KNN.")
        return
        
    # KNN Implementation on the valid subset
    coords = df[['latitude', 'longitude']].values
    
    # Test varying spatial radii (Neighbors)
    for K in [50, 150, 500]:
        nn = NearestNeighbors(n_neighbors=K, metric='euclidean', n_jobs=-1)
        nn.fit(coords)
        _, indices = nn.kneighbors(coords)
        
        # Calculate localized probability map
        local_thresholds = np.zeros(len(df))
        for i in range(len(df)):
            # The local threshold is the natural NIMBY base rate within the continuous K-nearest spatial zone
            local_thresholds[i] = df['y_true'].iloc[indices[i]].mean()
            
            # Fallback for empty/zero variance zones
            if local_thresholds[i] == 0: 
                local_thresholds[i] = global_thresh
                
        # Assign predictions explicitly avoiding discrete district boundaries
        df[f'pred_knn_{K}'] = (df['y_prob'] > local_thresholds).astype(int)
        fnr_k, fpr_k = compute_gaps(df, f'pred_knn_{K}')
        
        print(f"[7] Continuous Spatial Density Fairness (KNN, K={K})")
        print(f"    FNR Gap: {fnr_k:5.2f}%  |  FPR Gap: {fpr_k:5.2f}%")
        
        # We only export K=500 for the main table presentation
        if K == 500:
            update_metric("metricSensitivityFprKNN", f"{fpr_k:.2f}\\%")
            update_metric("metricSensitivityFnrKNN", f"{fnr_k:.2f}\\%")
        
    print("\nPhase 3 Spatial Experiment Complete.")

if __name__ == '__main__':
    run_experiment()
