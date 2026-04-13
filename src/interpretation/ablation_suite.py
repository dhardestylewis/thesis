import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier

# src/interpretation/ablation_suite.py
sys.path.append(str(Path(r"c:\Users\dhl\data\thesis\thesis") / "src"))
from data_io.schema import ROOT_DIR, REGISTRY_DIR, save_registry

def run_ablation_suite():
    print("[+] Running Cluster Ablation Suite...")
    
    # 1. Load Data
    labels = pd.read_parquet(REGISTRY_DIR / "label_registry.parquet")
    splits = pd.read_parquet(REGISTRY_DIR / "split_registry.parquet")
    X_full = pd.read_parquet(ROOT_DIR / "data" / "interim" / "stage_c_features_raw.parquet")
    
    lbl = labels[labels['label_version'] == 'label_v1_reconstructed_threshold_crossing'].drop_duplicates('case_id')
    spl = splits[splits['split_id'] == 'TEMP_OOD_2023_MAIN'].drop_duplicates('case_id')
    dataset = spl.merge(lbl, on='case_id').merge(X_full, on='case_id')
    
    train = dataset[dataset['role'] == 'train']
    test = dataset[dataset['role'] == 'test']
    
    meta_cols = ['case_id', 'as_of_date', 'feature_view', 'split_id', 'role', 'fold', 
                 'label_version', 'reconstructed_petition_share', 'threshold_crossed', 'year', 'filing_date']
    
    X_train_full = train.drop(columns=[c for c in meta_cols if c in train.columns], errors='ignore')
    X_test_full = test.drop(columns=[c for c in meta_cols if c in test.columns], errors='ignore')
    y_train = train['threshold_crossed']
    y_test = test['threshold_crossed']
    
    # 2. Base Performance
    print("    [*] Training Base Model...")
    cb_base = CatBoostClassifier(iterations=100, depth=4, verbose=0, random_seed=42)
    cb_base.fit(X_train_full, y_train)
    base_probs = cb_base.predict_proba(X_test_full)[:, 1]
    base_score = average_precision_score(y_test, base_probs)
    print(f"    [*] Base PR-AUC: {base_score:.4f}")
    
    # 3. Load Clusters
    with open(ROOT_DIR / "configs" / "features" / "semantic_clusters.yaml", 'r') as f:
        clusters_config = yaml.safe_load(f)['clusters']
        
    ablation_results = []
    
    for cluster_name, features in clusters_config.items():
        # Check which features exist
        present_features = [f for f in features if f in X_train_full.columns]
        if not present_features:
            continue
            
        print(f"    [*] Ablating Cluster: {cluster_name} ({len(present_features)} features)")
        
        # Drop cluster features
        X_train_ablated = X_train_full.drop(columns=present_features)
        X_test_ablated = X_test_full.drop(columns=present_features)
        
        # Re-fit
        cb_ablated = CatBoostClassifier(iterations=100, depth=4, verbose=0, random_seed=42)
        cb_ablated.fit(X_train_ablated, y_train)
        
        ablated_probs = cb_ablated.predict_proba(X_test_ablated)[:, 1]
        ablated_score = average_precision_score(y_test, ablated_probs)
        
        delta = base_score - ablated_score
        
        ablation_results.append({
            'cluster': cluster_name,
            'base_score': base_score,
            'ablated_score': ablated_score,
            'delta_prauc': delta,
            'n_features': len(present_features)
        })
        
    if not ablation_results:
        print("    [!] Warning: No clusters matched available features. Skipping visualization.")
        return

    df_ablation = pd.DataFrame(ablation_results)
    save_registry(df_ablation, "ablation_results")
    
    print("\n>>> CLUSTER ABLATION SUMMARY <<<")
    print(df_ablation.sort_values('delta_prauc', ascending=False))

if __name__ == "__main__":
    run_ablation_suite()
