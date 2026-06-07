import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier

# Paths
ROOT = Path(r"c:\Users\dhl\data\thesis\thesis")
PIPELINE_DATA = ROOT / "thesis_pipeline" / "data" / "final"

# Cluster Mapping (Subset)
CLUSTERS = {
    'parcel_scale': ['acreage', 'sqft'],
    'property_valuation': ['appraised_value', 'taxable_value'],
    'neighborhood_income': ['median_household_income', 'poverty_rate']
}

def run_ablation_tests():
    print("[+] Running Interpretation Ablation Suite...")
    
    # Load registries
    labels = pd.read_parquet(PIPELINE_DATA / "label_registry.parquet")
    features = pd.read_parquet(PIPELINE_DATA / "feature_registry.parquet")
    splits = pd.read_parquet(PIPELINE_DATA / "split_registry.parquet")
    
    # Task: label_v1, split_TEMP_OOD_2023_MAIN
    current_label = labels[labels['label_version'] == 'v1_reconstructed_threshold_crossing']
    current_features = features[features['feature_view'] == 'filing_date_public_admin_features']
    current_split = splits[splits['split_id'] == 'TEMP_OOD_2023_MAIN']
    
    data = current_split.merge(current_label, on=['case_number', 'as_of_date'])
    data = data.merge(current_features, on=['case_number', 'as_of_date', 'year'])
    
    train_data = data[data['usage'] == 'train']
    test_data = data[data['usage'] == 'test']
    
    X_train_full = train_data.drop(columns=['case_number', 'year', 'as_of_date', 'split_id', 'usage', 'fold', 'label_version', 'label_value', 'feature_view'], errors='ignore').select_dtypes(include=[np.number])
    y_train = train_data['label_value']
    
    X_test_full = test_data.drop(columns=['case_number', 'year', 'as_of_date', 'split_id', 'usage', 'fold', 'label_version', 'label_value', 'feature_view'], errors='ignore').select_dtypes(include=[np.number])
    y_test = test_data['label_value']
    
    # Baseline Performance
    print("    [*] Training Baseline Model...")
    cb = CatBoostClassifier(iterations=100, depth=4, random_seed=42, verbose=0)
    cb.fit(X_train_full, y_train)
    baseline_prauc = average_precision_score(y_test, cb.predict_proba(X_test_full)[:, 1])
    
    results = []
    
    # Leave-One-Cluster-Out (LOCO) Ablations
    for cluster, stems in CLUSTERS.items():
        print(f"    [~] Ablating cluster: {cluster}")
        # Identify columns in this cluster
        cluster_cols = [c for c in X_train_full.columns if any(s in c.lower() for s in stems)]
        if not cluster_cols: continue
        
        X_tr_ablated = X_train_full.drop(columns=cluster_cols)
        X_te_ablated = X_test_full.drop(columns=cluster_cols)
        
        cb_abl = CatBoostClassifier(iterations=100, depth=4, random_seed=42, verbose=0)
        cb_abl.fit(X_tr_ablated, y_train)
        
        ablated_prauc = average_precision_score(y_test, cb_abl.predict_proba(X_te_ablated)[:, 1])
        delta = baseline_prauc - ablated_prauc
        
        results.append({
            'cluster': cluster,
            'baseline_prauc': baseline_prauc,
            'ablated_prauc': ablated_prauc,
            'delta': delta,
            'significance': 'Robust' if delta > 0.01 else 'Spurious/Weak'
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(PIPELINE_DATA / "ablation_test_results.csv", index=False)
    
    print("\n>>> CLUSTER ABLATION SUMMARY <<<")
    print(df_res[['cluster', 'delta', 'significance']])

if __name__ == "__main__":
    run_ablation_tests()
