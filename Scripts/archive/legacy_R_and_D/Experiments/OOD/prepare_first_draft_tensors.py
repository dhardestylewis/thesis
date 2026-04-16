"""
prepare_first_draft_tensors.py
======================
Builds PyTorch tensors for ICP invariance testing on the structured NLP Agenda Features.
Methodology:
  - Environments: Temporal Environments (Zoning Year 2007-2024)
  - Target: Dissenting Council Vote (vote_no >= 1)
  - Features: NLP Agenda Covariates
"""
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
import os, json, time
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = r"c:\Users\dhl\data\thesis\thesis"
IN_PATH = os.path.join(PROJECT_DIR, "Data", "Zoning_Cases", "Processed_Data", "CSV", "submission_grade_goldmine_tensor.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "Analysis", "Data", "Tensors")
os.makedirs(OUT_DIR, exist_ok=True)

N_PCA_COMPONENTS = 10

def main():
    t0 = time.time()
    print("[*] Loading First Draft CSV Tensor Matrix...")
    df = pd.read_csv(IN_PATH)
    
    # 1. Target Vector (y)
    # We predict if the Zoning Case receives ANY dissenting votes from the Council
    df = df.dropna(subset=['vote_no']).copy()
    df['target_y'] = (df['vote_no'] >= 1.0).astype(int)
    
    # 2. Environment Vector (e)
    # Environments are temporal (each Year is an independent City Council regime)
    df['Meeting_Date'] = pd.to_datetime(df['Meeting_Date'], errors='coerce')
    df = df.dropna(subset=['Meeting_Date'])
    df['year'] = df['Meeting_Date'].dt.year.astype(int)
    
    # Valid years are our environments
    valid_envs = df['year'].value_counts()
    valid_envs = valid_envs[valid_envs >= 10].index.tolist()
    df = df[df['year'].isin(valid_envs)].copy()
    
    df['env_id'] = "Year_" + df['year'].astype(str)
    # Re-index environments 0 to N
    unique_years = sorted(df['year'].unique())
    year_map = {y: i for i, y in enumerate(unique_years)}
    df['env_label'] = df['year'].map(year_map)
    
    # 3. Features (X)
    NUMERIC_FEATURES = ['valid_petition', 'commission_disagree', 'is_npa', 'acreage',
                        'neighborhood_protest_contagion', 'neighborhood_median_wealth', 'neighborhood_density',
                        'FEDFUNDS', 'MORTGAGE30US', 'target_zoning_density', 'friction_white']
    CATEGORICAL_FEATURES = ['agent', 'orig_zoning', 'target_zoning', 'watershed']
    
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].fillna('Missing').astype(str)
        
    print(f"[+] Total Cases: {len(df)}")
    print(f"[+] Contested Vote Rate: {df['target_y'].mean():.2%}")
    print(f"[+] Total Active Environments: {len(unique_years)}")
    
    # Fold 1: Train on full dataset to find GLOBAL INVARIANCES
    fold_name = "fold_first_draft"
    fold_dir = os.path.join(OUT_DIR, fold_name)
    os.makedirs(fold_dir, exist_ok=True)
    
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    
    X_num = scaler.fit_transform(df[NUMERIC_FEATURES])
    X_cat_raw = encoder.fit_transform(df[CATEGORICAL_FEATURES])
    
    n_comp = min(N_PCA_COMPONENTS, X_cat_raw.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_cat = svd.fit_transform(X_cat_raw)
    
    X_train = np.hstack([X_num, X_cat]).astype(np.float32)
    y_train = df['target_y'].values.astype(np.float32)
    envs_train = df['env_label'].values.astype(np.int64)
    env_ids_train = df['env_id'].values
    
    torch.save(torch.from_numpy(X_train), os.path.join(fold_dir, "X_train.pt"))
    torch.save(torch.from_numpy(y_train), os.path.join(fold_dir, "y_train.pt"))
    torch.save(torch.from_numpy(envs_train), os.path.join(fold_dir, "envs_train.pt"))
    np.save(os.path.join(fold_dir, "env_ids_train.npy"), np.array(env_ids_train, dtype=str))
    
    cat_pca_names = [f'cat_pca_{i}' for i in range(n_comp)]
    full_feature_names = list(NUMERIC_FEATURES) + cat_pca_names
    unique_envs_str = sorted([e for e in df['env_id'].unique()])
    
    fold_meta = {
        'fold_name': fold_name,
        'train_end_year': int(max(unique_years)),
        'test_year': int(max(unique_years)),
        'train_size': len(df),
        'test_size': len(df),
        'n_features': X_train.shape[1],
        'numeric_features': list(NUMERIC_FEATURES),
        'full_feature_names': full_feature_names,
        'unique_envs': unique_envs_str,
        'n_numeric': len(NUMERIC_FEATURES),
        'n_pca_components': n_comp,
        'pca_explained_variance': float(svd.explained_variance_ratio_.sum()),
        'protest_rate_train': float(df['target_y'].mean()),
    }
    
    with open(os.path.join(fold_dir, 'metadata.json'), 'w') as f:
        json.dump(fold_meta, f, indent=2)
        
    # Global metadata array just for this single draft fold
    with open(os.path.join(OUT_DIR, 'cv_metadata.json'), 'w') as f:
        json.dump([fold_meta], f, indent=2)

    print(f"\n[+] Tensors physically generated. Ready for execution in icp_nonlinear.py!")

if __name__ == "__main__":
    main()
