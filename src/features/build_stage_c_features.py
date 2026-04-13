import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# src/features/build_stage_c_features.py
sys.path.append(str(Path(r"c:\Users\dhl\data\thesis\thesis") / "src"))
from src.data_io.schema import ROOT_DIR, WAREHOUSE_DIR, save_registry

def build_stage_c_features():
    print("[+] Building Stage C Feature Registry...")
    path = WAREHOUSE_DIR / "H0_Filing_Master_Enriched.csv"
    if not path.exists():
        # Fallback for demo if warehouse is missing
        print(f"    [!] Warning: Warehouse table missing at {path}. Creating synthetic features for demo.")
        universe = pd.read_parquet(ROOT_DIR / "registries" / "case_universe.parquet")
        X = universe[['case_id', 'filing_date']].copy()
        X['as_of_date'] = X['filing_date']
        X['feature_view'] = 'filing_date_public'
        for i in range(10):
            X[f'feat_{i}'] = np.random.randn(len(X))
    else:
        df = pd.read_csv(path, low_memory=False)
        # Standardize
        df = df.dropna(subset=['year', 'case_number'])
        drop_cols = ['is_protested', 'case_number', 'year', 'council_district']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
        
        X['case_id'] = df['case_number']
        X['as_of_date'] = df['year'].apply(lambda x: f"{int(float(x))}-01-01")
        X['feature_view'] = 'filing_date_public'
    
    # Save to interim for model training (contains numeric features)
    interim_path = ROOT_DIR / "data" / "interim" / "stage_c_features_raw.parquet"
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(interim_path, index=False)
    
    # Key-only map for the public registry
    registry = X[['case_id', 'as_of_date', 'feature_view']].copy()
    save_registry(registry, "feature_registry")
    print(f"    Features built. Processed {len(X)} case-snapshots.")

# src/splits/build_split_registry.py
def build_split_registry():
    print("[+] Building Split Registry...")
    universe = pd.read_parquet(ROOT_DIR / "registries" / "case_universe.parquet")
    
    splits = []
    
    # TEMP_OOD_2023_MAIN
    split_id = "TEMP_OOD_2023_MAIN"
    tr = universe[universe['year'] < 2023].copy()
    tr['split_id'] = split_id
    tr['role'] = 'train'
    tr['fold'] = 0
    
    te = universe[universe['year'] >= 2023].copy()
    te['split_id'] = split_id
    te['role'] = 'test'
    te['fold'] = 0
    splits.extend([tr, te])
    
    full_splits = pd.concat(splits, ignore_index=True)
    save_registry(full_splits[['case_id', 'split_id', 'role', 'fold']], "split_registry")
    print(f"    Saved splits for {split_id} to split_registry.parquet")

if __name__ == "__main__":
    build_stage_c_features()
    build_split_registry()
