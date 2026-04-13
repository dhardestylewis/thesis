import pandas as pd
import numpy as np
import os
from pathlib import Path

# Paths
ROOT = Path(r"c:\Users\dhl\data\thesis\thesis")
WAREHOUSE_DIR = ROOT / "Data" / "Warehouse_As_Of"
PIPELINE_DATA = ROOT / "thesis_pipeline" / "data" / "final"
PIPELINE_DATA.mkdir(parents=True, exist_ok=True)

def initialize_case_universe():
    print("[+] Initializing Case Universe...")
    # Using H0 Filing Baseline as the source for the universe of 7,074 discretionary cases
    path = WAREHOUSE_DIR / "H0_Filing_Master_Enriched.csv"
    if not path.exists():
        print(f"    [!] Error: {path} not found.")
        return

    df = pd.read_csv(path, low_memory=False)
    
    # Define the primary keys
    # case_id (could be case_number or standardized_tcad_id + year)
    # The user mentioned case_id as part of the composite key.
    # In Austin, case_number is unique for the petition process.
    
    # Locate Council District for Spatial Holdouts
    dist_col = 'ldb_council_district' if 'ldb_council_district' in df.columns else ('council_district_x' if 'council_district_x' in df.columns else 'council_district')
    if dist_col not in df.columns:
        df[dist_col] = 1 # Fallback
        
    universe = df[['case_number', 'standardized_tcad_id', 'year', dist_col]].copy()
    universe = universe.dropna(subset=['year', 'case_number'])
    universe = universe.rename(columns={dist_col: 'council_district'})
    universe['as_of_date'] = universe['year'].apply(lambda x: f"{int(float(x))}-01-01") # Simplified as-of
    
    # Save as parquet
    universe.to_parquet(PIPELINE_DATA / "case_universe.parquet", index=False)
    print(f"    Saved {len(universe)} cases to case_universe.parquet")

def initialize_label_registry():
    print("[+] Initializing Label Registry...")
    path = WAREHOUSE_DIR / "H0_Filing_Master_Enriched.csv"
    df = pd.read_csv(path, low_memory=False)
    
    # Extract existing reconstructed outcome
    labels = df[['case_number', 'year', 'is_protested']].copy()
    labels = labels.dropna(subset=['year', 'case_number'])
    labels['as_of_date'] = labels['year'].apply(lambda x: f"{int(float(x))}-01-01")
    labels['label_version'] = 'v1_reconstructed_threshold_crossing'
    labels['label_value'] = pd.to_numeric(labels['is_protested'], errors='coerce').fillna(0).astype(int)
    
    # Prepare labels registry
    label_registry = labels[['case_number', 'as_of_date', 'label_version', 'label_value']]
    
    # Save as parquet
    label_registry.to_parquet(PIPELINE_DATA / "label_registry.parquet", index=False)
    print(f"    Saved {len(label_registry)} labels to label_registry.parquet")

def initialize_split_registry():
    print("[+] Initializing Split Registry...")
    universe = pd.read_parquet(PIPELINE_DATA / "case_universe.parquet")
    
    splits = []
    
    # 1. Temporal ROLL_ORIGIN splits
    for anchor in [2021, 2022, 2023, 2024]:
        split_id = f"TEMP_ROLL_PRE{anchor}"
        mask = universe['year'] < anchor
        # Train on pre-anchor, Test on anchor
        tr = universe[mask].copy()
        tr['split_id'] = split_id
        tr['usage'] = 'train'
        tr['fold'] = 0
        
        te = universe[universe['year'] == anchor].copy()
        te['split_id'] = split_id
        te['usage'] = 'test'
        te['fold'] = 0
        
        splits.extend([tr, te])

    # 2. Primary OOD Split (e.g., test on 2023-2024)
    split_id = "TEMP_OOD_2023_MAIN"
    tr = universe[universe['year'] < 2023].copy()
    tr['split_id'] = split_id
    tr['usage'] = 'train'
    tr['fold'] = 0
    
    te = universe[universe['year'] >= 2023].copy()
    te['split_id'] = split_id
    te['usage'] = 'test'
    te['fold'] = 0
    splits.extend([tr, te])
    
    # 3. CV5 In-Distribution
    split_id = "CV5_IN_DIST_MAIN"
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(universe)):
        tr = universe.iloc[train_idx].copy()
        tr['split_id'] = split_id
        tr['usage'] = 'train'
        tr['fold'] = fold
        
        te = universe.iloc[test_idx].copy()
        te['split_id'] = split_id
        te['usage'] = 'test'
        te['fold'] = fold
        splits.extend([tr, te])

    full_splits = pd.concat(splits, ignore_index=True)
    full_splits.to_parquet(PIPELINE_DATA / "split_registry.parquet", index=False)
    print(f"    Saved {len(full_splits)} split rows to split_registry.parquet")

if __name__ == "__main__":
    initialize_case_universe()
    initialize_label_registry()
    initialize_split_registry()
