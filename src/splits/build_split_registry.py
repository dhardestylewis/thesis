import pandas as pd
import numpy as np
from pathlib import Path
import sys

# src/splits/build_split_registry.py
sys.path.append(str(Path(r"c:\Users\dhl\data\thesis\thesis") / "src"))
from src.data_io.schema import ROOT_DIR, save_registry

def build_split_registry():
    print("[+] Building Split Registry...")
    universe_path = ROOT_DIR / "registries" / "case_universe.parquet"
    if not universe_path.exists():
        print("    [!] Error: case_universe.parquet missing. Run step 00 first.")
        return

    universe = pd.read_parquet(universe_path)
    
    # Ensure 'year' column exists (extract from filing_date if needed)
    if 'year' not in universe.columns:
        universe['year'] = pd.to_datetime(universe['filing_date']).dt.year

    splits = []
    
    # 1. Primary Temporal OOD Split (Pre-2023 vs 2023+)
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
    
    # 2. CV5 In-Distribution
    cv_id = "CV5_IN_DIST_MAIN"
    shuffled = universe.sample(frac=1, random_seed=42).reset_index(drop=True)
    shuffled['split_id'] = cv_id
    shuffled['fold'] = np.arange(len(shuffled)) % 5
    shuffled['role'] = 'train' # Logic for CV usually sets role per fold, but registry tracks full membership
    splits.append(shuffled)
    
    full_splits = pd.concat(splits, ignore_index=True)
    save_registry(full_splits[['case_id', 'split_id', 'role', 'fold']], "split_registry")
    print(f"    Saved splits to split_registry.parquet")

if __name__ == "__main__":
    build_split_registry()
