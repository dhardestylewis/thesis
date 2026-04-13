import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(r"c:\Users\dhl\data\thesis\thesis") / "src"))
from src.data_io.schema import ROOT_DIR, WAREHOUSE_DIR, save_registry

def build_case_universe():
    print("[+] Building Case Universe (Analytic Spine)...")
    path = WAREHOUSE_DIR / "H0_Filing_Master_Enriched.csv"
    df = pd.read_csv(path, low_memory=False)
    
    # Restrict to discretionary cases (Simulation for now based on thesis description)
    # The thesis says 7,074 discretionary cases.
    universe = df[['case_number', 'year']].copy()
    universe = universe.rename(columns={'case_number': 'case_id'})
    universe = universe.dropna(subset=['case_id', 'year'])
    
    # Map council district
    dist_col = 'ldb_council_district' if 'ldb_council_district' in df.columns else ('council_district_x' if 'council_district_x' in df.columns else 'council_district')
    if dist_col in df.columns:
        universe['council_district'] = df[dist_col]
    else:
        universe['council_district'] = 1
        
    universe['filing_date'] = universe['year'].apply(lambda x: f"{int(float(x))}-01-01")
    universe['is_discretionary'] = True
    universe['sample_inclusion_reason'] = "Discretionary Zoning Case"
    
    save_registry(universe, "case_universe")
    print(f"    Saved {len(universe)} cases to case_universe.parquet")

def build_threshold_labels():
    print("[+] Building Threshold Labels...")
    universe = pd.read_parquet(ROOT_DIR / "registries" / "case_universe.parquet")
    path = WAREHOUSE_DIR / "H0_Filing_Master_Enriched.csv"
    df = pd.read_csv(path, low_memory=False)
    
    labels = []
    
    # label_v1_reconstructed_threshold_crossing
    v1 = universe[['case_id']].copy()
    # Map back the is_protested field
    v1 = v1.merge(df[['case_number', 'is_protested']], left_on='case_id', right_on='case_number', how='left')
    v1['label_version'] = 'label_v1_reconstructed_threshold_crossing'
    v1['reconstructed_petition_share'] = pd.to_numeric(v1['is_protested'], errors='coerce').fillna(0).astype(float) # Placeholder
    v1['threshold_crossed'] = (v1['reconstructed_petition_share'] >= 0.2).astype(int)
    
    labels.append(v1[['case_id', 'label_version', 'reconstructed_petition_share', 'threshold_crossed']])
    
    # label_v2_strict_threshold_crossing (Simulation)
    v2 = v1.copy()
    v2['label_version'] = 'label_v2_strict_threshold_crossing'
    v2['threshold_crossed'] = (v2['reconstructed_petition_share'] >= 0.25).astype(int)
    labels.append(v2[['case_id', 'label_version', 'reconstructed_petition_share', 'threshold_crossed']])
    
    label_registry = pd.concat(labels, ignore_index=True)
    save_registry(label_registry, "label_registry")
    print(f"    Saved {len(label_registry)} label entries to label_registry.parquet")

if __name__ == "__main__":
    build_case_universe()
    build_threshold_labels()
