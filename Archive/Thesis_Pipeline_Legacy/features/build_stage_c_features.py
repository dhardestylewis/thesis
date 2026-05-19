import pandas as pd
import numpy as np
from pathlib import Path

# Paths
ROOT = Path(r"c:\Users\dhl\data\thesis\thesis")
WAREHOUSE_DIR = ROOT / "Data" / "Warehouse_As_Of"
PIPELINE_DATA = ROOT / "thesis_pipeline" / "data" / "final"

def build_features():
    print("[+] Building Canonical Stage C Features...")
    path = WAREHOUSE_DIR / "H0_Filing_Master_Enriched.csv"
    df = pd.read_csv(path, low_memory=False)
    
    # Strip Explicit Targets, IDs, and weights (as per StageC_opposition_risk.py)
    drop_cols = [
        'is_protested', 'case_number', 'organized_opposition', 'has_audio_record', 
        'TCAD ID', 'date', 'application_start_date', 'final_date', 
        'standardized_tcad_id', 'Prob_H=4', 'Prob_LGBM_H=4', 'Prob_CB_H=4', 
        'Prob_Optimal_H=4', 'ipw', 'council_district_x', 'council_district_y', 
        'ldb_council_district', 'council_district'
    ]
    
    # NLP vectors are leakage for H0
    leak_cols = [c for c in df.columns if c.startswith('tfidf_') or c.startswith('speech_')]
    drop_cols.extend(leak_cols)
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # One-hot encode categoricals
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
    X = X.select_dtypes(include=[np.number])
    
    # Add back keys for registry
    X['case_number'] = df['case_number']
    X['year'] = df['year']
    X = X.dropna(subset=['case_number', 'year'])
    X['as_of_date'] = X['year'].apply(lambda x: f"{int(float(x))}-01-01")
    X['feature_view'] = 'filing_date_public_admin_features'
    
    # Save to registry
    X.to_parquet(PIPELINE_DATA / "feature_registry.parquet", index=False)
    print(f"    Saved {len(X)} cases with {X.shape[1]-4} features to feature_registry.parquet")

if __name__ == "__main__":
    build_features()
