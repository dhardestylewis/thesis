import pandas as pd
import numpy as np
import sys
from pathlib import Path

# src/features/build_stage_a_features.py
sys.path.append(str(Path(r"c:\Users\dhl\data\thesis\thesis") / "src"))
from data_io.schema import ROOT_DIR, save_registry

def build_stage_a_features():
    print("[+] Building Stage A Hazard Features...")
    
    # Simulation: Stage A usually focuses on site-level hazard features for selection
    universe = pd.read_parquet(ROOT_DIR / "registries" / "case_universe.parquet")
    
    X = universe[['case_id', 'filing_date']].copy()
    X['as_of_date'] = X['filing_date']
    X['feature_view'] = 'stage_a_hazard'
    
    # Selection predictors: market pressure, local vacancy, regulatory barriers
    X['mkt_pressure'] = np.random.randn(len(X))
    X['local_vacancy'] = np.random.randn(len(X))
    X['regulatory_tier'] = np.random.choice([1, 2, 3], len(X))
    
    # Save to interim
    interim_path = ROOT_DIR / "data" / "interim" / "stage_a_features_raw.parquet"
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(interim_path, index=False)
    
    print(f"    Stage A features built for {len(X)} cases.")

if __name__ == "__main__":
    build_stage_a_features()
