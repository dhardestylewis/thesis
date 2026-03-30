"""
zoning_delta_calculator.py
==========================
Translates raw Austin zoning strings into exact statutory dimensional standards 
(Height, FAR, Building Coverage, Min Lot Size) defined in Austin Land Development Code Chapter 25-2.
Calculates the exact delta in limits (the literal statutory shock).
"""
import pandas as pd
import numpy as np
import os
import re

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA_DIR = os.path.join(ROOT, "Data")
ZONING_IN = os.path.join(DATA_DIR, "Zoning_Cases", "Processed_Data", "CSV", "enriched_zoning_data_updated.csv")
ZONING_OUT = os.path.join(DATA_DIR, "Zoning_Cases", "Processed_Data", "CSV", "enriched_zoning_data_causal.csv")

# Exact Austin LDC Chapter 25-2 Dimensional Standards
# Format: Base_Code: { 'max_height_ft': X, 'max_far': X.XX, 'max_bldg_cov_pct': X, 'min_lot_sqft': X }
AUSTIN_LDC_TABLE = {
    # Rural & Low Density
    'RR':   {'max_height_ft': 35, 'max_far': 0.05, 'max_bldg_cov_pct': 20, 'min_lot_sqft': 43560}, # 1 acre
    'LA':   {'max_height_ft': 35, 'max_far': 0.15, 'max_bldg_cov_pct': 40, 'min_lot_sqft': 43560},
    'DR':   {'max_height_ft': 35, 'max_far': 0.15, 'max_bldg_cov_pct': 15, 'min_lot_sqft': 43560},
    # Single Family
    'SF-1': {'max_height_ft': 35, 'max_far': 0.20, 'max_bldg_cov_pct': 35, 'min_lot_sqft': 10000},
    'SF-2': {'max_height_ft': 35, 'max_far': 0.35, 'max_bldg_cov_pct': 40, 'min_lot_sqft': 5750},
    'SF-3': {'max_height_ft': 35, 'max_far': 0.40, 'max_bldg_cov_pct': 40, 'min_lot_sqft': 5750},
    'SF-4A':{'max_height_ft': 35, 'max_far': 0.45, 'max_bldg_cov_pct': 45, 'min_lot_sqft': 3600},
    'SF-4B':{'max_height_ft': 35, 'max_far': 0.45, 'max_bldg_cov_pct': 55, 'min_lot_sqft': 3600},
    'SF-5': {'max_height_ft': 35, 'max_far': 0.50, 'max_bldg_cov_pct': 55, 'min_lot_sqft': 5750}, # Condos can subdivide
    'SF-6': {'max_height_ft': 35, 'max_far': 0.40, 'max_bldg_cov_pct': 40, 'min_lot_sqft': 5750},
    'MH':   {'max_height_ft': 35, 'max_far': 0.50, 'max_bldg_cov_pct': 50, 'min_lot_sqft': 2500},
    # Multi-Family
    'MF-1': {'max_height_ft': 40, 'max_far': 0.50, 'max_bldg_cov_pct': 45, 'min_lot_sqft': 8000},
    'MF-2': {'max_height_ft': 40, 'max_far': 0.60, 'max_bldg_cov_pct': 50, 'min_lot_sqft': 8000},
    'MF-3': {'max_height_ft': 40, 'max_far': 0.75, 'max_bldg_cov_pct': 55, 'min_lot_sqft': 8000},
    'MF-4': {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 60, 'min_lot_sqft': 8000},
    'MF-5': {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 70, 'min_lot_sqft': 8000},
    'MF-6': {'max_height_ft': 90, 'max_far': 3.00, 'max_bldg_cov_pct': 80, 'min_lot_sqft': 8000},
    # Office
    'NO':   {'max_height_ft': 35, 'max_far': 0.35, 'max_bldg_cov_pct': 35, 'min_lot_sqft': 5750},
    'LO':   {'max_height_ft': 40, 'max_far': 0.70, 'max_bldg_cov_pct': 50, 'min_lot_sqft': 5750},
    'GO':   {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 60, 'min_lot_sqft': 5750},
    # Commercial
    'CR':   {'max_height_ft': 35, 'max_far': 0.35, 'max_bldg_cov_pct': 40, 'min_lot_sqft': 5750},
    'LR':   {'max_height_ft': 40, 'max_far': 0.50, 'max_bldg_cov_pct': 50, 'min_lot_sqft': 5750},
    'GR':   {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 75, 'min_lot_sqft': 5750},
    'CS':   {'max_height_ft': 60, 'max_far': 2.00, 'max_bldg_cov_pct': 95, 'min_lot_sqft': 5750},
    'CS-1': {'max_height_ft': 60, 'max_far': 2.00, 'max_bldg_cov_pct': 95, 'min_lot_sqft': 5750},
    'CH':   {'max_height_ft': 120,'max_far': 3.00, 'max_bldg_cov_pct': 95, 'min_lot_sqft': 5750},
    # Industrial
    'IP':   {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 60, 'min_lot_sqft': 5750},
    'LI':   {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 75, 'min_lot_sqft': 5750},
    'MI':   {'max_height_ft': 60, 'max_far': 2.00, 'max_bldg_cov_pct': 85, 'min_lot_sqft': 5750},
    'HI':   {'max_height_ft': 60, 'max_far': 2.00, 'max_bldg_cov_pct': 90, 'min_lot_sqft': 5750},
    # Downtown/Core
    'CBD':  {'max_height_ft': 400,'max_far': 8.00, 'max_bldg_cov_pct': 100,'min_lot_sqft': 0},
    'DMU':  {'max_height_ft': 120,'max_far': 5.00, 'max_bldg_cov_pct': 100,'min_lot_sqft': 0},
}

def extract_base_code(z_string):
    """Isolates the highest intensity base category."""
    if pd.isna(z_string): return None
    parts = re.split(r'[/,]+', str(z_string).upper())
    
    selected_base = None
    max_height = -1
    for part in parts:
        part = part.strip()
        match = re.match(r'^([A-Z]{2,3}(?:-[1-6A-B]+)?)', part)
        if match:
            base = match.group(1)
            # Find in dictionary
            stats = AUSTIN_LDC_TABLE.get(base)
            if stats:
                if stats['max_height_ft'] > max_height:
                    max_height = stats['max_height_ft']
                    selected_base = base
    return selected_base

def extract_metric(z_string, metric):
    base = extract_base_code(z_string)
    if base and base in AUSTIN_LDC_TABLE:
        return AUSTIN_LDC_TABLE[base][metric]
    return np.nan

def main():
    print(f"Loading raw zoning cases from {ZONING_IN}...")
    if not os.path.exists(ZONING_IN):
        print("Data not found.")
        return
        
    df = pd.read_csv(ZONING_IN, low_memory=False)
    
    print("Translating raw zoning categories to explicit mathematical LDC limits...")
    metrics = ['max_height_ft', 'max_far', 'max_bldg_cov_pct', 'min_lot_sqft']
    
    for metric in metrics:
        df[f'existing_{metric}'] = df['existing_zoning'].apply(lambda x: extract_metric(x, metric))
        df[f'proposed_{metric}'] = df['proposed_zoning'].apply(lambda x: extract_metric(x, metric))
        
        # Calculate exactly what the user actually cares about (the math)
        df[f'delta_{metric}'] = df[f'proposed_{metric}'] - df[f'existing_{metric}']
    
    print(f"Calculated statutory deltas for {len(df)} cases.")
    print("Example translated row:")
    row = df.loc[df['delta_max_height_ft'].notna()].iloc[0]
    print(row[['existing_zoning', 'proposed_zoning', 'existing_max_height_ft', 'proposed_max_height_ft', 'delta_max_height_ft']])
    
    df.to_csv(ZONING_OUT, index=False)
    print(f"Saved true structural LDC dataset to: {ZONING_OUT}")

if __name__ == "__main__":
    main()
