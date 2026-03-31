import pandas as pd
import numpy as np
import os

# Paths
ROOT_DIR = r"C:\Users\dhl\data\thesis\thesis"
COA_ZONING_PATH = os.path.join(ROOT_DIR, "Data", "CoA_Open_Data", "Zoning", "ZC_current_edir-dcnf.csv")
OUT_DIR = os.path.join(ROOT_DIR, "Data", "Warehouse_As_Of", "Build")
os.makedirs(OUT_DIR, exist_ok=True)

def build_case_master():
    print("Loading City of Austin Zoning Cases...")
    df = pd.read_csv(COA_ZONING_PATH)
    
    # 1. Base Case Master Table
    case_master = df[['CASE_NUMBER', 'CASE_NAME', 'CASE_TYPE', 'WORK_TYPE', 
                      'SUB_TYPE', 'DETAILED_STATUS', 'DESCRIPTION_OF_WORK', 
                      'TCAD_ID', 'LOCATION']].copy()
    
    # Extract embedded dates if present in status/description (Temporary fallback before full scrape)
    # The actual filing/notice/pc dates will be merged from the scraped agenda dataset later.
    case_master['filing_date'] = np.nan
    case_master['notice_date'] = np.nan 
    case_master['petition_deadline'] = np.nan
    case_master['planning_commission_date'] = np.nan
    case_master['council_date'] = np.nan
    case_master['withdrawal_date'] = np.nan
    
    # Austin LDC Chapter 25-2 Dimensional Standards
    AUSTIN_LDC_TABLE = {
        'RR':   {'max_height_ft': 35, 'max_far': 0.05, 'max_bldg_cov_pct': 20, 'min_lot_sqft': 43560},
        'LA':   {'max_height_ft': 35, 'max_far': 0.15, 'max_bldg_cov_pct': 40, 'min_lot_sqft': 43560},
        'DR':   {'max_height_ft': 35, 'max_far': 0.15, 'max_bldg_cov_pct': 15, 'min_lot_sqft': 43560},
        'SF-1': {'max_height_ft': 35, 'max_far': 0.20, 'max_bldg_cov_pct': 35, 'min_lot_sqft': 10000},
        'SF-2': {'max_height_ft': 35, 'max_far': 0.35, 'max_bldg_cov_pct': 40, 'min_lot_sqft': 5750},
        'SF-3': {'max_height_ft': 35, 'max_far': 0.40, 'max_bldg_cov_pct': 40, 'min_lot_sqft': 5750},
        'SF-4A':{'max_height_ft': 35, 'max_far': 0.45, 'max_bldg_cov_pct': 45, 'min_lot_sqft': 3600},
        'SF-4B':{'max_height_ft': 35, 'max_far': 0.45, 'max_bldg_cov_pct': 55, 'min_lot_sqft': 3600},
        'SF-5': {'max_height_ft': 35, 'max_far': 0.50, 'max_bldg_cov_pct': 55, 'min_lot_sqft': 5750},
        'SF-6': {'max_height_ft': 35, 'max_far': 0.40, 'max_bldg_cov_pct': 40, 'min_lot_sqft': 5750},
        'MH':   {'max_height_ft': 35, 'max_far': 0.50, 'max_bldg_cov_pct': 50, 'min_lot_sqft': 2500},
        'MF-1': {'max_height_ft': 40, 'max_far': 0.50, 'max_bldg_cov_pct': 45, 'min_lot_sqft': 8000},
        'MF-2': {'max_height_ft': 40, 'max_far': 0.60, 'max_bldg_cov_pct': 50, 'min_lot_sqft': 8000},
        'MF-3': {'max_height_ft': 40, 'max_far': 0.75, 'max_bldg_cov_pct': 55, 'min_lot_sqft': 8000},
        'MF-4': {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 60, 'min_lot_sqft': 8000},
        'MF-5': {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 70, 'min_lot_sqft': 8000},
        'MF-6': {'max_height_ft': 90, 'max_far': 3.00, 'max_bldg_cov_pct': 80, 'min_lot_sqft': 8000},
        'NO':   {'max_height_ft': 35, 'max_far': 0.35, 'max_bldg_cov_pct': 35, 'min_lot_sqft': 5750},
        'LO':   {'max_height_ft': 40, 'max_far': 0.70, 'max_bldg_cov_pct': 50, 'min_lot_sqft': 5750},
        'GO':   {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 60, 'min_lot_sqft': 5750},
        'CR':   {'max_height_ft': 35, 'max_far': 0.35, 'max_bldg_cov_pct': 40, 'min_lot_sqft': 5750},
        'LR':   {'max_height_ft': 40, 'max_far': 0.50, 'max_bldg_cov_pct': 50, 'min_lot_sqft': 5750},
        'GR':   {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 75, 'min_lot_sqft': 5750},
        'CS':   {'max_height_ft': 60, 'max_far': 2.00, 'max_bldg_cov_pct': 95, 'min_lot_sqft': 5750},
        'CS-1': {'max_height_ft': 60, 'max_far': 2.00, 'max_bldg_cov_pct': 95, 'min_lot_sqft': 5750},
        'CH':   {'max_height_ft': 120,'max_far': 3.00, 'max_bldg_cov_pct': 95, 'min_lot_sqft': 5750},
        'IP':   {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 60, 'min_lot_sqft': 5750},
        'LI':   {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 75, 'min_lot_sqft': 5750},
        'MI':   {'max_height_ft': 60, 'max_far': 2.00, 'max_bldg_cov_pct': 85, 'min_lot_sqft': 5750},
        'HI':   {'max_height_ft': 60, 'max_far': 2.00, 'max_bldg_cov_pct': 90, 'min_lot_sqft': 5750},
        'CBD':  {'max_height_ft': 400,'max_far': 8.00, 'max_bldg_cov_pct': 100,'min_lot_sqft': 0},
        'DMU':  {'max_height_ft': 120,'max_far': 5.00, 'max_bldg_cov_pct': 100,'min_lot_sqft': 0},
    }

    import re
    def extract_base_code(z_string):
        if pd.isna(z_string): return None
        parts = re.split(r'[/,]+', str(z_string).upper())
        selected_base = None
        max_height = -1
        for part in parts:
            match = re.match(r'^([A-Z]{2,3}(?:-[1-6A-B]+)?)', part.strip())
            if match:
                base = match.group(1)
                stats = AUSTIN_LDC_TABLE.get(base)
                if stats and stats['max_height_ft'] > max_height:
                    max_height = stats['max_height_ft']
                    selected_base = base
        return selected_base

    def extract_metric(z_string, metric):
        base = extract_base_code(z_string)
        return AUSTIN_LDC_TABLE[base][metric] if base in AUSTIN_LDC_TABLE else np.nan

    # Preserve RAW variables
    case_master['zoning_from'] = df['EXISTING_ZONING']
    case_master['zoning_to'] = df['PROPOSED_ZONING']
    
    # Calculate dimensional vectors
    case_master['requested_height'] = df['PROPOSED_ZONING'].apply(lambda x: extract_metric(x, 'max_height_ft')) - df['EXISTING_ZONING'].apply(lambda x: extract_metric(x, 'max_height_ft'))
    case_master['requested_far'] = df['PROPOSED_ZONING'].apply(lambda x: extract_metric(x, 'max_far')) - df['EXISTING_ZONING'].apply(lambda x: extract_metric(x, 'max_far'))
    case_master['requested_bldg_cov'] = df['PROPOSED_ZONING'].apply(lambda x: extract_metric(x, 'max_bldg_cov_pct')) - df['EXISTING_ZONING'].apply(lambda x: extract_metric(x, 'max_bldg_cov_pct'))
    case_master['requested_lot_sqft'] = df['PROPOSED_ZONING'].apply(lambda x: extract_metric(x, 'min_lot_sqft')) - df['EXISTING_ZONING'].apply(lambda x: extract_metric(x, 'min_lot_sqft'))
    case_master['requested_units'] = np.nan

    
    print(f"Constructed Case Master skeleton with {len(case_master)} records.")
    case_master.to_csv(os.path.join(OUT_DIR, "case_master.csv"), index=False)
    
    return case_master

def build_policy_calendar(case_master):
    print("Building Policy Calendar...")
    # Initialize policy calendar mapped to cases
    policy_df = pd.DataFrame({'CASE_NUMBER': case_master['CASE_NUMBER']})
    
    # Policy: 2022 Council Regime Change (assuming council_date mapping)
    policy_df['council_regime_2022'] = 0 # To be updated with exact date logic
    
    # Policy: HOME Phase 1 (Feb 5, 2024 Application-Acceptance Start)
    home1_date = pd.to_datetime('2024-02-05')
    
    # Policy: HOME Phase 2 (Aug 16, 2024 Application-Acceptance Start)
    home2_date = pd.to_datetime('2024-08-16')
    
    # Policy: HB 24 (Sept 1, 2025 Effective Date)
    hb24_date = pd.to_datetime('2025-09-01')
    
    print("Policy Calendar flags staged. Temporal merges require scraped 'as-of' dates.")
    policy_df.to_csv(os.path.join(OUT_DIR, "policy_calendar.csv"), index=False)
    
    return policy_df

if __name__ == "__main__":
    cm = build_case_master()
    pc = build_policy_calendar(cm)
    print("Step 1 data warehouse generation complete.")
