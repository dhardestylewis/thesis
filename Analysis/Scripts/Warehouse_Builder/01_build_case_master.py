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
    
    # Placeholders for ZONING FROM / TO, UNITS, FAR, HEIGHT
    case_master['zoning_from'] = np.nan
    case_master['zoning_to'] = np.nan
    case_master['requested_units'] = np.nan
    case_master['requested_far'] = np.nan
    case_master['requested_height'] = np.nan
    
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
