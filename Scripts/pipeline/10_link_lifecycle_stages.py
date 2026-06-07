import os
import pandas as pd
import numpy as np
import glob
import csv
import statistics

# Configuration Paths
BASE_DIR = r"C:\Users\dhl\data\Thesis\thesis"
DATA_DIR = os.path.join(BASE_DIR, "Data")
ZONING_PATH = os.path.join(DATA_DIR, "Zoning_Cases", "Processed_Data", "CSV", "zoning_land_use_merged_data.csv")
PERMIT_PATH = os.path.join(DATA_DIR, "CoA_Open_Data", "Issued_Building_Permits.csv")
EARS_DIR = os.path.join(DATA_DIR, "raw", "EARS")
OUT_PATH = os.path.join(DATA_DIR, "Panel", "cradle_to_grave_dataset.csv")

# Ensure large field limits for dirty data
csv.field_size_limit(2147483647)

def fetch_macro_ppi():
    print("Fetching Construction PPI (WPUSI012011) from FRED...")
    ppi = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=WPUSI012011", 
                      parse_dates=['observation_date'], na_values='.')
    ppi.rename(columns={'observation_date': 'date', 'WPUSI012011': 'ppi'}, inplace=True)
    ppi['ppi'] = pd.to_numeric(ppi['ppi'], errors='coerce')
    ppi = ppi.dropna().sort_values('date')
    # Get annual average PPI for indexation
    ppi['year'] = ppi['date'].dt.year
    annual_ppi = ppi.groupby('year')['ppi'].mean().to_dict()
    return annual_ppi

def load_zoning():
    print("Loading Zoning Cases Baseline...")
    df = pd.read_csv(ZONING_PATH, usecols=['case_number', 'tcad_id', 'application_start_date', 'gross_site_area_acres', 'existing_zoning', 'proposed_zoning'])
    df['tcad_id'] = df['tcad_id'].astype(str).str.split('.').str[0].str.zfill(10)
    df['app_year'] = pd.to_datetime(df['application_start_date'], errors='coerce').dt.year
    df = df[df['tcad_id'].notna() & df['app_year'].notna()]
    # Drop duplicates to maintain 1:1 mapping for simplicity (take latest application)
    df = df.sort_values('application_start_date').drop_duplicates('case_number', keep='last')
    return df

def load_permits():
    print("Loading Building Permits (Execution)...")
    permits = {} # tcad -> {'total_cost': 0, 'issue_year': min, 'final_year': max, 'status': []}
    
    with open(PERMIT_PATH, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        headers = [h.upper().strip() for h in next(reader)]
        
        tcad_idx = headers.index('TCAD_ID') if 'TCAD_ID' in headers else -1
        val_idx = headers.index('TOTAL_JOB_VALUATION') if 'TOTAL_JOB_VALUATION' in headers else -1
        issue_idx = headers.index('ISSUE_DATE') if 'ISSUE_DATE' in headers else -1
        final_idx = headers.index('FINAL_DATE') if 'FINAL_DATE' in headers else -1
        type_idx = headers.index('WORK_TYPE') if 'WORK_TYPE' in headers else -1
        status_idx = headers.index('PERMIT_STATUS') if 'PERMIT_STATUS' in headers else -1
        
        if -1 in (tcad_idx, val_idx, issue_idx, final_idx):
            print("Missing permit columns!")
            return {}
            
        for row in reader:
            if len(row) <= max(tcad_idx, val_idx, issue_idx, final_idx, type_idx, status_idx): continue
            
            tcad = str(row[tcad_idx]).strip().split('.')[0].zfill(10)
            if not tcad or tcad == '0000000000': continue
            
            # Filter for New Construction / Commercial
            p_type = str(row[type_idx]).upper()
            if 'NEW' not in p_type and 'COMMERCIAL' not in p_type and 'MULTI' not in p_type:
                continue
                
            try: cost = float(row[val_idx])
            except: cost = 0
            
            # Parse dates
            def get_yr(d):
                d = str(d).strip()
                if '/' in d: return int(d.split('/')[-1][:4])
                elif '-' in d: return int(d.split('-')[0][:4])
                return 0
                
            issue_yr = get_yr(row[issue_idx])
            final_yr = get_yr(row[final_idx])
            status = str(row[status_idx]).upper().strip()
            
            if tcad not in permits:
                permits[tcad] = {'cost': 0, 'issue_year': 9999, 'final_year': 0, 'statuses': set()}
            
            permits[tcad]['cost'] += cost
            permits[tcad]['statuses'].add(status)
            if issue_yr > 0: permits[tcad]['issue_year'] = min(permits[tcad]['issue_year'], issue_yr)
            if final_yr > 0: permits[tcad]['final_year'] = max(permits[tcad]['final_year'], final_yr)
            
    return permits

def load_ears_stabilization():
    print("Loading EARS Longitudinal Panel (Realization)...")
    ears_files = glob.glob(os.path.join(EARS_DIR, "ears_*.csv"))
    ears_state = {} # tcad -> { year: {imprv_val, luc, year_built} }
    
    all_dfs = []
    for f in sorted(ears_files):
        year = int(os.path.basename(f).replace('ears_', '').replace('.csv', ''))
        try:
            df = pd.read_csv(f, usecols=['account_number', 'improvement_market_value', 'land_use_code', 'year_built'], on_bad_lines='skip')
            df['year'] = year
            df['account_number'] = df['account_number'].astype(str).str.split('.').str[0].str.zfill(10)
            df = df[~df['account_number'].isin(['0000000000', 'NAN', 'NAN0000000'])]
            df['improvement_market_value'] = pd.to_numeric(df['improvement_market_value'], errors='coerce')
            df['year_built'] = pd.to_numeric(df['year_built'], errors='coerce')
            all_dfs.append(df)
        except Exception as e:
            print(f"Error loading EARS {year}: {e}")
            
    if all_dfs:
        full_ears = pd.concat(all_dfs, ignore_index=True)
        # Convert to dictionary structure instantly
        ears_state = {}
        # Ensure index is unique by dropping duplicates
        full_ears = full_ears.drop_duplicates(subset=['account_number', 'year'], keep='last')
        grouped_dict = full_ears.set_index(['account_number', 'year'])[['improvement_market_value', 'land_use_code', 'year_built']].to_dict(orient='index')
        
        for (acc, year), vals in grouped_dict.items():
            if acc not in ears_state:
                ears_state[acc] = {}
            ears_state[acc][year] = {
                'imprv_val': vals['improvement_market_value'],
                'luc': vals['land_use_code'],
                'year_built': vals['year_built']
            }
    return ears_state

def build_pipeline():
    annual_ppi = fetch_macro_ppi()
    df_zoning = load_zoning()
    permits = load_permits()
    ears = load_ears_stabilization()
    
    results = []
    
    print("\nMerging Cradle-to-Grave Lifecycle...")
    for _, row in df_zoning.iterrows():
        tcad = row['tcad_id']
        case = row['case_number']
        app_yr = row['app_year']
        
        # 1. Base Project Data
        proj = {
            'case_number': case,
            'tcad_id': tcad,
            'application_year': app_yr,
            'zoning_friction_active': 1,
            'gross_site_area_acres': row['gross_site_area_acres'],
            'proposed_zoning': row['proposed_zoning']
        }
        
        # 2. Execution Phase (Permits)
        if tcad in permits:
            p = permits[tcad]
            # Only consider permits pulled *after* zoning application
            if p['issue_year'] != 9999 and p['issue_year'] >= app_yr:
                proj['permit_issue_year'] = p['issue_year']
                proj['permit_final_year'] = p['final_year'] if p['final_year'] > 0 else np.nan
                proj['declared_cost'] = p['cost']
                
                # Inflation adjustment (index cost from issue_year to final_year)
                issue_ppi = annual_ppi.get(proj['permit_issue_year'], annual_ppi.get(2020))
                final_ppi = annual_ppi.get(proj['permit_final_year'], annual_ppi.get(2024))
                
                if issue_ppi and final_ppi and issue_ppi > 0:
                    proj['inflation_adjusted_cost'] = proj['declared_cost'] * (final_ppi / issue_ppi)
                else:
                    proj['inflation_adjusted_cost'] = proj['declared_cost']
                
                # Check for Abandonment
                is_expired = 'EXPIRED' in p['statuses']
                is_final = 'FINAL' in p['statuses']
                proj['is_abandoned'] = 1 if (is_expired and not is_final) else 0
                
            else:
                proj['is_abandoned'] = 1 if (2024 - app_yr >= 5) else 0
        else:
            # No permit pulled within 5 years of application -> Abandoned
            proj['is_abandoned'] = 1 if (2024 - app_yr >= 5) else 0
            
        # 3. Realization Phase (EARS Stabilization)
        if tcad in ears:
            e_hist = ears[tcad]
            available_years = sorted(list(e_hist.keys()))
            
            # Determine stabilization target year
            stab_year = proj.get('permit_final_year', np.nan)
            if pd.notna(stab_year):
                target_yr = int(stab_year) + 2
                
                # Find the closest available EARS year (<= target_yr or latest)
                target_ears = min([y for y in available_years if y >= target_yr] + [available_years[-1]])
                
                data = e_hist[target_ears]
                proj['stabilized_ears_year'] = target_ears
                proj['stabilized_imprv_value'] = data['imprv_val'] if pd.notna(data['imprv_val']) else 0
                proj['final_land_use_code'] = data['luc']
                proj['final_year_built'] = data['year_built']
                
                # ROI Calculation
                adj_cost = proj.get('inflation_adjusted_cost', 0)
                if adj_cost > 0 and proj.get('stabilized_imprv_value', 0) > 0:
                    proj['value_created'] = proj['stabilized_imprv_value'] - adj_cost
                    proj['roi_pct'] = (proj['value_created'] / adj_cost) * 100
                else:
                    proj['value_created'] = 0
                    proj['roi_pct'] = 0
                    
        results.append(proj)
        
    df_final = pd.DataFrame(results)
    df_final.to_csv(OUT_PATH, index=False)
    
    print(f"\n--- LIFECYCLE LINKAGE COMPLETE ---")
    print(f"Total Zoning Cases Tracked: {len(df_final):,}")
    print(f"Projects Abandoned (Friction Victim): {df_final['is_abandoned'].sum():,} ({(df_final['is_abandoned'].mean()*100):.1f}%)")
    completed = df_final[df_final['roi_pct'].notna() & (df_final['roi_pct'] != 0)]
    print(f"Fully Stabilized Projects with ROI: {len(completed):,}")
    if len(completed) > 0:
        print(f"Median Stabilized Value Created: ${completed['value_created'].median():,.0f}")
        print(f"Median Stabilized Structure ROI: {completed['roi_pct'].median():.1f}%")
        
    print(f"\nTarget dataset saved to: {OUT_PATH}")

if __name__ == "__main__":
    build_pipeline()
