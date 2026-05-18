import os
import pandas as pd
import numpy as np
import glob

# Configuration
BASE_DIR = r"C:\Users\dhl\data\Thesis\thesis"
DATA_DIR = os.path.join(BASE_DIR, "Data")
CRADLE_PATH = os.path.join(DATA_DIR, "Panel", "cradle_to_grave_dataset.csv")
X_PATH = os.path.join(DATA_DIR, "Panel", "cross_sectional_dml_panel.csv")
EARS_DIR = os.path.join(DATA_DIR, "raw", "EARS")
OUT_PATH = os.path.join(DATA_DIR, "Panel", "annual_lifecycle_panel.csv")

def fetch_macro_ppi():
    print("Fetching Construction PPI (WPUSI012011) from FRED...")
    try:
        ppi = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=WPUSI012011", 
                          parse_dates=['observation_date'], na_values='.')
        ppi.rename(columns={'observation_date': 'date', 'WPUSI012011': 'ppi'}, inplace=True)
        ppi['ppi'] = pd.to_numeric(ppi['ppi'], errors='coerce')
        ppi = ppi.dropna()
        ppi['year'] = ppi['date'].dt.year
        return ppi.groupby('year')['ppi'].mean().to_dict()
    except Exception as e:
        print(f"Error fetching FRED: {e}. Proceeding without PPI.")
        return {}

def load_ears():
    print("Loading EARS Longitudinal Array...")
    ears_files = glob.glob(os.path.join(EARS_DIR, "ears_*.csv"))
    ears_state = {} # acc -> {year: {imprv, luc, built}}
    
    for f in sorted(ears_files):
        year = int(os.path.basename(f).replace('ears_', '').replace('.csv', ''))
        try:
            df = pd.read_csv(f, usecols=['account_number', 'improvement_market_value', 'land_use_code', 'year_built'], on_bad_lines='skip')
            df['account_number'] = df['account_number'].astype(str).str.split('.').str[0].str.zfill(10)
            df = df[~df['account_number'].isin(['0000000000', 'NAN', 'NAN0000000'])]
            df['improvement_market_value'] = pd.to_numeric(df['improvement_market_value'], errors='coerce')
            df['year_built'] = pd.to_numeric(df['year_built'], errors='coerce')
            
            # Keep latest if duplicates exist
            df = df.drop_duplicates(subset=['account_number'], keep='last')
            
            # Add to dict
            for _, row in df.iterrows():
                acc = row['account_number']
                if acc not in ears_state:
                    ears_state[acc] = {}
                ears_state[acc][year] = {
                    'imprv_val': row['improvement_market_value'],
                    'luc': row['land_use_code'],
                    'year_built': row['year_built']
                }
        except Exception as e:
            print(f"Error loading EARS {year}: {e}")
            
    return ears_state

def build_panel():
    annual_ppi = fetch_macro_ppi()
    ears = load_ears()
    
    print("Loading Project Baselines...")
    df_base = pd.read_csv(CRADLE_PATH)
    df_x = pd.read_csv(X_PATH)
    
    # Merge static DML features into the baseline to prep for massive unrolling
    static_exclude = ['latitude', 'longitude', 'appraised_value', 'building_age'] # Remove potentially dynamic ones that overlap
    x_features = [c for c in df_x.columns if c not in static_exclude]
    df_base = pd.merge(df_base, df_x[x_features], on='case_number', how='inner')
    
    panel_rows = []
    
    print(f"Unrolling {len(df_base)} projects across time...")
    
    for _, row in df_base.iterrows():
        app_year = int(row['application_year'])
        if pd.isna(app_year): continue
        
        tcad = str(row['tcad_id']).split('.')[0].zfill(10)
        issue_year = row['permit_issue_year']
        final_year = row['permit_final_year']
        is_abandoned = row['is_abandoned']
        
        # Iterate from application year up to 2024
        for year in range(app_year, 2025):
            r = row.to_dict()
            r['year'] = year
            r['years_since_application'] = year - app_year
            r['macro_ppi_t'] = annual_ppi.get(year, np.nan)
            
            # --- LIFECYCLE PHASE LOGIC ---
            # Mutually exclusive flags
            p_ent = 0
            p_exec = 0
            p_stab = 0
            p_post = 0
            p_abnd = 0
            
            if is_abandoned == 1:
                # If abandoned, it sits in entitlement until 5 years pass, then officially 'abandoned'
                if year - app_year < 5:
                    p_ent = 1
                else:
                    p_abnd = 1
            else:
                if pd.notna(issue_year) and pd.notna(final_year):
                    if year < issue_year:
                        p_ent = 1
                    elif issue_year <= year <= final_year:
                        p_exec = 1
                    elif final_year < year <= final_year + 2:
                        p_stab = 1
                    elif year > final_year + 2:
                        p_post = 1
                elif pd.notna(issue_year) and pd.isna(final_year):
                    # Stuck in execution forever
                    if year < issue_year:
                        p_ent = 1
                    else:
                        p_exec = 1
                else:
                    # No permit issue year but not abandoned (edge case, assume entitlement)
                    p_ent = 1

            r['phase_entitlement'] = p_ent
            r['phase_execution'] = p_exec
            r['phase_stabilization'] = p_stab
            r['phase_post_stabilization'] = p_post
            r['phase_abandoned'] = p_abnd
            
            # String representation for easy plotting
            if p_ent: phase_str = "1_Entitlement"
            elif p_exec: phase_str = "2_Execution"
            elif p_stab: phase_str = "3_Stabilization"
            elif p_post: phase_str = "4_Post_Stabilization"
            elif p_abnd: phase_str = "5_Abandoned"
            else: phase_str = "Unknown"
            r['current_lifecycle_phase'] = phase_str
            
            # --- EARS FINANCIAL TRACKING ---
            ears_val = np.nan
            ears_built = np.nan
            ears_luc = ""
            
            if tcad in ears and year in ears[tcad]:
                data = ears[tcad][year]
                ears_val = data['imprv_val']
                ears_built = data['year_built']
                ears_luc = data['luc']
            else:
                # If EARS is missing for this exact year, try to forward fill from previous years if available
                if tcad in ears:
                    past_years = [y for y in ears[tcad].keys() if y < year]
                    if past_years:
                        closest_past = max(past_years)
                        data = ears[tcad][closest_past]
                        ears_val = data['imprv_val']
                        ears_built = data['year_built']
                        ears_luc = data['luc']
            
            r['ears_appraised_imprv_value_t'] = ears_val
            r['ears_year_built_t'] = ears_built
            r['ears_land_use_code_t'] = ears_luc
            
            panel_rows.append(r)
            
    print(f"Generated {len(panel_rows):,} total longitudinal records.")
    
    df_panel = pd.DataFrame(panel_rows)
    df_panel.to_csv(OUT_PATH, index=False)
    print(f"Annual Lifecycle Panel saved to: {OUT_PATH}")

if __name__ == "__main__":
    build_panel()
