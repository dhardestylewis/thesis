import pandas as pd
import re
import numpy as np

zoning_metrics_dict = {
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
    'SF':   {'max_height_ft': 35, 'max_far': 0.40, 'max_bldg_cov_pct': 40, 'min_lot_sqft': 5750}, # default SF
    'MH':   {'max_height_ft': 35, 'max_far': 0.50, 'max_bldg_cov_pct': 50, 'min_lot_sqft': 2500},
    'MF-1': {'max_height_ft': 40, 'max_far': 0.50, 'max_bldg_cov_pct': 45, 'min_lot_sqft': 8000},
    'MF-2': {'max_height_ft': 40, 'max_far': 0.60, 'max_bldg_cov_pct': 50, 'min_lot_sqft': 8000},
    'MF-3': {'max_height_ft': 40, 'max_far': 0.75, 'max_bldg_cov_pct': 55, 'min_lot_sqft': 8000},
    'MF-4': {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 60, 'min_lot_sqft': 8000},
    'MF-5': {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 70, 'min_lot_sqft': 8000},
    'MF-6': {'max_height_ft': 90, 'max_far': 3.00, 'max_bldg_cov_pct': 80, 'min_lot_sqft': 8000},
    'MF':   {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 60, 'min_lot_sqft': 8000}, # default MF
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
    'TOD':  {'max_height_ft': 60, 'max_far': 2.00, 'max_bldg_cov_pct': 100,'min_lot_sqft': 0},
    'PUD':  {'max_height_ft': 60, 'max_far': 2.00, 'max_bldg_cov_pct': 100,'min_lot_sqft': 0},
    'ERC':  {'max_height_ft': 60, 'max_far': 2.00, 'max_bldg_cov_pct': 100,'min_lot_sqft': 0},
    'P':    {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 60, 'min_lot_sqft': 5750},
    'AG':   {'max_height_ft': 35, 'max_far': 0.05, 'max_bldg_cov_pct': 20, 'min_lot_sqft': 435600},
    'W':    {'max_height_ft': 35, 'max_far': 0.00, 'max_bldg_cov_pct': 0,  'min_lot_sqft': 435600}
}

def extract_base_zone(z):
    if pd.isna(z) or not z: return None
    import re
    match = re.search(r'^([A-Z]+(?:-[0-9]+[A-Z]*)?)', str(z).upper())
    if match:
        base = match.group(1)
        if base in zoning_metrics_dict: return base
        base = base.split('-')[0]
        if base in zoning_metrics_dict: return base
    return None

df = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv")

for metric in ['max_height_ft', 'max_far', 'max_bldg_cov_pct', 'min_lot_sqft']:
    df[f'Initial_{metric}'] = np.nan
    df[f'Requested_{metric}'] = np.nan
    df[f'Approved_{metric}'] = np.nan
    
for idx, row in df.iterrows():
    init_z = extract_base_zone(row['Initial_Zoning'])
    req_z = extract_base_zone(row['Requested_Zoning'])
    app_z = extract_base_zone(row['Final_Zoning'])
    
    if init_z and init_z in zoning_metrics_dict:
        df.at[idx, 'Initial_max_height_ft'] = zoning_metrics_dict[init_z]['max_height_ft']
        df.at[idx, 'Initial_max_far'] = zoning_metrics_dict[init_z]['max_far']
        df.at[idx, 'Initial_max_bldg_cov_pct'] = zoning_metrics_dict[init_z]['max_bldg_cov_pct']
        df.at[idx, 'Initial_min_lot_sqft'] = zoning_metrics_dict[init_z]['min_lot_sqft']
        
    if req_z and req_z in zoning_metrics_dict:
        df.at[idx, 'Requested_max_height_ft'] = zoning_metrics_dict[req_z]['max_height_ft']
        df.at[idx, 'Requested_max_far'] = zoning_metrics_dict[req_z]['max_far']
        df.at[idx, 'Requested_max_bldg_cov_pct'] = zoning_metrics_dict[req_z]['max_bldg_cov_pct']
        df.at[idx, 'Requested_min_lot_sqft'] = zoning_metrics_dict[req_z]['min_lot_sqft']
        
    if app_z and app_z in zoning_metrics_dict:
        df.at[idx, 'Approved_max_height_ft'] = zoning_metrics_dict[app_z]['max_height_ft']
        df.at[idx, 'Approved_max_far'] = zoning_metrics_dict[app_z]['max_far']
        df.at[idx, 'Approved_max_bldg_cov_pct'] = zoning_metrics_dict[app_z]['max_bldg_cov_pct']
        df.at[idx, 'Approved_min_lot_sqft'] = zoning_metrics_dict[app_z]['min_lot_sqft']

df['Delta_Requested_Height'] = df['Requested_max_height_ft'] - df['Initial_max_height_ft']
df['Delta_Approved_Height'] = df['Approved_max_height_ft'] - df['Initial_max_height_ft']
df['Height_Attrition'] = df['Delta_Requested_Height'] - df['Delta_Approved_Height']

df['Delta_Requested_FAR'] = df['Requested_max_far'] - df['Initial_max_far']
df['Delta_Approved_FAR'] = df['Approved_max_far'] - df['Initial_max_far']
df['FAR_Attrition'] = df['Delta_Requested_FAR'] - df['Delta_Approved_FAR']

print("Successfully generated continuous density metrics.")
print(df[['case_number', 'Initial_Zoning', 'Requested_Zoning', 'Final_Zoning', 'Delta_Requested_Height', 'Delta_Approved_Height', 'Height_Attrition']].dropna(subset=['Height_Attrition']).head(15))

df.to_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv", index=False)
