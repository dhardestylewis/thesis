import pandas as pd
import json
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
    match = re.search(r'^([A-Z]+(?:-[0-9]+[A-Z]*)?)', str(z).upper())
    if match:
        base = match.group(1)
        if base in zoning_metrics_dict: return base
        base = base.split('-')[0]
        if base in zoning_metrics_dict: return base
    return None

df = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv")

df['Staff_Recommendation'] = pd.Series(dtype='object')

for idx, row in df.iterrows():
    traj_str = row['Zoning_Trajectory']
    if pd.notna(traj_str) and str(traj_str).strip() != 'nan':
        try:
            trajectory = json.loads(traj_str)
            staff_rec = None
            for event in trajectory:
                if 'staff_recommendation' in event and event['staff_recommendation']:
                    staff_rec = event['staff_recommendation']
            if staff_rec:
                df.at[idx, 'Staff_Recommendation'] = staff_rec
        except Exception as e:
            pass

for metric in ['max_height_ft', 'max_far', 'max_bldg_cov_pct', 'min_lot_sqft']:
    df[f'Staff_{metric}'] = np.nan
    
for idx, row in df.iterrows():
    staff_z = extract_base_zone(row['Staff_Recommendation'])
    if staff_z and staff_z in zoning_metrics_dict:
        df.at[idx, 'Staff_max_height_ft'] = zoning_metrics_dict[staff_z]['max_height_ft']
        df.at[idx, 'Staff_max_far'] = zoning_metrics_dict[staff_z]['max_far']
        df.at[idx, 'Staff_max_bldg_cov_pct'] = zoning_metrics_dict[staff_z]['max_bldg_cov_pct']
        df.at[idx, 'Staff_min_lot_sqft'] = zoning_metrics_dict[staff_z]['min_lot_sqft']

df['Phase_Requested_SqFt'] = df['shape_area'] * df['Requested_max_far']
df['Phase_Staff_SqFt'] = df['shape_area'] * df['Staff_max_far'].fillna(df['Requested_max_far'])
df['Phase_Approved_SqFt'] = df['shape_area'] * df['Approved_max_far']

df['Staff_Attrition_SqFt'] = df['Phase_Requested_SqFt'] - df['Phase_Staff_SqFt']
df['Council_Attrition_SqFt'] = df['Phase_Staff_SqFt'] - df['Phase_Approved_SqFt']

df['Staff_Effective_Height'] = df['Staff_max_height_ft'].fillna(df['Requested_max_height_ft'])

def apply_cap(row):
    zoned_height = row['Staff_Effective_Height']
    comp_cap = row['GIS_Compatibility_Height_Cap']
    if pd.isna(zoned_height):
        return np.nan
    return min(zoned_height, comp_cap)

df['Staff_Effective_Height'] = df.apply(apply_cap, axis=1)
df.loc[df['Effective_Approved_Height'] > 9000, 'Effective_Approved_Height'] = np.nan

df['Staff_Attrition_Height'] = df['Requested_max_height_ft'] - df['Staff_Effective_Height']
df['Council_Attrition_Height'] = df['Staff_Effective_Height'] - df['Effective_Approved_Height']

# Drop the columns that used the forbidden word just to be safe
cols_to_drop = [c for c in df.columns if 'Friction' in c or 'friction' in c]
df = df.drop(columns=cols_to_drop)

print("Successfully calculated Temporal Attrition Deltas.")
print(df[['case_number', 'Requested_Zoning', 'Staff_Recommendation', 'Final_Zoning', 'Staff_Attrition_Height', 'Council_Attrition_Height']].dropna(subset=['Staff_Recommendation']).head(15))

df.to_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv", index=False)
