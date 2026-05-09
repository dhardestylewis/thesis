import pandas as pd
import geopandas as gpd
import re
import numpy as np
import json

print("1. Loading Master Dataset...")
df = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv")

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
    'SF':   {'max_height_ft': 35, 'max_far': 0.40, 'max_bldg_cov_pct': 40, 'min_lot_sqft': 5750},
    'MH':   {'max_height_ft': 35, 'max_far': 0.50, 'max_bldg_cov_pct': 50, 'min_lot_sqft': 2500},
    'MF-1': {'max_height_ft': 40, 'max_far': 0.50, 'max_bldg_cov_pct': 45, 'min_lot_sqft': 8000},
    'MF-2': {'max_height_ft': 40, 'max_far': 0.60, 'max_bldg_cov_pct': 50, 'min_lot_sqft': 8000},
    'MF-3': {'max_height_ft': 40, 'max_far': 0.75, 'max_bldg_cov_pct': 55, 'min_lot_sqft': 8000},
    'MF-4': {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 60, 'min_lot_sqft': 8000},
    'MF-5': {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 70, 'min_lot_sqft': 8000},
    'MF-6': {'max_height_ft': 90, 'max_far': 3.00, 'max_bldg_cov_pct': 80, 'min_lot_sqft': 8000},
    'MF':   {'max_height_ft': 60, 'max_far': 1.00, 'max_bldg_cov_pct': 60, 'min_lot_sqft': 8000},
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

print("2. Parsing Base Zoning Constraints...")
for metric in ['max_height_ft', 'max_far', 'max_bldg_cov_pct', 'min_lot_sqft']:
    df[f'Initial_{metric}'] = np.nan
    df[f'Requested_{metric}'] = np.nan
    df[f'Approved_{metric}'] = np.nan

df['Staff_Recommendation'] = np.nan

for idx, row in df.iterrows():
    # Staff parsing
    traj_str = row.get('Zoning_Trajectory')
    if pd.notna(traj_str) and str(traj_str).strip() != 'nan':
        try:
            trajectory = json.loads(traj_str)
            staff_rec = None
            for event in trajectory:
                if 'staff_recommendation' in event and event['staff_recommendation']:
                    staff_rec = event['staff_recommendation']
            if staff_rec:
                df.at[idx, 'Staff_Recommendation'] = staff_rec
        except:
            pass

    # Zoning Metrics
    init_z = extract_base_zone(row.get('Initial_Zoning'))
    req_z = extract_base_zone(row.get('Requested_Zoning'))
    app_z = extract_base_zone(row.get('Final_Zoning'))
    
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
        
for metric in ['max_height_ft', 'max_far', 'max_bldg_cov_pct', 'min_lot_sqft']:
    df[f'Staff_{metric}'] = np.nan
    
for idx, row in df.iterrows():
    staff_z = extract_base_zone(row['Staff_Recommendation'])
    if staff_z and staff_z in zoning_metrics_dict:
        df.at[idx, 'Staff_max_height_ft'] = zoning_metrics_dict[staff_z]['max_height_ft']
        df.at[idx, 'Staff_max_far'] = zoning_metrics_dict[staff_z]['max_far']
        df.at[idx, 'Staff_max_bldg_cov_pct'] = zoning_metrics_dict[staff_z]['max_bldg_cov_pct']
        df.at[idx, 'Staff_min_lot_sqft'] = zoning_metrics_dict[staff_z]['min_lot_sqft']

print("3. Executing Spatial Proximity Engine...")
RAW_GEOJSON = r"c:\Users\dhl\data\Thesis\thesis\Data\CoA_Open_Data\Zoning_Cases_Raw_Download.geojson"
LDB_CSV = r"c:\Users\dhl\data\Thesis\thesis\Data\CoA_Open_Data\LDB_2021_kk8y-6cmt.csv"

gdf_cases = gpd.read_file(RAW_GEOJSON)[['case_number', 'geometry']]
gdf_cases['case_number'] = gdf_cases['case_number'].str.strip()
gdf_cases = gdf_cases.to_crs(epsg=2277)

print("   Loading Land Database (LDB) for Single-Family proximities...")
df_ldb = pd.read_csv(LDB_CSV, usecols=['the_geom', 'LU_DESC', 'GEN_LU_DES'])

def parse_geom(geom_str):
    if pd.isna(geom_str): return None
    try:
        from shapely import wkt
        return wkt.loads(geom_str)
    except:
        return None

df_ldb['geometry'] = df_ldb['the_geom'].apply(parse_geom)
gdf_ldb = gpd.GeoDataFrame(df_ldb, geometry='geometry', crs="EPSG:4326")
gdf_ldb = gdf_ldb.to_crs(epsg=2277)

gdf_sf = gdf_ldb[gdf_ldb['GEN_LU_DES'].isin(['Single Family'])]

nearest = gpd.sjoin_nearest(gdf_cases, gdf_sf[['geometry']], distance_col="SF_Distance_ft", how='left')
min_dists = nearest.groupby('case_number')['SF_Distance_ft'].min().reset_index()

df = pd.merge(df, min_dists, on='case_number', how='left')

def get_comp_cap(dist_ft):
    if pd.isna(dist_ft) or dist_ft > 540: return 9999.0
    if dist_ft <= 50: return 30.0
    if dist_ft <= 100: return 40.0
    return 40.0 + ((dist_ft - 100) / 10.0)

df['GIS_Compatibility_Height_Cap'] = df['SF_Distance_ft'].apply(get_comp_cap)

print("4. Calculating Volumetric and Developer Metrics...")
# Effective Heights
def calc_effective(row):
    zoned_height = row['Approved_max_height_ft']
    comp_cap = row['GIS_Compatibility_Height_Cap']
    if pd.isna(zoned_height): return np.nan
    return min(zoned_height, comp_cap)

df['Effective_Approved_Height'] = df.apply(calc_effective, axis=1)
df.loc[df['Effective_Approved_Height'] > 9000, 'Effective_Approved_Height'] = np.nan

df['Staff_Effective_Height'] = df['Staff_max_height_ft'].fillna(df['Requested_max_height_ft'])
df['Staff_Effective_Height'] = df.apply(lambda r: min(r['Staff_Effective_Height'], r.get('GIS_Compatibility_Height_Cap', 9999.0)) if pd.notna(r['Staff_Effective_Height']) else np.nan, axis=1)

# Height Attrition
df['Requested_Effective_Height'] = df['Requested_max_height_ft'].fillna(df['Initial_max_height_ft'])
df['Requested_Effective_Height'] = df.apply(lambda r: min(r['Requested_Effective_Height'], r.get('GIS_Compatibility_Height_Cap', 9999.0)) if pd.notna(r['Requested_Effective_Height']) else np.nan, axis=1)

df['Delta_Requested_Height'] = df['Requested_Effective_Height'] - df['Initial_max_height_ft']
df['Delta_Approved_Height'] = df['Effective_Approved_Height'] - df['Initial_max_height_ft']
df['Height_Attrition'] = df['Delta_Requested_Height'] - df['Delta_Approved_Height']

df['Staff_Attrition_Height'] = df['Requested_Effective_Height'] - df['Staff_Effective_Height']
df['Council_Attrition_Height'] = df['Staff_Effective_Height'] - df['Effective_Approved_Height']
df['Friction_Staff_Height'] = df['Staff_Effective_Height'] - df['Requested_Effective_Height']
df['Friction_Council_Height'] = df['Effective_Approved_Height'] - df['Staff_Effective_Height']

# FAR / Volume Attrition
df['Requested_Max_SqFt'] = df['shape_area'] * df['Requested_max_far']
df['Approved_Max_SqFt'] = df['shape_area'] * df['Approved_max_far']
df['SqFt_Attrition_Volume'] = df['Requested_Max_SqFt'] - df['Approved_Max_SqFt']

df['Phase_Requested_SqFt'] = df['shape_area'] * df['Requested_max_far']
df['Phase_Staff_SqFt'] = df['shape_area'] * df['Staff_max_far'].fillna(df['Requested_max_far'])
df['Phase_Approved_SqFt'] = df['shape_area'] * df['Approved_max_far']

df['Friction_Staff_SqFt'] = df['Phase_Staff_SqFt'] - df['Phase_Requested_SqFt']
df['Friction_Council_SqFt'] = df['Phase_Approved_SqFt'] - df['Phase_Staff_SqFt']

# Unit Developer Metric
def calculate_units(row, phase):
    zoning_col = f'{phase}_Zoning'
    zoning = str(row.get(zoning_col, '')).upper()
    sqft_col = f'{phase}_Max_SqFt'
    lot_col = f'{phase}_min_lot_sqft'
    area = row.get('shape_area', 0)
    
    if pd.isna(area) or area == 0:
        return np.nan
        
    if any(zoning.startswith(prefix) for prefix in ['SF', 'RR', 'LA', 'DR']):
        min_lot = row.get(lot_col)
        if pd.isna(min_lot) or min_lot == 0:
            return np.nan
        return area / min_lot
    else:
        max_sqft = row.get(sqft_col)
        if pd.isna(max_sqft) or max_sqft == 0:
            return np.nan
        return max_sqft / 1000.0

df['Developer_Requested_Units'] = df.apply(lambda row: calculate_units(row, 'Requested'), axis=1)
df['Developer_Approved_Units'] = df.apply(lambda row: calculate_units(row, 'Approved'), axis=1)
df['Unit_Yield_Attrition'] = df['Developer_Requested_Units'] - df['Developer_Approved_Units']

print("5. Overwriting final file...")
df.to_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv", index=False)
df.to_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv", index=False)
print("Pipeline Successfully Completed!")
