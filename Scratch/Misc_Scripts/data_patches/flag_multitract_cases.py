import pandas as pd
import json

print("Loading data...", flush=True)
model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df = pd.read_csv(model_csv)

def is_single_tract(case_num):
    case = str(case_num).upper()
    if case == 'NAN': return False
    
    if case.startswith('NPA'): return False
    if case.startswith('C814'): return False
    if case.startswith('C12M'): return False
    if '.' in case: return False
    
    return True

df['Is_Single_Tract'] = df['Core_Case'].apply(is_single_tract)

for idx, row in df.iterrows():
    if not row['Is_Single_Tract']:
        df.at[idx, 'Zoning_Trajectory'] = None
        df.at[idx, 'Initial_Zoning'] = None
        df.at[idx, 'Requested_Zoning'] = None
        df.at[idx, 'Final_Zoning'] = None

print("\nFiltering Results:")
print(f"Total Cases: {len(df)}")
print(f"Single-Tract Cases: {df['Is_Single_Tract'].sum()}")
print(f"Multi-Tract/Macro Cases Excluded: {(~df['Is_Single_Tract']).sum()}")
print(f"Pristine Trajectories remaining: {df[df['Is_Single_Tract']]['Zoning_Trajectory'].notna().sum()}")

df.to_csv(model_csv, index=False)
print("Master dataset updated with Is_Single_Tract flag.")
