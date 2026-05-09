import pandas as pd
z = pd.read_csv('Data/final/model_ready_zoning_data.csv', low_memory=False)
z['app_date'] = pd.to_datetime(z['application_start_date'], errors='coerce')
z['tcad'] = z['tcad_id'].astype(str)
z = z.dropna(subset=['tcad', 'app_date']).sort_values('app_date')
bw = pd.read_csv('Scratch/Modeling/Causal_Inference/05_G_Computation_LSTMs/biweekly_panel.csv', low_memory=False)
cases = bw.groupby('case_number').agg(has_petition=('petition_event', 'max')).reset_index()
z = z.merge(cases, on='case_number', how='left').fillna({'has_petition':0})
z['is_withdrawn'] = z['detailed_status'].str.contains('Withdraw|Void', case=False, na=False).astype(int)

withdrawn = z[z['is_withdrawn']==1]
pet_changes = []
unpet_changes = []

for _, w in withdrawn.iterrows():
    subs = z[(z['tcad']==w['tcad']) & (z['app_date'] > w['app_date'])]
    if len(subs) > 0:
        s = subs.iloc[0]
        z1 = str(w['Requested_Zoning']).strip()
        z2 = str(s['Requested_Zoning']).strip()
        if z1 != 'nan' and z2 != 'nan' and z1 != z2:
            if w['has_petition'] == 1:
                pet_changes.append(f"{z1}  --->  {z2}")
            else:
                unpet_changes.append(f"{z1}  --->  {z2}")

print("=== PETITIONED ZONING CHANGES (ALL 4) ===")
for p in pet_changes:
    print(p)

print("\n=== UNPETITIONED ZONING CHANGES (SAMPLE OF 10) ===")
import random
random.seed(42)
for u in random.sample(unpet_changes, min(10, len(unpet_changes))):
    print(u)
