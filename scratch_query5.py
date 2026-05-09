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

print("=== PETITIONED REAPPEARANCES (HEIGHT & FAR) ===")
for _, w in withdrawn[withdrawn['has_petition']==1].iterrows():
    subs = z[(z['tcad']==w['tcad']) & (z['app_date'] > w['app_date'])]
    if len(subs) > 0:
        s = subs.iloc[0]
        h1 = w['Requested_max_height_ft']
        h2 = s['Requested_max_height_ft']
        f1 = w['Requested_max_far']
        f2 = s['Requested_max_far']
        print(f"[{w['application_start_date']}] {w['Requested_Zoning']} (Ht: {h1}, FAR: {f1})  --->  [{s['application_start_date']}] {s['Requested_Zoning']} (Ht: {h2}, FAR: {f2})")

print("\n=== UNPETITIONED REAPPEARANCES WITH HEIGHT CHANGES (SAMPLE) ===")
unpet_changes = []
for _, w in withdrawn[withdrawn['has_petition']==0].iterrows():
    subs = z[(z['tcad']==w['tcad']) & (z['app_date'] > w['app_date'])]
    if len(subs) > 0:
        s = subs.iloc[0]
        h1 = w['Requested_max_height_ft']
        h2 = s['Requested_max_height_ft']
        if pd.notna(h1) and pd.notna(h2) and h1 != h2:
            unpet_changes.append(f"[{w['application_start_date']}] Ht: {h1} ---> [{s['application_start_date']}] Ht: {h2}")

import random
random.seed(42)
for u in random.sample(unpet_changes, min(10, len(unpet_changes))):
    print(u)
