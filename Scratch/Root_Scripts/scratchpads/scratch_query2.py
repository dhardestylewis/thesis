import pandas as pd
z = pd.read_csv('Data/final/model_ready_zoning_data.csv', low_memory=False)
z['app_date'] = pd.to_datetime(z['application_start_date'], errors='coerce')
z['tcad'] = z['tcad_id'].astype(str)
z = z.dropna(subset=['tcad', 'app_date']).sort_values('app_date')
bw = pd.read_csv('Scratch/Modeling/Causal_Inference/05_G_Computation_LSTMs/biweekly_panel.csv', low_memory=False)
cases = bw.groupby('case_number').agg(has_petition=('petition_event', 'max')).reset_index()
z = z.merge(cases, on='case_number', how='left').fillna({'has_petition':0})
z['is_withdrawn'] = z['detailed_status'].str.contains('Withdraw|Void', case=False, na=False).astype(int)
withdrawn = z[(z['is_withdrawn']==1) & (z['has_petition']==1)]
print("--- PETITIONED WITHDRAWN -> REAPPEARANCE ZONING ---")
for _, w in withdrawn.iterrows():
    subs = z[(z['tcad']==w['tcad']) & (z['app_date'] > w['app_date'])]
    if len(subs) > 0:
        print(f"[{w['application_start_date']}] {w['Requested_Zoning']}  --->  [{subs.iloc[0]['application_start_date']}] {subs.iloc[0]['Requested_Zoning']}")
