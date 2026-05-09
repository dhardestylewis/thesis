import pandas as pd
z = pd.read_csv('Data/final/model_ready_zoning_data.csv', low_memory=False)
z['is_withdrawn'] = z['detailed_status'].str.contains('Withdraw|Void', case=False, na=False).astype(int)
z['app_date'] = pd.to_datetime(z['application_start_date'], errors='coerce')
z['tcad'] = z['tcad_id'].astype(str)
z = z.dropna(subset=['tcad', 'app_date']).sort_values('app_date')
bw = pd.read_csv('Scratch/Modeling/Causal_Inference/05_G_Computation_LSTMs/biweekly_panel.csv', low_memory=False)
cases = bw.groupby('case_number').agg(has_petition=('petition_event', 'max')).reset_index()
z = z.merge(cases, on='case_number', how='left').fillna({'has_petition':0})

withdrawn = z[z['is_withdrawn']==1]
results = []
for _, w in withdrawn.iterrows():
    subs = z[(z['tcad']==w['tcad']) & (z['app_date'] > w['app_date'])]
    if len(subs) > 0:
        s = subs.iloc[0]
        h1 = w['Requested_max_height_ft']
        h2 = s['Requested_max_height_ft']
        if pd.notna(h1) and pd.notna(h2):
            results.append({'has_pet': w['has_petition'], 'h1': h1, 'h2': h2, 'delta': h2-h1})

res = pd.DataFrame(results)
print('--- REAPPEARANCE HEIGHT DELTA ---')
print(res.groupby('has_pet')['delta'].mean().round(2))
print('--- PERCENT WITH HEIGHT REDUCTIONS ---')
print(res.groupby('has_pet').apply(lambda x: (x['delta'] < 0).mean().round(2), include_groups=False))
