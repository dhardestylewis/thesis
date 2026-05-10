import pandas as pd
import re

z = pd.read_csv('Data/final/model_ready_zoning_data.csv', low_memory=False)
z['app_date'] = pd.to_datetime(z['application_start_date'], errors='coerce')
z['tcad'] = z['tcad_id'].astype(str)
z = z.dropna(subset=['tcad', 'app_date']).sort_values('app_date')

bw = pd.read_csv('Scratch/Modeling/Causal_Inference/05_G_Computation_LSTMs/biweekly_panel.csv', low_memory=False)
cases = bw.groupby('case_number').agg(has_petition=('petition_event', 'max')).reset_index()
z = z.merge(cases, on='case_number', how='left').fillna({'has_petition':0})
z['is_withdrawn'] = z['detailed_status'].str.contains('Withdraw|Void', case=False, na=False).astype(int)

withdrawn = z[(z['is_withdrawn']==1) & (z['has_petition']==1)]
target_cases = []
for _, w in withdrawn.iterrows():
    subs = z[(z['tcad']==w['tcad']) & (z['app_date'] > w['app_date'])]
    if len(subs) > 0:
        s = subs.iloc[0]
        z1 = str(w['Requested_Zoning']).strip()
        z2 = str(s['Requested_Zoning']).strip()
        if z1 != 'nan' and z2 != 'nan' and z1 != z2:
            target_cases.append(w['case_number'])

print(f"Target Cases to Investigate: {target_cases}")

df_comm = pd.read_csv('Data/interim/commission_transcripts.csv')
df_council = pd.read_csv('Data/interim/zoning_cases_with_council_votes.csv')

def clean_case(x):
    return re.sub(r'(\.0[1-9]|\.[A-Z0-9]+)$', '', str(x).strip().upper()) if pd.notna(x) else x

df_comm['Core_Case'] = df_comm['Case_Number'].apply(clean_case) if 'Case_Number' in df_comm.columns else None
if 'Core_Case' not in df_comm.columns:
    df_comm['Core_Case'] = ''
df_council['Core_Case'] = df_council['Case_Number'].apply(clean_case)

for case in target_cases:
    case_clean = clean_case(case)
    print(f"\n======================================")
    print(f"CASE: {case}")
    print(f"======================================")
    
    # Council
    c_matches = df_council[df_council['Core_Case'] == case_clean]
    if len(c_matches) > 0:
        print(f"--- COUNCIL TRANSCRIPTS ---")
        for idx, row in c_matches.iterrows():
            text = str(row['Vote_Transcript'])
            print(f"Date: {row.get('Meeting_Date', 'Unknown')} | Text snippet: {text[:300]}...")
    else:
        print(f"--- NO COUNCIL TRANSCRIPTS FOUND ---")
        
    # Commission
    print(f"\n--- COMMISSION TRANSCRIPTS (Scanning by regex) ---")
    found_comm = False
    for idx, row in df_comm.iterrows():
        text = str(row.get('Raw_Text', ''))
        if case_clean in text:
            start = text.find(case_clean)
            print(f"Found in text: ...{text[max(0, start-100):min(len(text), start+200)]}...")
            found_comm = True
    if not found_comm:
        print("--- NO COMMISSION TRANSCRIPTS FOUND ---")
