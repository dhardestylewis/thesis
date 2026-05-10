import pandas as pd
import re
import numpy as np

print("Loading data...", flush=True)
model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df_model = pd.read_csv(model_csv)
df_comm = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv")
df_council = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\zoning_cases_with_council_votes.csv")

valid_cases = set(df_model['Core_Case'].dropna().unique())

case_pattern = re.compile(r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4})')
pattern_from_to = re.compile(r'(?i)from\s+(.{0,60}?)\b((?:SF|MF|CS|GR|LO|GO|CH|LI|MI|DR|AG|P|RR|CBD|DMU|TOD|PUD|ERC|W|NO|IP|CR)(?:-[0-9A-Z]+)*)\b(.{0,20}?)\bto\b(.{0,60}?)\b((?:SF|MF|CS|GR|LO|GO|CH|LI|MI|DR|AG|P|RR|CBD|DMU|TOD|PUD|ERC|W|NO|IP|CR)(?:-[0-9A-Z]+)*)\b')
pattern_to = re.compile(r'(?i)(?:change|request|proposed|reclassify).{0,60}?\b((?:SF|MF|CS|GR|LO|GO|CH|LI|MI|DR|AG|P|RR|CBD|DMU|TOD|PUD|ERC|W|NO|IP|CR)(?:-[0-9A-Z]+)*)\b')
pattern_all = re.compile(r'\b((?:SF|MF|CS|GR|LO|GO|CH|LI|MI|DR|AG|P|RR|CBD|DMU|TOD|PUD|ERC|W|NO|IP|CR)(?:-[0-9A-Z]+)*)\b')

comm_dict = {}
print("Scanning Commission Transcripts...", flush=True)
for i, text in enumerate(df_comm['Raw_Text']):
    if pd.isna(text): continue
    text_str = str(text).upper()
    for m in case_pattern.finditer(text_str):
        case = m.group(1)
        if case in valid_cases:
            if case not in comm_dict:
                idx = m.start()
                start = max(0, idx - 100)
                end = min(len(text_str), idx + 600)
                window = text_str[start:end]
                
                match = pattern_from_to.search(window)
                if match:
                    comm_dict[case] = (match.group(2).upper(), match.group(5).upper())
                else:
                    match_to = pattern_to.search(window)
                    if match_to:
                        comm_dict[case] = ("UNKNOWN", match_to.group(1).upper())

print(f"Found Commission Zoning for {len(comm_dict)} cases.", flush=True)

council_dict = {}
print("Scanning Council Transcripts...", flush=True)
for i, text in enumerate(df_council['Vote_Transcript']):
    if pd.isna(text): continue
    text_str = str(text).upper()
    case = str(df_council.iloc[i]['Case_Number']).upper()
    
    if case in valid_cases:
        if case not in council_dict:
            match = pattern_from_to.search(text_str)
            if match:
                council_dict[case] = (match.group(2).upper(), match.group(5).upper())
            else:
                match_to = pattern_to.search(text_str)
                if match_to:
                    council_dict[case] = ("UNKNOWN", match_to.group(1).upper())

print(f"Found Council Zoning for {len(council_dict)} cases.", flush=True)

final_zoning_dict = {}
for i, text in enumerate(df_council['Vote_Transcript']):
    if pd.isna(text): continue
    text_str = str(text).upper()
    case = str(df_council.iloc[i]['Case_Number']).upper()
    
    if case in valid_cases:
        matches = pattern_all.findall(text_str)
        if matches:
            final_zoning_dict[case] = matches[-1]

print("Applying to master dataset...", flush=True)
df_model['Initial_Zoning'] = None
df_model['Requested_Zoning'] = None
df_model['Final_Zoning'] = None

for idx, row in df_model.iterrows():
    case = str(row['Core_Case'])
    
    c_init, c_req = comm_dict.get(case, (None, None))
    cc_init, cc_req = council_dict.get(case, (None, None))
    
    best_init = c_init if c_init else cc_init
    best_req = c_req if c_req else cc_req
    
    final_z = None
    if row.get('Derived_Status', '').startswith('Approved'):
        if cc_req and cc_req != 'UNKNOWN':
            final_z = cc_req
        else:
            final_z = final_zoning_dict.get(case)
            
    df_model.at[idx, 'Initial_Zoning'] = best_init
    df_model.at[idx, 'Requested_Zoning'] = best_req
    df_model.at[idx, 'Final_Zoning'] = final_z

print("\nExtraction Complete!", flush=True)
print(f"Cases with Requested Zoning found: {df_model['Requested_Zoning'].notna().sum()} / {len(df_model)}", flush=True)
print(f"Cases with Final Zoning found: {df_model['Final_Zoning'].notna().sum()}", flush=True)

df_model.to_csv(model_csv, index=False)
