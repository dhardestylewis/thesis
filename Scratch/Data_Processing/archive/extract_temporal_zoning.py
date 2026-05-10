import pandas as pd
import re
import numpy as np
import json

print("Loading data...", flush=True)
model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df_model = pd.read_csv(model_csv)
df_comm = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv")
df_council = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\zoning_cases_with_council_votes.csv")

valid_cases = set(df_model['Core_Case'].dropna().unique())

case_pattern = re.compile(r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4})')
base_zones = r'(?:SF|MF|CS|GR|LO|GO|CH|LI|MI|DR|AG|P|RR|CBD|DMU|TOD|PUD|ERC|W|NO|IP|CR)'
suffix = r'(?:-[A-Z0-9]+)'
optional_number = r'(?:[0-9]+[A-Z]*)'
zone_regex = r'\b' + base_zones + r'(?:' + optional_number + r')?' + r'(?:' + suffix + r'){0,4}\b'

pattern_req_to = re.compile(r'(?i)request.{0,40}?(' + zone_regex + r').{0,30}?\bto\b.{0,30}?(' + zone_regex + r')')
pattern_from_to = re.compile(r'(?i)from.{0,40}?(' + zone_regex + r').{0,30}?\bto\b.{0,30}?(' + zone_regex + r')')
pattern_proposed = re.compile(r'(?i)(?:proposed|request|change).{0,50}?(' + zone_regex + r')')
pattern_all = re.compile(r'(' + zone_regex + r')')

trajectory_dict = {case: [] for case in valid_cases}

def clean_date(d_str):
    try:
        return pd.to_datetime(d_str).strftime('%Y-%m-%d')
    except:
        return None

print("Scanning Commission Transcripts...", flush=True)
for i, row in df_comm.iterrows():
    text = row['Raw_Text']
    if pd.isna(text): continue
    
    date_str = str(row.get('Date', 'Unknown Date'))
    date_val = clean_date(date_str) if date_str != 'Unknown Date' else None
    
    text_str = str(text).upper()
    for m in case_pattern.finditer(text_str):
        case = m.group(1)
        if case in valid_cases:
            idx = m.start()
            start = max(0, idx - 50)
            end = min(len(text_str), idx + 800)
            window = text_str[start:end]
            
            req_zoning = None
            
            match = pattern_req_to.search(window)
            if match:
                req_zoning = match.group(2).upper()
            else:
                match = pattern_from_to.search(window)
                if match:
                    req_zoning = match.group(2).upper()
                else:
                    match_prop = pattern_proposed.search(window)
                    if match_prop:
                        req_zoning = match_prop.group(1).upper()
            
            if req_zoning:
                event = {
                    "phase": "Commission",
                    "date": date_val,
                    "requested_zoning": req_zoning
                }
                trajectory_dict[case].append(event)


print("Scanning Council Transcripts...", flush=True)
for i, row in df_council.iterrows():
    text = row['Vote_Transcript']
    if pd.isna(text): continue
    
    date_str = str(row.get('Meeting_Date', 'Unknown Date'))
    date_val = clean_date(date_str) if date_str != 'Unknown Date' else None
    
    text_str = str(text).upper()
    case = str(row['Case_Number']).upper()
    
    if case in valid_cases:
        req_zoning = None
        match = pattern_from_to.search(text_str)
        if match:
            req_zoning = match.group(2).upper()
        else:
            match_req = pattern_req_to.search(text_str)
            if match_req:
                req_zoning = match_req.group(2).upper()
            else:
                match_prop = pattern_proposed.search(text_str)
                if match_prop:
                    req_zoning = match_prop.group(1).upper()
                    
        if req_zoning:
            event = {
                "phase": "Council",
                "date": date_val,
                "requested_zoning": req_zoning
            }
            trajectory_dict[case].append(event)
            
        matches = pattern_all.findall(text_str)
        if matches:
            final_zoning = matches[-1]
            event = {
                "phase": "Council",
                "date": date_val,
                "approved_zoning": final_zoning
            }
            trajectory_dict[case].append(event)


print("Building temporal trajectories...", flush=True)
df_model['Zoning_Trajectory'] = None

for idx, row in df_model.iterrows():
    case = str(row['Core_Case'])
    events = trajectory_dict.get(case, [])
    
    if len(events) > 0:
        clean_events = []
        for e in events:
            if not clean_events:
                clean_events.append(e)
            else:
                prev = clean_events[-1]
                if e.get('requested_zoning') and prev.get('requested_zoning') == e.get('requested_zoning') and e.get('phase') == prev.get('phase'):
                    continue
                if e.get('approved_zoning') and prev.get('approved_zoning') == e.get('approved_zoning') and e.get('phase') == prev.get('phase'):
                    continue
                clean_events.append(e)
                
        df_model.at[idx, 'Zoning_Trajectory'] = json.dumps(clean_events)

print("\nExtraction Complete!", flush=True)
print(f"Cases with Trajectories found: {df_model['Zoning_Trajectory'].notna().sum()} / {len(df_model)}", flush=True)

df_model.to_csv(model_csv, index=False)
