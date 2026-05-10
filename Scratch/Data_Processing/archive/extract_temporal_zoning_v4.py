import pandas as pd
import re
import numpy as np
import json

print("Loading data...", flush=True)
model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df_model = pd.read_csv(model_csv)
df_comm = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv")
df_council = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\zoning_cases_with_council_votes.csv")

valid_cases = set(df_model['case_number'].dropna().astype(str).str.upper().unique())
case_pattern = re.compile(r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4}(?:\.[A-Z0-9]+)?)')

base_zones = r'(?:SF|MF|CS|GR|LO|GO|CH|LI|MI|DR|AG|P|RR|CBD|DMU|TOD|PUD|ERC|W|NO|IP|CR)'
suffix = r'(?:-[A-Z0-9]+)'
optional_number = r'(?:[0-9]+[A-Z]*)'
zone_regex = r'\b' + base_zones + r'(?:' + optional_number + r')?' + r'(?:' + suffix + r'){0,4}\b'

pattern_req_to = re.compile(r'(?i)request.{0,40}?(' + zone_regex + r').{0,30}?\bto\b.{0,30}?(' + zone_regex + r')')
pattern_from_to = re.compile(r'(?i)from.{0,40}?(' + zone_regex + r').{0,30}?\bto\b.{0,30}?(' + zone_regex + r')')
pattern_proposed = re.compile(r'(?i)(?:proposed|request|change).{0,50}?(' + zone_regex + r')')
pattern_staff = re.compile(r'(?i)staff\s*rec.*?(?:recommendation of |for |to )?(' + zone_regex + r')')
pattern_all = re.compile(r'(' + zone_regex + r')')

trajectory_dict = {case: [] for case in valid_cases}

def clean_date(d_str):
    try:
        return pd.to_datetime(d_str).strftime('%Y-%m-%d')
    except:
        return None

def process_transcript(df, phase, text_col, date_col):
    for i, row in df.iterrows():
        text = row[text_col]
        if pd.isna(text): continue
        
        date_str = str(row.get(date_col, 'Unknown Date'))
        date_val = clean_date(date_str) if date_str != 'Unknown Date' else None
        
        text_str = str(text).upper()
        for m in case_pattern.finditer(text_str):
            case = m.group(1)
            if case in valid_cases:
                idx = m.start()
                start = max(0, idx - 50)
                end = min(len(text_str), idx + 800)
                window = text_str[start:end]
                
                existing = None
                requested = None
                staff = None
                
                match_req = pattern_req_to.search(window)
                if match_req:
                    existing = match_req.group(1).upper()
                    requested = match_req.group(2).upper()
                else:
                    match_from = pattern_from_to.search(window)
                    if match_from:
                        existing = match_from.group(1).upper()
                        requested = match_from.group(2).upper()
                    else:
                        match_prop = pattern_proposed.search(window)
                        if match_prop:
                            requested = match_prop.group(1).upper()
                
                match_staff = pattern_staff.search(window)
                if match_staff:
                    staff = match_staff.group(1).upper()
                    if staff == 'NO':
                        staff = None
                    
                if requested or staff or existing:
                    event = {"phase": phase, "date": date_val}
                    if existing: event["existing_zoning"] = existing
                    if requested: event["requested_zoning"] = requested
                    if staff: event["staff_recommendation"] = staff
                    trajectory_dict[case].append(event)
                    
                if phase == "Council":
                    matches = pattern_all.findall(window)
                    if matches:
                        final_zoning = matches[-1]
                        if final_zoning == 'NO' and 'NEIGHBORHOOD OFFICE' not in window:
                            pass 
                        else:
                            event = {"phase": "Council", "date": date_val, "approved_zoning": final_zoning}
                            trajectory_dict[case].append(event)

print("Scanning Commission Transcripts...", flush=True)
process_transcript(df_comm, "Commission", "Raw_Text", "Date")

print("Scanning Council Transcripts...", flush=True)
process_transcript(df_council, "Council", "Vote_Transcript", "Meeting_Date")


print("Building temporal trajectories...", flush=True)
df_model['Zoning_Trajectory'] = None
df_model['Initial_Zoning'] = None
df_model['Requested_Zoning'] = None
df_model['Final_Zoning'] = None

for idx, row in df_model.iterrows():
    case = str(row['case_number']).upper()
    events = trajectory_dict.get(case, [])
    
    if len(events) > 0:
        collapsed_events = []
        
        grouped = {}
        for e in events:
            key = (e.get('phase'), e.get('date'))
            if key not in grouped:
                grouped[key] = {"phase": e.get('phase'), "date": e.get('date')}
            
            if e.get('existing_zoning'): grouped[key]["existing_zoning"] = e.get('existing_zoning')
            if e.get('requested_zoning'): grouped[key]["requested_zoning"] = e.get('requested_zoning')
            if e.get('staff_recommendation'): grouped[key]["staff_recommendation"] = e.get('staff_recommendation')
            if e.get('approved_zoning'): grouped[key]["approved_zoning"] = e.get('approved_zoning')
            
        seen_keys = set()
        for e in events:
            key = (e.get('phase'), e.get('date'))
            if key not in seen_keys:
                collapsed_events.append(grouped[key])
                seen_keys.add(key)
                
        final_events = []
        for e in collapsed_events:
            has_zoning = any(k in e for k in ['existing_zoning', 'requested_zoning', 'staff_recommendation', 'approved_zoning'])
            if has_zoning:
                final_events.append(e)

        if final_events:
            df_model.at[idx, 'Zoning_Trajectory'] = json.dumps(final_events)
            
            existings = [e.get('existing_zoning') for e in final_events if e.get('existing_zoning')]
            reqs = [e.get('requested_zoning') for e in final_events if e.get('requested_zoning')]
            finals = [e.get('approved_zoning') for e in final_events if e.get('approved_zoning')]
            
            if existings: df_model.at[idx, 'Initial_Zoning'] = existings[0]
            if reqs: df_model.at[idx, 'Requested_Zoning'] = reqs[-1]
            if finals: df_model.at[idx, 'Final_Zoning'] = finals[-1]

print("\nExtraction Complete!", flush=True)
print(f"Cases with Trajectories found: {df_model['Zoning_Trajectory'].notna().sum()} / {len(df_model)}", flush=True)

df_model.to_csv(model_csv, index=False)
