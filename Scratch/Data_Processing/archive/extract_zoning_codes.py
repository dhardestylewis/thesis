import pandas as pd
import re
import numpy as np

print("Loading data...")
model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df_model = pd.read_csv(model_csv)
df_comm = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv")
df_council = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\zoning_cases_with_council_votes.csv")

def extract_zones_robust(case_num, text_series):
    for text in text_series:
        if pd.isna(text): continue
        
        for m in re.finditer(re.escape(case_num), str(text).upper()):
            idx = m.start()
            start = max(0, idx - 100)
            end = min(len(text), idx + 600)
            window = text[start:end]
            
            pattern = re.compile(r'(?i)from\s+(.{0,60}?)\b((?:SF|MF|CS|GR|LO|GO|CH|LI|MI|DR|AG|P|RR|CBD|DMU|TOD|PUD|ERC|W|NO|IP|CR)(?:-[0-9A-Z]+)*)\b(.{0,20}?)\bto\b(.{0,60}?)\b((?:SF|MF|CS|GR|LO|GO|CH|LI|MI|DR|AG|P|RR|CBD|DMU|TOD|PUD|ERC|W|NO|IP|CR)(?:-[0-9A-Z]+)*)\b')
            match = pattern.search(window)
            if match:
                return match.group(2).upper(), match.group(5).upper()
                
            pattern_to = re.compile(r'(?i)(?:change|request|proposed|reclassify).{0,60}?\b((?:SF|MF|CS|GR|LO|GO|CH|LI|MI|DR|AG|P|RR|CBD|DMU|TOD|PUD|ERC|W|NO|IP|CR)(?:-[0-9A-Z]+)*)\b')
            match_to = pattern_to.search(window)
            if match_to:
                return "UNKNOWN", match_to.group(1).upper()
                
    return None, None

print("Extracting zoning codes from transcripts. This may take a moment...")

initial_zones = []
requested_zones = []
final_zones = []

for idx, row in df_model.iterrows():
    case = str(row['Core_Case'])
    if pd.isna(case) or case == 'NAN':
        initial_zones.append(None)
        requested_zones.append(None)
        final_zones.append(None)
        continue
        
    init_z, req_z = extract_zones_robust(case, df_comm['Raw_Text'])
    
    council_texts = df_council[df_council['Case_Number'].astype(str).str.contains(case, regex=False, na=False)]['Vote_Transcript']
    c_init, c_req = extract_zones_robust(case, council_texts)
    
    best_init = init_z if init_z else c_init
    best_req = req_z if req_z else c_req
    
    final_z = None
    if row.get('Derived_Status', '').startswith('Approved'):
        if c_req and c_req != 'UNKNOWN':
            final_z = c_req
        else:
            if len(council_texts) > 0:
                last_text = str(council_texts.iloc[-1]).upper()
                pattern_all = re.compile(r'\b((?:SF|MF|CS|GR|LO|GO|CH|LI|MI|DR|AG|P|RR|CBD|DMU|TOD|PUD|ERC|W|NO|IP|CR)(?:-[0-9A-Z]+)*)\b')
                matches = pattern_all.findall(last_text)
                if matches:
                    final_z = matches[-1]
                    
    initial_zones.append(best_init)
    requested_zones.append(best_req)
    final_zones.append(final_z)
    
    if idx % 500 == 0:
        print(f"Processed {idx} / {len(df_model)} cases")

df_model['Initial_Zoning'] = initial_zones
df_model['Requested_Zoning'] = requested_zones
df_model['Final_Zoning'] = final_zones

print("\nExtraction Complete!")
print(f"Cases with Requested Zoning found: {df_model['Requested_Zoning'].notna().sum()} / {len(df_model)}")

df_model.to_csv(model_csv, index=False)
