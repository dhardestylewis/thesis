import pandas as pd
import re

# To catalogue OCR errors, we will scan the Raw_Text of the 135 suspicious cases
# and look for zoning codes with spaces in them or weird suffixes.
df = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv')
df_comm = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv')

# Load the list of 135 suspicious cases we just generated (we can approximate by re-running the logic)
suspicious_cases = []
density_rank = ['P', 'AG', 'RR', 'DR', 'SF', 'SF-1', 'SF-2', 'SF-3', 'SF-4A', 'SF-4B', 'SF-5', 'SF-6', 'MF-1', 'MF-2', 'MF-3', 'MF-4', 'MF-5', 'MF-6', 'NO', 'LO', 'GO', 'LR', 'GR', 'CS', 'CS-1', 'CH', 'IP', 'MI', 'LI', 'CBD', 'DMU', 'ERC', 'TOD']

def get_base_zone(z):
    if not z: return None
    base = re.sub(r'-(?:CO|NP|MU|V|PDA|H).*', '', str(z))
    return base

import json
for idx, row in df.iterrows():
    traj_str = str(row['Zoning_Trajectory'])
    if traj_str != 'nan' and traj_str != 'None':
        try:
            trajectory = json.loads(traj_str)
            for event in trajectory:
                req = event.get('requested_zoning')
                stf = event.get('staff_recommendation')
                b_req = get_base_zone(req)
                b_stf = get_base_zone(stf)
                if b_req and b_stf and b_req in density_rank and b_stf in density_rank:
                    if density_rank.index(b_stf) > density_rank.index(b_req):
                        suspicious_cases.append(row['case_number'])
                        break
        except: pass

print(f"Found {len(suspicious_cases)} cases. Scanning for OCR spacing errors...")

base_zones = r'(?:SF|MF|CS|GR|LO|GO|CH|LI|MI|DR|AG|P|RR|CBD|DMU|TOD|PUD|ERC|W|NO|IP|CR)'
# Relaxed regex to find zoning codes WITH spaces or OCR artifacts
relaxed_regex = re.compile(r'\b' + base_zones + r'\s*-\s*[0-9]+[A-Z]*\s*(?:-\s*[A-Z0-9]+\s*){0,4}\b', re.IGNORECASE)

found_errors = set()
for case in suspicious_cases[:50]: # just sample 50
    texts = df_comm[df_comm['Raw_Text'].str.contains(str(case), case=False, na=False)]['Raw_Text'].values
    if len(texts) > 0:
        text = texts[0].upper()
        # Find all relaxed matches that have a space in them
        for m in relaxed_regex.finditer(text):
            val = m.group(0)
            if ' ' in val:
                found_errors.add(val.strip())

for e in list(found_errors)[:20]:
    print(f"OCR Error: '{e}'")
