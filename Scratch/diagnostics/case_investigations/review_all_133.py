import pandas as pd
import json

df = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv')

density_rank = ['P', 'AG', 'RR', 'DR', 'SF', 'SF-1', 'SF-2', 'SF-3', 'SF-4A', 'SF-4B', 'SF-5', 'SF-6', 'MF-1', 'MF-2', 'MF-3', 'MF-4', 'MF-5', 'MF-6', 'NO', 'LO', 'GO', 'LR', 'GR', 'CS', 'CS-1', 'CH', 'IP', 'MI', 'LI', 'CBD', 'DMU', 'ERC', 'TOD']

def get_base_zone(z):
    if not z: return None
    import re
    base = re.sub(r'-(?:CO|NP|MU|V|PDA|H).*', '', str(z))
    return base

suspicious = []
for idx, row in df.iterrows():
    traj_str = str(row['Zoning_Trajectory'])
    if traj_str != 'nan' and traj_str != 'None':
        try:
            trajectory = json.loads(traj_str)
            for event in trajectory:
                req = event.get('requested_zoning')
                stf = event.get('staff_recommendation')
                if req and stf:
                    b_req = get_base_zone(req)
                    b_stf = get_base_zone(stf)
                    if b_req in density_rank and b_stf in density_rank:
                        if density_rank.index(b_stf) > density_rank.index(b_req):
                            suspicious.append({
                                'case': row['case_number'],
                                'req': req,
                                'stf': stf
                            })
                            break
        except: pass

print(f"Total Suspicious: {len(suspicious)}")

# Find ones that have numbers longer than 2 digits or don't look like standard modifiers
weird_ocr = []
standard = []

import re
valid_modifier = re.compile(r'^([A-Z]+(?:-[A-Z0-9]+){0,4})$')

for s in suspicious:
    req = s['req']
    stf = s['stf']
    # If string is longer than 15 chars or contains a sequence of >2 digits, it's probably an OCR error
    if len(req) > 15 or len(stf) > 15 or re.search(r'\d{3,}', req) or re.search(r'\d{3,}', stf):
        weird_ocr.append(s)
    else:
        standard.append(s)

print(f"\n--- Probable OCR Errors ({len(weird_ocr)}) ---")
for w in weird_ocr:
    print(f"{w['case']}: Req={w['req']}, Stf={w['stf']}")

print(f"\n--- Standard but illogical density ({len(standard)}) ---")
for i, st in enumerate(standard):
    print(f"{st['case']}: Req={st['req']}, Stf={st['stf']}")
    if i > 10: 
        print("...")
        break
