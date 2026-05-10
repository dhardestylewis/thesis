import pandas as pd
import json
import re

df = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv')

suspicious_cases = []

# Define basic zone ranking for density (rough approximation for QA)
# Lower index = lower density
density_rank = ['P', 'AG', 'RR', 'DR', 'SF', 'SF-1', 'SF-2', 'SF-3', 'SF-4A', 'SF-4B', 'SF-5', 'SF-6', 'MF-1', 'MF-2', 'MF-3', 'MF-4', 'MF-5', 'MF-6', 'NO', 'LO', 'GO', 'LR', 'GR', 'CS', 'CS-1', 'CH', 'IP', 'MI', 'LI', 'CBD', 'DMU', 'ERC', 'TOD']

def get_base_zone(z):
    if not z: return None
    # Strip modifiers like -CO, -NP, -MU, -V, -PDA
    base = re.sub(r'-(?:CO|NP|MU|V|PDA|H).*', '', str(z))
    return base

for idx, row in df.iterrows():
    traj_str = str(row['Zoning_Trajectory'])
    if traj_str != 'nan' and traj_str != 'None':
        try:
            trajectory = json.loads(traj_str)
            if not trajectory: continue
            
            case = row['case_number']
            reason = None
            
            # Anomaly 1: Extremely long trajectory for a single parcel
            if len(trajectory) > 6:
                reason = "Hyper-active trajectory (>6 events)"
                
            else:
                for event in trajectory:
                    ext = event.get('existing_zoning')
                    req = event.get('requested_zoning')
                    stf = event.get('staff_recommendation')
                    app = event.get('approved_zoning')
                    
                    b_ext = get_base_zone(ext)
                    b_req = get_base_zone(req)
                    b_stf = get_base_zone(stf)
                    b_app = get_base_zone(app)
                    
                    # Anomaly 2: Staff Recommending HIGHER density than Developer requested
                    if b_req and b_stf and b_req in density_rank and b_stf in density_rank:
                        if density_rank.index(b_stf) > density_rank.index(b_req):
                            # Allow if they are just different classes (e.g. SF to LO)
                            # But if it's MF-2 to MF-4, that's weird
                            reason = f"Staff Rec ({stf}) higher density than Request ({req})"
                            break
                            
            if reason:
                suspicious_cases.append({
                    "case": case,
                    "reason": reason,
                    "trajectory": trajectory
                })
        except Exception as e:
            pass

print(f"Total Suspicious Cases Found: {len(suspicious_cases)}")
for s in suspicious_cases[:5]:
    print(f"\n--- {s['case']} [{s['reason']}] ---")
    print(json.dumps(s['trajectory'], indent=2))
