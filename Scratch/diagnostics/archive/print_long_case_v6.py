import pandas as pd
import json

df = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv')

ideal_case = None
ideal_traj = []

for idx, row in df.iterrows():
    case = str(row['case_number']).upper()
    if not case.startswith('C14-'): continue
    
    traj_str = str(row['Zoning_Trajectory'])
    if traj_str != 'nan' and traj_str != 'None':
        try:
            trajectory = json.loads(traj_str)
            phases = set([e.get('phase') for e in trajectory if e.get('phase')])
            
            zones = set([e.get('requested_zoning') for e in trajectory if e.get('requested_zoning')])
            zones.update([e.get('approved_zoning') for e in trajectory if e.get('approved_zoning')])
            
            if 'Commission' in phases and 'Council' in phases and len(zones) > 1:
                # We want a case with multiple steps in Commission and a final Council vote
                comm_events = [e for e in trajectory if e.get('phase') == 'Commission']
                if len(comm_events) >= 2 and len(trajectory) < 8:
                    ideal_case = case
                    ideal_traj = trajectory
                    break
        except:
            pass

print(f"Case: {ideal_case}")
print(json.dumps(ideal_traj, indent=2))
