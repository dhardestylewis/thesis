import pandas as pd
import json

df = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv')

ideal_case = None
ideal_traj = []

for idx, row in df.iterrows():
    traj_str = str(row['Zoning_Trajectory'])
    if traj_str != 'nan' and traj_str != 'None':
        try:
            trajectory = json.loads(traj_str)
            phases = set([e.get('phase') for e in trajectory if e.get('phase')])
            
            # Extract all zoning strings to find variations
            zones = set([e.get('requested_zoning') for e in trajectory if e.get('requested_zoning')])
            zones.update([e.get('approved_zoning') for e in trajectory if e.get('approved_zoning')])
            
            # Look for a case that has Commission AND Council events, and changes zones at least once
            if 'Commission' in phases and 'Council' in phases and len(zones) > 1:
                # To find a really good one, let's look for one with > 2 events
                if len(trajectory) > 2 and len(trajectory) < 8:
                    ideal_case = row['case_number']
                    ideal_traj = trajectory
                    break
        except:
            pass

print(f"Case: {ideal_case}")
print(json.dumps(ideal_traj, indent=2))
