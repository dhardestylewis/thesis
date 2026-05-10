import pandas as pd
import json

df = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv')

longest_case = None
longest_trajectory = []
max_events = 0

for idx, row in df.iterrows():
    traj_str = str(row['Zoning_Trajectory'])
    if traj_str != 'nan' and traj_str != 'None':
        try:
            trajectory = json.loads(traj_str)
            # Filter to cases that have changing requested zones
            zones = set([e.get('requested_zoning') for e in trajectory if e.get('requested_zoning') and e.get('requested_zoning') != 'UNKNOWN'])
            
            if len(zones) >= 3:
                # Let's find one that looks reasonable
                if len(trajectory) < 15: # don't print a 50-event corrupted loop
                    longest_case = row['Core_Case']
                    longest_trajectory = trajectory
                    break
        except:
            pass

print(f"Case: {longest_case}")
print(json.dumps(longest_trajectory, indent=2))
