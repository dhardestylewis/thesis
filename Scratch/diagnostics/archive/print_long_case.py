import pandas as pd
import json

df = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv')

longest_case = None
longest_trajectory = []
max_events = 0

print("Scanning for protracted cases...")
for idx, row in df.iterrows():
    traj_str = str(row['Zoning_Trajectory'])
    if traj_str != 'nan' and traj_str != 'None':
        try:
            trajectory = json.loads(traj_str)
            if len(trajectory) > max_events:
                # Let's verify it actually has different requested zones or spans time
                requested_zones = set([e.get('requested_zoning') for e in trajectory if e.get('requested_zoning')])
                if len(requested_zones) > 1 and len(trajectory) > 2:
                    max_events = len(trajectory)
                    longest_case = row['Core_Case']
                    longest_trajectory = trajectory
        except:
            pass

print(f"\nFound protracted case: {longest_case}")
print(f"Total Trajectory Events: {len(longest_trajectory)}")
print(json.dumps(longest_trajectory, indent=2))
