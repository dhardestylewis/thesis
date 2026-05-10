import pandas as pd
import json

df = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv')

longest_case = None
longest_trajectory = []
max_dates = 0

for idx, row in df.iterrows():
    traj_str = str(row['Zoning_Trajectory'])
    if traj_str != 'nan':
        try:
            trajectory = json.loads(traj_str)
            dates = set([e.get('date') for e in trajectory if e.get('date')])
            # Filter to cases that have changing requested zones
            zones = set([e.get('requested_zoning') for e in trajectory if e.get('requested_zoning')])
            
            if len(dates) > max_dates and len(zones) > 1:
                max_dates = len(dates)
                longest_case = row['Core_Case']
                longest_trajectory = trajectory
        except:
            pass

print(f"Case: {longest_case}, Unique Dates: {max_dates}")
print(json.dumps(longest_trajectory, indent=2))
