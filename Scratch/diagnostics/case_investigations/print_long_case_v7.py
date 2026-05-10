import pandas as pd
import json

df = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv')

case_row = df[df['case_number'] == 'C14-2010-0084']
if len(case_row) > 0:
    traj_str = str(case_row.iloc[0]['Zoning_Trajectory'])
    print(f"Case: C14-2010-0084")
    try:
        print(json.dumps(json.loads(traj_str), indent=2))
    except:
        print(traj_str)
