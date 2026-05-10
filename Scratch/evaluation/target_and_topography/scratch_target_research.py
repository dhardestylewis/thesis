import pandas as pd
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), nrows=1000)

potential_targets = [c for c in df.columns if 'protest' in c.lower() or 'area_share' in c.lower() or 'contagion' in c.lower() or 'opp' in c.lower()]
print("Available Alternative Targets:")
print(potential_targets)
