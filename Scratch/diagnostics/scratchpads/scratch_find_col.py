import pandas as pd
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), nrows=10)

print("Area Share cols:", [c for c in df.columns if 'area' in c.lower() or 'share' in c.lower() or 'signed' in c.lower()])
