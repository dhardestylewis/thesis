import pandas as pd
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), nrows=100)
geo_cols = [c for c in df.columns if any(x in c.lower() for x in ['zip', 'neigh', 'tract', 'block', 'geo'])]
print("Potential GeoAgg Columns:", geo_cols)
