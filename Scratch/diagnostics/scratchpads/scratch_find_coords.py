import pandas as pd
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

try:
    cm = pd.read_csv(os.path.join(DATA, 'Build', 'case_master.csv'), nrows=10)
    print("case_master.csv columns:")
    print(cm.columns.tolist())
except Exception as e:
    print(e)
