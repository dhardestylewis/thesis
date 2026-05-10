import pandas as pd
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)

print("\nValue Counts for Target Variable: 'is_protested'")
print(df['is_protested'].value_counts(dropna=False))

if 'organized_opposition' in df.columns:
    print("\nValue Counts for 'organized_opposition'")
    print(df['organized_opposition'].value_counts(dropna=False))
