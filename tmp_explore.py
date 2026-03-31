import pandas as pd
import os

df = pd.read_csv(r'C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\H0_Filing_Complete.csv', low_memory=False)
print("Total cases:", len(df))
print("Columns:", df.columns.tolist())

# Attempt to find the ETJ or Withdrawal flags if they exist
# the original SODA case_master might have "CASE_TYPE", "DETAILED_STATUS"
cm = pd.read_csv(r'C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\Build\case_master.csv', low_memory=False)
df = df.merge(cm[['CASE_NUMBER', 'CASE_TYPE', 'DETAILED_STATUS']], left_on='case_number', right_on='CASE_NUMBER', how='left')

major = df[df['CASE_TYPE'].str.contains('Zoning|PUD|NPA', case=False, na=False)]
print("Major targets:", len(major))

non_withdrawn = major[~major['DETAILED_STATUS'].str.contains('Withdrawn|Void', case=False, na=False)]
print("Non-withdrawn:", len(non_withdrawn))

# Check ETJ - sometimes it's in LOCATION or zoning_from
non_etj = non_withdrawn[~non_withdrawn['CASE_NUMBER'].str.contains('ETJ', case=False, na=False)]
print("Non-ETJ approx:", len(non_etj))

