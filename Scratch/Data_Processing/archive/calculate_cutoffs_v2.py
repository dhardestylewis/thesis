import pandas as pd
import numpy as np

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df = pd.read_csv(model_csv)

p95_cutoff = 729

current_date = pd.to_datetime('2024-05-01')
df['App_Date'] = pd.to_datetime(df['application_start_date'], errors='coerce')

def assign_status_v2(row):
    if pd.notna(row.get('Final_Council_Date')):
        return "Completed (Scraped)"
        
    if pd.notna(row.get('approval_date')) or pd.notna(row.get('final_date')):
        return "Completed (Unscraped)"
        
    if pd.isna(row['App_Date']):
        return "Unknown"
        
    days_since_app = (current_date - row['App_Date']).days
    if days_since_app > p95_cutoff:
        return "Withdrawn_or_Dead"
    else:
        return "Ongoing"

df['Derived_Status'] = df.apply(assign_status_v2, axis=1)

print("NEW Status Classification Breakdown:")
print(df['Derived_Status'].value_counts())

df.to_csv(model_csv, index=False)
print("Updated model_ready_zoning_data.csv with corrected statuses.")
