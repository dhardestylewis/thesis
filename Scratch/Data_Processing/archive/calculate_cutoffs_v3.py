import pandas as pd
import numpy as np

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df = pd.read_csv(model_csv)

current_date = pd.to_datetime('2024-05-01')
df['App_Date'] = pd.to_datetime(df['application_start_date'], errors='coerce')

def calc_days(row):
    if pd.notna(row.get('Final_Council_Date')) and pd.notna(row['App_Date']):
        return (pd.to_datetime(row['Final_Council_Date']) - row['App_Date']).days
    if (pd.notna(row.get('approval_date')) or pd.notna(row.get('final_date'))) and pd.notna(row['App_Date']):
        final_str = row.get('approval_date') if pd.notna(row.get('approval_date')) else row.get('final_date')
        # Some dates might just be years, let's coerce carefully
        final = pd.to_datetime(final_str, errors='coerce')
        if pd.notna(final):
            return (final - row['App_Date']).days
    return np.nan

df['label_real_days_in_pipeline'] = df.apply(calc_days, axis=1)

valid_days = df[(df['label_real_days_in_pipeline'] >= 0) & (df['label_real_days_in_pipeline'] < 5000)]['label_real_days_in_pipeline']

p100_cutoff = valid_days.max()
print(f"Absolute Maximum Pipeline Velocity (100th Percentile): {int(p100_cutoff)} days")

def assign_status_v3(row):
    if pd.notna(row.get('Final_Council_Date')):
        return "Completed (Scraped)"
        
    if pd.notna(row.get('approval_date')) or pd.notna(row.get('final_date')):
        return "Completed (Unscraped)"
        
    if pd.isna(row['App_Date']):
        return "Unknown"
        
    days_since_app = (current_date - row['App_Date']).days
    if days_since_app > p100_cutoff:
        return "Withdrawn_or_Dead"
    else:
        return "Ongoing"

df['Derived_Status'] = df.apply(assign_status_v3, axis=1)

print("NEW Status Classification Breakdown:")
print(df['Derived_Status'].value_counts())

df.to_csv(model_csv, index=False)
print("Updated model_ready_zoning_data.csv with 100th percentile cutoffs.")
