import pandas as pd
import numpy as np

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df = pd.read_csv(model_csv)

current_date = pd.to_datetime('2024-05-01')
df['App_Date'] = pd.to_datetime(df['application_start_date'], errors='coerce')
p100_cutoff = 4976

def assign_status_v4(row):
    has_council = pd.notna(row.get('Final_Council_Date'))
    has_approval = pd.notna(row.get('approval_date')) or pd.notna(row.get('final_date'))
    
    if pd.isna(row['App_Date']):
        return "Unknown"
        
    days_since_app = (current_date - row['App_Date']).days
    is_dead = days_since_app > p100_cutoff
    
    if has_approval:
        if has_council:
            return "Approved (Scraped)"
        else:
            return "Approved (Unscraped)"
            
    # At this point, it has NO approval date.
    if has_council:
        if is_dead:
            return "Dead (At Council)"
        else:
            return "Ongoing (At Council)"
    else:
        if is_dead:
            return "Dead (Pre-Council)"
        else:
            return "Ongoing (Pre-Council)"

df['Derived_Status'] = df.apply(assign_status_v4, axis=1)

print("NEW Attrition-Split Status Breakdown:")
print(df['Derived_Status'].value_counts())

df.to_csv(model_csv, index=False)
print("Updated model_ready_zoning_data.csv with split attrition.")
