import pandas as pd
import numpy as np

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
comm_cases_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\commission_reached_cases.csv"

df = pd.read_csv(model_csv)

try:
    df_comm_cases = pd.read_csv(comm_cases_csv)
    commission_cases = set(df_comm_cases['Core_Case'].unique())
except FileNotFoundError:
    print("WARNING: commission_reached_cases.csv not found!")
    commission_cases = set()

current_date = pd.to_datetime('2024-05-01')
df['App_Date'] = pd.to_datetime(df['application_start_date'], errors='coerce')
p100_cutoff = 4976

def assign_status_v5(row):
    has_council = pd.notna(row.get('Final_Council_Date'))
    has_approval = pd.notna(row.get('approval_date')) or pd.notna(row.get('final_date'))
    has_comm = row['Core_Case'] in commission_cases
    
    if pd.isna(row['App_Date']):
        return "Unknown"
        
    days_since_app = (current_date - row['App_Date']).days
    is_dead = days_since_app > p100_cutoff
    
    if has_approval:
        if has_council:
            return "Approved (Scraped)"
        else:
            return "Approved (Unscraped)"
            
    # Has no approval date.
    if has_council:
        if is_dead: return "Dead (At Council)"
        return "Ongoing (At Council)"
        
    if has_comm:
        if is_dead: return "Dead (At Commission)"
        return "Ongoing (At Commission)"
        
    # Never hit commission or council.
    if is_dead: return "Dead (At Application)"
    return "Ongoing (At Application)"

df['Derived_Status'] = df.apply(assign_status_v5, axis=1)

print("NEW Final Attrition-Split Status Breakdown:")
print(df['Derived_Status'].value_counts())

df.to_csv(model_csv, index=False)
print("Updated model_ready_zoning_data.csv with final attrition.")
