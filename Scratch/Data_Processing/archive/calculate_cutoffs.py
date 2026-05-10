import pandas as pd
import numpy as np
import re
from datetime import datetime

master_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV\zoning_land_use_merged_data.csv"
votes_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\zoning_cases_with_council_votes.csv"
output_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"

df_master = pd.read_csv(master_csv)
df_votes = pd.read_csv(votes_csv)

def clean_case(c):
    c = str(c).upper().strip()
    m = re.search(r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4})', c)
    return m.group(1) if m else c

df_master['Core_Case'] = df_master['case_number'].apply(clean_case)
df_votes['Core_Case'] = df_votes['Case_Number'].apply(clean_case)

def extract_date(text):
    m = re.search(r'([A-Z][a-z]+\s+\d{1,2},\s+\d{4})', str(text))
    return pd.to_datetime(m.group(1)) if m else pd.NaT

df_votes['Council_Date'] = df_votes['Meeting_Date'].apply(extract_date)
case_agg = df_votes.groupby('Core_Case').agg(
    Final_Council_Date=('Council_Date', 'max'),
    Council_Appearances=('Council_Date', 'count')
).reset_index()

df_model = pd.merge(df_master, case_agg, on='Core_Case', how='left')
df_model['App_Date'] = pd.to_datetime(df_model['application_start_date'], errors='coerce')

# Calculate Days_in_Pipeline for completed cases
completed_cases = df_model.dropna(subset=['App_Date', 'Final_Council_Date']).copy()
completed_cases['Days_in_Pipeline'] = (completed_cases['Final_Council_Date'] - completed_cases['App_Date']).dt.days
completed_cases = completed_cases[(completed_cases['Days_in_Pipeline'] >= 0) & (completed_cases['Days_in_Pipeline'] < 2500)]

# Phase 1: The Statistical Cutoff
p95_cutoff = completed_cases['Days_in_Pipeline'].quantile(0.95)
print(f"95th Percentile Pipeline Velocity: {int(p95_cutoff)} days")

current_date = pd.to_datetime('2024-05-01')

def assign_status(row):
    if pd.notna(row['Final_Council_Date']):
        return "Completed"
    if pd.isna(row['App_Date']):
        return "Unknown"
    
    days_since_app = (current_date - row['App_Date']).days
    if days_since_app > p95_cutoff:
        return "Withdrawn_or_Dead"
    else:
        return "Ongoing"

df_model['Derived_Status'] = df_model.apply(assign_status, axis=1)

# Add days in pipeline to the final model where possible
df_model['Days_in_Pipeline'] = (df_model['Final_Council_Date'] - df_model['App_Date']).dt.days

print("Status Classification Breakdown:")
print(df_model['Derived_Status'].value_counts())

df_model.to_csv(output_csv, index=False)
print("Saved base model data to:", output_csv)
