"""
Phase 5: Feature Engineering (Remand Logic)
Analyzes the temporal timeline generated in Phase 4 to calculate structural delays and 
remand counts (the number of times a case was delayed or postponed).
"""

import pandas as pd
import re

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
votes_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\zoning_cases_with_council_votes.csv"
comm_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv"

df_model = pd.read_csv(model_csv)
df_votes = pd.read_csv(votes_csv)
df_comm = pd.read_csv(comm_csv)

def clean_case(c):
    c = str(c).upper().strip()
    m = re.search(r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4})', c)
    return m.group(1) if m else c

# Get First Council Date
df_votes['Core_Case'] = df_votes['Case_Number'].apply(clean_case)
def extract_date(text):
    m = re.search(r'([A-Z][a-z]+\s+\d{1,2},\s+\d{4})', str(text))
    return pd.to_datetime(m.group(1)) if m else pd.NaT

df_votes['Council_Date'] = df_votes['Meeting_Date'].apply(extract_date)
first_council = df_votes.groupby('Core_Case')['Council_Date'].min().reset_index()
first_council.columns = ['Core_Case', 'First_Council_Date']

# Get Commission Dates
df_comm['Core_Case'] = df_comm['Filename'].apply(clean_case)
df_comm['Commission_Date'] = df_comm['Raw_Text'].apply(extract_date)

# Merge
df_comm = pd.merge(df_comm, first_council, on='Core_Case', how='left')

# Calculate Remands
def is_remand(row):
    if pd.notna(row['Commission_Date']) and pd.notna(row['First_Council_Date']):
        return row['Commission_Date'] > row['First_Council_Date']
    return False

df_comm['Is_Remand'] = df_comm.apply(is_remand, axis=1)

remand_counts = df_comm.groupby('Core_Case')['Is_Remand'].sum().reset_index()
remand_counts.columns = ['Core_Case', 'Remand_Count']

# Merge to model
if 'Remand_Count' in df_model.columns:
    df_model = df_model.drop(columns=['Remand_Count'])
df_model = pd.merge(df_model, remand_counts, on='Core_Case', how='left')
df_model['Remand_Count'] = df_model['Remand_Count'].fillna(0).astype(int)

print(f"Found {df_model['Remand_Count'].sum()} total remands based on chronological cross-referencing.")
df_model.to_csv(model_csv, index=False)
