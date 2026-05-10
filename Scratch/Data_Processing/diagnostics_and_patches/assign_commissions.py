import pandas as pd
import re
import time

print("Loading transcript CSV...")
df_comm = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv")

print("Loading indexes...")
df_plan = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\planning_commission_index.csv")
df_zap = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\zoning_platting_commission_index.csv")

pc_docs = set(df_plan['Doc_ID'].astype(str))
zap_docs = set(df_zap['Doc_ID'].astype(str))

def get_comm_type(filename):
    try:
        doc_id = filename.split('_')[1]
        if doc_id in pc_docs and doc_id in zap_docs:
            return 'Both'
        elif doc_id in pc_docs:
            return 'PC'
        elif doc_id in zap_docs:
            return 'ZAP'
        else:
            return 'Unknown'
    except:
        return 'Unknown'

df_comm['Comm_Type'] = df_comm['Filename'].apply(get_comm_type)

print("Loading master zoning cases...")
model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df_model = pd.read_csv(model_csv)
cases_to_find = set(df_model['Core_Case'].dropna().unique())

pattern = r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4})'

case_comm_mapping = {}

start = time.time()
print("Scanning PC documents...")
pc_text = " ".join(df_comm[df_comm['Comm_Type'] == 'PC']['Filename'].fillna('')) + " " + " ".join(df_comm[df_comm['Comm_Type'] == 'PC']['Raw_Text'].astype(str).fillna(''))
pc_text = pc_text.upper()
pc_cases = set(re.findall(pattern, pc_text)).intersection(cases_to_find)

print("Scanning ZAP documents...")
zap_text = " ".join(df_comm[df_comm['Comm_Type'] == 'ZAP']['Filename'].fillna('')) + " " + " ".join(df_comm[df_comm['Comm_Type'] == 'ZAP']['Raw_Text'].astype(str).fillna(''))
zap_text = zap_text.upper()
zap_cases = set(re.findall(pattern, zap_text)).intersection(cases_to_find)

for case in pc_cases:
    case_comm_mapping[case] = 'PC'
    
for case in zap_cases:
    if case in case_comm_mapping:
        case_comm_mapping[case] = 'Both'
    else:
        case_comm_mapping[case] = 'ZAP'

print(f"Total matched cases assigned a commission: {len(case_comm_mapping)}")

df_model['Commission_Type'] = df_model['Core_Case'].map(case_comm_mapping)

current_date = pd.to_datetime('2024-05-01')
df_model['App_Date'] = pd.to_datetime(df_model['application_start_date'], errors='coerce')
p100_cutoff = 4976

def assign_detailed_status(row):
    has_council = pd.notna(row.get('Final_Council_Date'))
    has_approval = pd.notna(row.get('approval_date')) or pd.notna(row.get('final_date'))
    comm_type = row.get('Commission_Type')
    
    if pd.isna(row['App_Date']):
        return "Unknown"
        
    days_since_app = (current_date - row['App_Date']).days
    is_dead = days_since_app > p100_cutoff
    
    if has_approval:
        if has_council:
            return "Approved (Scraped)"
        else:
            return "Approved (Unscraped)"
            
    if has_council:
        if is_dead: return "Dead (At Council)"
        return "Ongoing (At Council)"
        
    if pd.notna(comm_type):
        if comm_type == 'Both': comm_type = 'PC'
        if is_dead: return f"Dead (At {comm_type})"
        return f"Ongoing (At {comm_type})"
        
    if is_dead: return "Dead (At Application)"
    return "Ongoing (At Application)"

df_model['Derived_Status'] = df_model.apply(assign_detailed_status, axis=1)

print("\nNEW PC/ZAP Split Breakdown:")
print(df_model['Derived_Status'].value_counts())

df_model.to_csv(model_csv, index=False)
