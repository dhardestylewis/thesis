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

def extract_date(text):
    m = re.search(r'([A-Z][a-z]+\s+\d{1,2},\s+\d{4})', str(text))
    return pd.to_datetime(m.group(1)) if m else pd.NaT

df_votes['Core_Case'] = df_votes['Case_Number'].apply(clean_case)
df_votes['Date'] = df_votes['Meeting_Date'].apply(extract_date)
df_votes['Event_Type'] = 'Council'

df_comm['Core_Case'] = df_comm['Filename'].apply(clean_case)
df_comm['Date'] = df_comm['Raw_Text'].apply(extract_date)

def get_comm_type(text):
    text_lower = str(text).lower()
    if 'zoning and platting' in text_lower:
        return 'ZAP'
    return 'PC'

df_comm['Event_Type'] = df_comm['Raw_Text'].apply(get_comm_type)

events_council = df_votes[['Core_Case', 'Date', 'Event_Type']].dropna(subset=['Date']).copy()
events_comm = df_comm[['Core_Case', 'Date', 'Event_Type']].dropna(subset=['Date']).copy()
events = pd.concat([events_council, events_comm])

df_model['App_Date'] = pd.to_datetime(df_model['application_start_date'], errors='coerce')
apps = df_model[['Core_Case', 'App_Date']].copy()
apps.columns = ['Core_Case', 'Date']
apps['Event_Type'] = 'App'
apps = apps.dropna(subset=['Date'])

events = pd.concat([events, apps])
events = events.sort_values(by=['Core_Case', 'Date'])

traces = {}
for case, group in events.groupby('Core_Case'):
    path = []
    council_counter = 1
    for _, event in group.iterrows():
        etype = event['Event_Type']
        if etype == 'Council':
            path.append(f"Council_Rd_{council_counter}")
            council_counter += 1
        elif etype in ['ZAP', 'PC']:
            if len(path) > 0 and 'Council' in path[-1]:
                path.append(f"{etype}_Remand")
            else:
                path.append(etype)
        else:
            path.append(etype)
            
    status = df_model[df_model['Core_Case'] == case]['Derived_Status'].values
    if len(status) > 0:
        stat = status[0]
        if 'Completed' in stat:
            path.append('Approved')
        elif 'Dead' in stat:
            path.append('Dead')
        else:
            path.append('Ongoing')
            
    clean_path = []
    for p in path:
        if not clean_path or clean_path[-1] != p:
            clean_path.append(p)
            
    traces[case] = " -> ".join(clean_path)

df_traces = pd.DataFrame(list(traces.items()), columns=['Core_Case', 'Process_Trace'])
top_traces = df_traces['Process_Trace'].value_counts()
print(top_traces.head(10))

# Export top 5 unrolled traces for Mermaid
with open(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\top_traces.txt", "w") as f:
    f.write(top_traces.head(5).to_string())
