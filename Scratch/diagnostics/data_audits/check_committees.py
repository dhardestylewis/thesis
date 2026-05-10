import pandas as pd
import re

df = pd.read_csv('c:/Users/dhl/data/Thesis/thesis/Data/austin_council_meetings_index.csv')
committee_df = df[df['Meeting_Text'].str.lower().str.contains('committee')]

def extract_committee(text):
    # Remove standard date prefixes (e.g. "May 7, 2026 ")
    text = re.sub(r'^[A-Za-z]+\s+\d{1,2},\s+\d{4}\s*', '', text)
    # Remove standard meeting suffixes
    text = re.sub(r'(?i)\s*(Regular Meeting|Special Called Meeting|Special Meeting|Work Session|Meeting|Cancel.*?|Resched.*?).*', '', text)
    return text.strip()

committee_df['Committee_Name'] = committee_df['Meeting_Text'].apply(extract_committee)

counts = committee_df['Committee_Name'].value_counts()
print("---TOP COMMITTEES---")
for k, v in counts.head(30).items():
    print(f"{v} - {k}")
