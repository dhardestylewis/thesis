import pandas as pd
import re

df_model = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv')
df_comm = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv')

def clean(c):
    m = re.search(r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4})', str(c).upper())
    return m.group(1) if m else str(c).upper()

total = set(df_model['Core_Case'].unique())
scraped = set(df_comm['Filename'].apply(clean).unique())

print(f'Total Unique Cases in Model: {len(total)}')
print(f'Unique Cases Scraped from Commission: {len(scraped)}')
print(f'Missing Commission PDFs: {len(total - scraped)}')
