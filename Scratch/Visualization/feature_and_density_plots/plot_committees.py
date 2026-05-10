import pandas as pd
import matplotlib.pyplot as plt
import re
import html

df = pd.read_csv('c:/Users/dhl/data/Thesis/thesis/Data/austin_council_meetings_index.csv')
committee_df = df[df['Meeting_Text'].str.lower().str.contains('committee')].copy()

def extract_committee(text):
    text = html.unescape(text) # fix &amp;
    text = re.sub(r'^[A-Za-z]+\s+\d{1,2},\s+\d{4}\s*', '', text)
    text = re.sub(r'(?i)\s*(Regular Meeting|Special Called Meeting|Special Meeting|Work Session|Meeting|Cancel.*?|Resched.*?).*', '', text)
    return text.strip()

committee_df['Committee_Name'] = committee_df['Meeting_Text'].apply(extract_committee)

# Pivot table
pivot_df = committee_df.groupby(['Year', 'Committee_Name']).size().unstack(fill_value=0)

# Keep top 10 committees, group the rest as 'Other Committees'
top_10 = committee_df['Committee_Name'].value_counts().head(10).index
other_cols = [c for c in pivot_df.columns if c not in top_10]
pivot_df['Other Committees'] = pivot_df[other_cols].sum(axis=1)

cols_to_keep = list(top_10) + ['Other Committees']
pivot_df = pivot_df[cols_to_keep]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(14, 8))

colors = plt.cm.tab20.colors[:len(cols_to_keep)]
pivot_df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8)

plt.title('Austin City Council: Committee Meeting Breakdown (2007-2026)', fontsize=16, pad=20)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Meetings', fontsize=14)
plt.legend(title='Specific Committees (Top 10)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

out_path = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\committee_trends.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Plot saved to {out_path}")
