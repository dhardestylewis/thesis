import pandas as pd
import matplotlib.pyplot as plt
import csv

# Read data
df = pd.read_csv('c:/Users/dhl/data/Thesis/thesis/Data/austin_council_meetings_index.csv')

def classify_meeting(text):
    text = text.lower()
    if 'regular meeting' in text and 'city council' in text: return 'City Council Regular'
    elif 'work session' in text: return 'Work Session'
    elif 'special called' in text: return 'Special Called'
    elif 'committee' in text: return 'Committee Meeting'
    elif 'housing finance' in text or 'ahfc' in text: return 'AHFC Board'
    elif 'joint' in text: return 'Joint Meeting'
    else: return 'Other/Special'

df['Meeting_Type'] = df['Meeting_Text'].apply(classify_meeting)

# Pivot table for plotting
pivot_df = df.groupby(['Year', 'Meeting_Type']).size().unstack(fill_value=0)

# Ensure consistent color mapping and order
cols_order = ['City Council Regular', 'Work Session', 'Committee Meeting', 'Special Called', 'AHFC Board', 'Joint Meeting', 'Other/Special']
existing_cols = [c for c in cols_order if c in pivot_df.columns]
pivot_df = pivot_df[existing_cols]

# Plot
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(12, 7))

# Colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
pivot_df.plot(kind='bar', stacked=True, ax=ax, color=colors[:len(existing_cols)], width=0.8)

plt.title('Austin City Council Meetings by Type (2007-2026)', fontsize=16, pad=20)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Meetings', fontsize=14)
plt.legend(title='Meeting Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

# Save plot to conversation artifacts dir
out_path = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\meeting_trends.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Plot saved to {out_path}")

# Print stats to console
print("---STATS---")
stats = df['Meeting_Type'].value_counts()
for k, v in stats.items():
    print(f"{k}: {v}")

print("\n---YEARLY AVG---")
yearly = df.groupby('Year').size()
print(f"Average Meetings/Year: {yearly.mean():.1f}")
print(f"Min: {yearly.min()} ({yearly.idxmin()})")
print(f"Max: {yearly.max()} ({yearly.idxmax()})")
