import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

master_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV\zoning_land_use_merged_data.csv"
votes_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\zoning_cases_with_council_votes.csv"

df_master = pd.read_csv(master_csv)
df_votes = pd.read_csv(votes_csv)

def clean_case(c):
    c = str(c).upper().strip()
    m = re.search(r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4})', c)
    return m.group(1) if m else c

df_master['Core_Case'] = df_master['case_number'].apply(clean_case)
df_master['Year'] = pd.to_datetime(df_master['application_start_date'], errors='coerce').dt.year
df_modern = df_master[df_master['Year'] >= 2009].copy()

total_applied = df_modern['Core_Case'].nunique()

df_votes['Core_Case'] = df_votes['Case_Number'].apply(clean_case)
# Get cases that made it to Council
council_cases = df_modern[df_modern['Core_Case'].isin(df_votes['Core_Case'])]
total_reached_council = council_cases['Core_Case'].nunique()

# Get cases that were postponed
case_counts = df_votes.groupby('Core_Case').size()
postponed_cases = case_counts[case_counts > 1].index
total_postponed = council_cases[council_cases['Core_Case'].isin(postponed_cases)]['Core_Case'].nunique()

swift_cases = total_reached_council - total_postponed

# The Valley of Death cases (died before Council)
died_early = total_applied - total_reached_council

# Simple Bar Chart for Pipeline Drop-off
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid", context="talk")

x = ['1. Application Filed', '2. Reached Council', '3A. Swift Vote', '3B. Postponed Loop']
y = [total_applied, total_reached_council, swift_cases, total_postponed]
colors = ['#34495e', '#3498db', '#2ecc71', '#e74c3c']

bars = plt.bar(x, y, color=colors, edgecolor='black')
plt.title('The Zoning Valley of Death: Case Progression (2009-2024)', fontsize=18, fontweight='bold')
plt.ylabel('Number of Zoning Cases', fontsize=14)

# Add value labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{int(yval):,}", ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add percentage annotations
plt.text(0.5, total_applied * 0.85, f"Lost to Administrative\nAttrition: {died_early:,} cases\n({(died_early/total_applied)*100:.1f}%)", ha='center', fontsize=12, color='#e74c3c', style='italic', bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e74c3c'))

plt.text(2.5, total_reached_council * 0.8, f"Dragged by NIMBYs\n{total_postponed:,} cases\n({(total_postponed/total_reached_council)*100:.1f}% of Council cases)", ha='center', fontsize=12, color='#e74c3c', style='italic', bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e74c3c'))

plt.tight_layout()
plt.savefig(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\state_progression.png", dpi=300)
plt.close()

print("State progression plot generated.")
