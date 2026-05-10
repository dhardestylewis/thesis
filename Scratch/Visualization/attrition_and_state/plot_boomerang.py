import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import matplotlib.dates as mdates

votes_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\zoning_cases_with_council_votes.csv"
master_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV\zoning_land_use_merged_data.csv"

df_votes = pd.read_csv(votes_csv)
df_master = pd.read_csv(master_csv)

def clean_case(c):
    c = str(c).upper().strip()
    m = re.search(r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4})', c)
    return m.group(1) if m else c

df_votes['Core_Case'] = df_votes['Case_Number'].apply(clean_case)
df_master['Core_Case'] = df_master['case_number'].apply(clean_case)

def extract_date(text):
    m = re.search(r'([A-Z][a-z]+\s+\d{1,2},\s+\d{4})', str(text))
    return pd.to_datetime(m.group(1)) if m else pd.NaT

df_votes['Council_Date'] = df_votes['Meeting_Date'].apply(extract_date)
df_votes = df_votes.dropna(subset=['Council_Date'])

case_counts = df_votes.groupby('Core_Case').size()

# Find a case with exactly 6 appearances to make a clean plot
stalled_cases = case_counts[case_counts == 6].index

if len(stalled_cases) > 0:
    top_case = stalled_cases[0]
    case_timeline = df_votes[df_votes['Core_Case'] == top_case].sort_values('Council_Date')
    
    app_date_raw = df_master[df_master['Core_Case'] == top_case]['application_start_date'].values[0]
    app_date = pd.to_datetime(app_date_raw)
    
    dates = [app_date] + case_timeline['Council_Date'].tolist()
    labels = ['1. Application Filed'] + [f'City Council Reading / Postponement {i+1}' for i in range(len(case_timeline)-1)] + ['Final Council Vote']
    
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.set_theme(style="whitegrid", context="talk")

    ax.plot(dates, [1]*len(dates), "-o", color="#34495e", markerfacecolor="#e74c3c", markersize=12, linewidth=3)

    for i, (date, label) in enumerate(zip(dates, labels)):
        y_pos = 1.05 if i % 2 == 0 else 0.95
        va = 'bottom' if i % 2 == 0 else 'top'
        
        ax.text(date, y_pos, f"{label}\n{date.strftime('%b %d, %Y')}", 
                ha='center', va=va, fontsize=11, fontweight='bold',
                bbox=dict(facecolor='#ecf0f1', alpha=0.9, edgecolor='#bdc3c7', boxstyle='round,pad=0.5'))

    ax.set_yticks([])
    ax.set_title(f'The "Boomerang" Effect: Administrative Friction Timeline for Case {top_case}', fontsize=18, fontweight='bold', pad=20)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=0, fontsize=10)
    
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\boomerang_timeline.png", dpi=300)
    plt.close()
    print(f"Boomerang timeline plot generated for case {top_case}.")
else:
    print("Could not find a suitable case to plot.")
