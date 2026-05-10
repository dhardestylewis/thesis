import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df = pd.read_csv(model_csv)

sns.set_theme(style="whitegrid", context="talk")

# Plot 1: The New Status Breakdown (Dead vs Ongoing vs Completed)
plt.figure(figsize=(10, 6))
status_counts = df['Derived_Status'].value_counts()
# Create color mapping manually
color_map = {
    'Withdrawn_or_Dead': '#e74c3c', 
    'Completed (Scraped)': '#2ecc71', 
    'Completed (Unscraped)': '#27ae60',
    'Ongoing': '#f39c12', 
    'Unknown': '#95a5a6'
}
colors = [color_map.get(idx, '#bdc3c7') for idx in status_counts.index]

ax = sns.barplot(x=status_counts.index, y=status_counts.values, palette=colors)
plt.title('Zoning Outcomes (Corrected for Unscraped Transcripts)', fontsize=16, fontweight='bold')
plt.ylabel('Number of Cases', fontsize=14)
plt.xticks(rotation=15, ha='right', fontsize=10)
for i, v in enumerate(status_counts.values):
    ax.text(i, v + 20, f"{v:,}", ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\status_breakdown.png", dpi=300)
plt.close()

# Plot 2: NIMBY Sentiment Distribution
df_opposed = df[df['Opposition_Volume'] > 0]
if len(df_opposed) > 0:
    plt.figure(figsize=(10, 6))
    sns.histplot(df_opposed['Aggregate_Sentiment'], bins=20, kde=True, color='#8e44ad', edgecolor='black')
    plt.title('Distribution of NIMBY Sentiment (VADER Polarity Score)', fontsize=18, fontweight='bold')
    plt.xlabel('Sentiment Score (-1.0 = Highly Hostile, 1.0 = Highly Supportive)', fontsize=14)
    plt.ylabel('Number of Cases', fontsize=14)
    plt.axvline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\sentiment_distribution.png", dpi=300)
    plt.close()

# Plot 3: Primary Complaints
df_complaints = df[df['Primary_Complaint'] != 'None']
if len(df_complaints) > 0:
    plt.figure(figsize=(10, 6))
    complaint_counts = df_complaints['Primary_Complaint'].value_counts()
    ax = sns.barplot(x=complaint_counts.index, y=complaint_counts.values, palette='viridis')
    plt.title('Primary NIMBY Complaints (Keyword Density)', fontsize=18, fontweight='bold')
    plt.ylabel('Number of Cases', fontsize=14)
    for i, v in enumerate(complaint_counts.values):
        ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\primary_complaints.png", dpi=300)
    plt.close()
    
print("Visuals generated.")
