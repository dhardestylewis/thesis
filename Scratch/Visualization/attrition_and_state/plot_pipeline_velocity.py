import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set aesthetic style
sns.set_theme(style="whitegrid", context="talk")

master_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV\zoning_land_use_merged_data.csv"
votes_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\zoning_cases_with_council_votes.csv"

# Load data
df_master = pd.read_csv(master_csv)
df_votes = pd.read_csv(votes_csv)

# Clean case numbers for joining
def clean_case(c):
    c = str(c).upper().strip()
    m = re.search(r'((?:C14|C814|NPA|C14H|C17)(?:-[A-Z0-9]+)?-\d{2,4}-\d{2,4})', c)
    return m.group(1) if m else c

df_master['Core_Case'] = df_master['case_number'].apply(clean_case)
df_votes['Core_Case'] = df_votes['Case_Number'].apply(clean_case)

# Calculate Application Date
df_master['App_Date'] = pd.to_datetime(df_master['application_start_date'], errors='coerce')

def extract_date(text):
    m = re.search(r'([A-Z][a-z]+\s+\d{1,2},\s+\d{4})', str(text))
    return pd.to_datetime(m.group(1)) if m else pd.NaT

df_votes['Council_Date'] = df_votes['Meeting_Date'].apply(extract_date)

# Aggregate by Core_Case to capture multi-reading or postponed cases
case_agg = df_votes.groupby('Core_Case').agg(
    First_Council_Date=('Council_Date', 'min'),
    Final_Council_Date=('Council_Date', 'max'),
    Council_Appearances=('Council_Date', 'count')
).reset_index()

# Merge to calculate deltas
df_plot = pd.merge(case_agg, df_master[['Core_Case', 'App_Date']], on='Core_Case', how='inner')
df_plot = df_plot.dropna(subset=['App_Date', 'Final_Council_Date'])

# Calculate Days in Pipeline (from application to final council vote)
df_plot['Days_in_Pipeline'] = (df_plot['Final_Council_Date'] - df_plot['App_Date']).dt.days

# Filter absurd outliers (data errors)
df_plot = df_plot[(df_plot['Days_in_Pipeline'] >= 0) & (df_plot['Days_in_Pipeline'] < 2500)]

# Plot 1: Distribution of Days in Pipeline
plt.figure(figsize=(12, 6))
sns.histplot(df_plot['Days_in_Pipeline'], bins=60, kde=True, color='#2ecc71', edgecolor='black')
plt.title('Zoning Case Velocity: Days from Application to Final Council Vote', fontsize=18, fontweight='bold')
plt.xlabel('Days in Pipeline', fontsize=14)
plt.ylabel('Number of Zoning Cases', fontsize=14)
plt.axvline(df_plot['Days_in_Pipeline'].median(), color='red', linestyle='--', label=f"Median: {int(df_plot['Days_in_Pipeline'].median())} Days")
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\pipeline_velocity.png", dpi=300)
plt.close()

# Plot 2: Council Appearances (Postponements)
plt.figure(figsize=(10, 6))
sns.countplot(x='Council_Appearances', data=df_plot[df_plot['Council_Appearances'] <= 8], palette='viridis')
plt.title('Administrative Friction: Number of Council Appearances per Case', fontsize=18, fontweight='bold')
plt.xlabel('Total Agenda Appearances (Readings & Postponements)', fontsize=14)
plt.ylabel('Number of Cases', fontsize=14)
plt.tight_layout()
plt.savefig(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\council_appearances.png", dpi=300)
plt.close()

print("Plots generated successfully.")
