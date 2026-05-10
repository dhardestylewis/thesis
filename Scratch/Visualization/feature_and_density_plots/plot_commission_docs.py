import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
plan_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\planning_commission_index.csv"
zap_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\zoning_platting_commission_index.csv"
output_dir = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"

# Load data
df_plan = pd.read_csv(plan_csv)
df_plan['Commission'] = 'Planning Commission'

df_zap = pd.read_csv(zap_csv)
df_zap['Commission'] = 'Zoning & Platting Commission'

df = pd.concat([df_plan, df_zap], ignore_index=True)

# Categorize documents
def categorize(text):
    text = str(text).lower()
    if 'agenda' in text or 'minutes' in text:
        return 'Agenda/Minutes'
    if 'staff report' in text:
        return 'Staff Report'
    if 'opposition' in text:
        return 'Opposition'
    if 'support' in text:
        return 'Support'
    if 'public comment' in text:
        return 'Public Comment'
    if 'postponement' in text:
        return 'Postponement Request'
    if 'backup' in text:
        return 'Other Backup'
    return 'Other'

df['Doc_Category'] = df['Doc_Text'].apply(categorize)

# Drop any data before 2009 to avoid 404 years
df = df[df['Year'] >= 2009]

# --- PLOT 1: Overall Document Types Bar Chart ---
plt.figure(figsize=(12, 6))
order = ['Other Backup', 'Staff Report', 'Agenda/Minutes', 'Postponement Request', 'Public Comment', 'Opposition', 'Support', 'Other']
sns.countplot(data=df, y='Doc_Category', order=order, palette='viridis')
plt.title("Distribution of Document Types (Both Commissions, 2009-2026)")
plt.xlabel("Total Documents")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "doc_types_distribution.png"), dpi=300)
plt.close()

# --- PLOT 2: NIMBY vs Staff Documents Over Time ---
# We focus on the growth of specific documents like Public Comments and Opposition vs Staff Reports
df_focus = df[df['Doc_Category'].isin(['Staff Report', 'Public Comment', 'Opposition', 'Support'])]
yearly_counts = df_focus.groupby(['Year', 'Doc_Category']).size().reset_index(name='Count')

plt.figure(figsize=(14, 7))
sns.lineplot(data=yearly_counts, x='Year', y='Count', hue='Doc_Category', marker='o', linewidth=2.5, palette=['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e'])
plt.title("Growth of NIMBY-Related Evidence Over Time (2009-2026)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Number of Documents Published")
plt.xticks(range(int(df['Year'].min()), int(df['Year'].max())+1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "nimby_evidence_timeline.png"), dpi=300)
plt.close()

# --- PLOT 3: Total Document Volume by Commission ---
yearly_comm = df.groupby(['Year', 'Commission']).size().reset_index(name='Total Docs')
plt.figure(figsize=(14, 7))
sns.barplot(data=yearly_comm, x='Year', y='Total Docs', hue='Commission', palette='Set2')
plt.title("Total Document Publishing Volume: Planning vs ZAP Commission", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Total Documents")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "commission_volume_comparison.png"), dpi=300)
plt.close()

print("Plots generated successfully!")
