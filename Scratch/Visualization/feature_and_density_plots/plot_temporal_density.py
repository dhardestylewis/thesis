import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df = pd.read_csv(model_csv)

df_plot = df[df['Derived_Status'].isin(['Completed (Scraped)', 'Completed (Unscraped)'])].copy()
df_plot['Status'] = 'Approved'

# Filter out absurd anomalies
df_approved = df_plot[(df_plot['Days_in_Pipeline'].notna()) & (df_plot['Days_in_Pipeline'] < 1500) & (df_plot['Days_in_Pipeline'] >= 0)]

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid", context="talk")

ax = sns.violinplot(x="Days_in_Pipeline", y="Status", data=df_approved, color="#3498db", inner="quartile", orient="h")
plt.title('Temporal Friction: Distinguishing Swift vs Delayed Approvals', fontsize=18, fontweight='bold')
plt.xlabel('Time Between Application and Final Approval (Days)', fontsize=14)
plt.ylabel('')

# Annotate median and 95th percentile
median = df_approved['Days_in_Pipeline'].median()
p95 = df_approved['Days_in_Pipeline'].quantile(0.95)

plt.axvline(median, color='black', linestyle='--', linewidth=2, label=f'Median Approval: {int(median)} Days')
plt.axvline(p95, color='#e74c3c', linestyle=':', linewidth=3, label=f'95th Percentile Delay: {int(p95)} Days')

plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\temporal_density.png", dpi=300)
print("Temporal density generated.")
