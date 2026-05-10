import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df = pd.read_csv(model_csv)

df['Year'] = pd.to_datetime(df['application_start_date'], errors='coerce').dt.year
df = df.dropna(subset=['Year'])
df['Year'] = df['Year'].astype(int)
df_modern = df[(df['Year'] >= 2009) & (df['Year'] <= 2024)]

years = sorted(df_modern['Year'].unique())

data_rows = []

for year in years:
    df_y = df_modern[df_modern['Year'] == year]
    year_volume = len(df_y)
    
    if year_volume == 0:
        continue
        
    unr_app = len(df_y[df_y['Derived_Status'] == 'Unresolved (At Application)'])
    unr_pc = len(df_y[df_y['Derived_Status'] == 'Unresolved (At PC)'])
    unr_zap = len(df_y[df_y['Derived_Status'] == 'Unresolved (At ZAP)'])
    unr_council = len(df_y[df_y['Derived_Status'] == 'Unresolved (At Council)'])

    unscraped = len(df_y[df_y['Derived_Status'] == 'Approved (Unscraped)'])
    scraped = len(df_y[df_y['Derived_Status'] == 'Approved (Scraped)'])
    total_approved = unscraped + scraped

    data_rows.append({
        'Year': year,
        'Total Filed': year_volume,
        'Approved': total_approved,
        'Unresolved (Application)': unr_app,
        'Unresolved (Planning Commission)': unr_pc,
        'Unresolved (ZAP)': unr_zap,
        'Unresolved (Council)': unr_council,
        'Pct Approved': total_approved / year_volume * 100,
        'Pct Unresolved (Application)': unr_app / year_volume * 100,
        'Pct Unresolved (Planning Commission)': unr_pc / year_volume * 100,
        'Pct Unresolved (ZAP)': unr_zap / year_volume * 100,
        'Pct Unresolved (Council)': unr_council / year_volume * 100,
    })

df_agg = pd.DataFrame(data_rows)

# Plot 1: Raw Volume
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

plt.plot(df_agg['Year'], df_agg['Total Filed'], label='Total Filed', color='black', linewidth=3, linestyle='--')
plt.plot(df_agg['Year'], df_agg['Approved'], label='Approved', color='#2ecc71', linewidth=2.5)
plt.plot(df_agg['Year'], df_agg['Unresolved (Application)'], label='Unresolved (Application)', color='#34495e', linewidth=2)
plt.plot(df_agg['Year'], df_agg['Unresolved (Planning Commission)'], label='Unresolved (Planning Commission)', color='#2980b9', linewidth=2)
plt.plot(df_agg['Year'], df_agg['Unresolved (ZAP)'], label='Unresolved (ZAP)', color='#3498db', linewidth=2)
plt.plot(df_agg['Year'], df_agg['Unresolved (Council)'], label='Unresolved (Council)', color='#8e44ad', linewidth=2)

plt.title('Austin Zoning Pipeline: Absolute Volume of Attrition (2009-2024)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Cases', fontsize=12)
plt.xticks(years)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\attrition_volume_lines.png", dpi=300)
plt.close()

# Plot 2: Percentage
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

plt.plot(df_agg['Year'], df_agg['Pct Approved'], label='Approved', color='#2ecc71', linewidth=2.5)
plt.plot(df_agg['Year'], df_agg['Pct Unresolved (Application)'], label='Unresolved (Application)', color='#34495e', linewidth=2)
plt.plot(df_agg['Year'], df_agg['Pct Unresolved (Planning Commission)'], label='Unresolved (Planning Commission)', color='#2980b9', linewidth=2)
plt.plot(df_agg['Year'], df_agg['Pct Unresolved (ZAP)'], label='Unresolved (ZAP)', color='#3498db', linewidth=2)
plt.plot(df_agg['Year'], df_agg['Pct Unresolved (Council)'], label='Unresolved (Council)', color='#8e44ad', linewidth=2)

plt.title('Austin Zoning Pipeline: Rate of Attrition (2009-2024)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Filed Cases (%)', fontsize=12)
plt.xticks(years)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\attrition_rate_lines.png", dpi=300)
plt.close()

print("Line plots generated successfully.")
