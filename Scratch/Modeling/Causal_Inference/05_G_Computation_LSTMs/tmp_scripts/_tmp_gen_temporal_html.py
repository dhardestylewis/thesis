import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

PANEL_PATH = r'c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv'
ARTIFACT_DIR = r'C:\Users\dhl\.gemini\antigravity\brain\52e35f87-22e1-4135-9cf3-329ccde9b487'
PLOT_DIR = os.path.join(ARTIFACT_DIR, 'scratch', 'temporal_eda_plots')
os.makedirs(PLOT_DIR, exist_ok=True)
HTML_PATH = os.path.join(ARTIFACT_DIR, 'temporal_eda_dashboard.html')

print('Loading panel for Temporal EDA...')
df = pd.read_csv(PANEL_PATH, low_memory=False)

# Recreate targets
final_ht = df['pdf_reduced_to_ft'].fillna(df.get('pdf_requested_height_ft', 0))
df['Target_Height_Concession'] = df.get('pdf_requested_height_ft', 0) - final_ht

exclude = ['case_number', 'period_start', 'App_Date', 'Final_Council_Date', 'period_end', 'latitude', 'longitude', 'the_geom', 'tcad_id', 'parcel_id_10', 'property_id', 'status_date', 'application_start_date', 'approval_date', 'final_date', 'period_seq']
features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

# Limit period_seq to the 95th percentile to avoid extremely noisy right tails
max_period = int(df['period_seq'].quantile(0.95))
df_temporal = df[df['period_seq'] <= max_period]

print(f'Generating temporal plots for {len(features)} features (max period_seq = {max_period})...')

html = ['<html><head><style>body {font-family: Arial, sans-serif; padding: 20px;} img {max-width: 600px; margin: 10px; border: 1px solid #ccc; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);} .plot-container {display: inline-block; margin-bottom: 30px;}</style></head><body>']
html.append('<h1>Temporal EDA Dashboard: Average Feature Trajectories</h1>')
html.append(f'<p>Showing the mean trajectory of each feature across biweekly <strong>period_seq</strong> (capped at {max_period} periods to exclude single-case long tails).</p>')

groups = {
    'Outcomes & Targets': ['Target_Height_Concession', 'resolved', 'censored', 'label_real_days_in_pipeline', 'vote_event'],
    'PDF Features (Treatment & Targets)': [c for c in features if c.startswith('pdf_')],
    'NLP Friction': [c for c in features if 'nlp' in c.lower()],
    'Petition Dynamics': [c for c in features if 'petition' in c.lower()],
    'Spatial Gravity': [c for c in features if 'active_' in c.lower() or 'knn_' in c.lower() or 'dist_' in c.lower()],
    'Macro Context (FRED)': ['mortgage_rate_30yr', 'fed_funds_rate', 'local_unemployment_rate']
}

plotted = set()

for group_name, g_feats in groups.items():
    html.append(f'<h2>{group_name}</h2>')
    for f in g_feats:
        if f in features and f not in plotted:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                means = df_temporal.groupby('period_seq')[f].mean()
                ax.plot(means.index, means.values, color='firebrick', linewidth=2)
                
                ax.set_title(f'Mean {f} over Time')
                ax.set_xlabel('Biweekly Period Sequence')
                ax.set_ylabel(f'Mean {f}')
                ax.grid(True, linestyle='--', alpha=0.6)
                
                plt.tight_layout()
                img_name = f'temporal_{f}.png'
                img_path = os.path.join(PLOT_DIR, img_name)
                plt.savefig(img_path, dpi=100)
                plt.close(fig)
                
                html.append(f'<div class="plot-container"><h3>{f}</h3><img src="scratch/temporal_eda_plots/{img_name}" /></div>')
                plotted.add(f)
            except Exception as e:
                print(f'Error on {f}: {e}')

html.append('</body></html>')

with open(HTML_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html))

print(f'Done! Saved Temporal Dashboard to {HTML_PATH}')
