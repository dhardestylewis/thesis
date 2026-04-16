import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, re

# Add Scripts dir to path for thesis style
sys.path.append(os.path.abspath('Scripts'))
try:
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

ROOT_DIR = os.path.abspath('.')
FIG_DIR = os.path.join(ROOT_DIR, 'Thesis_Draft', 'Draft_v1', 'Figures', 'exhibits')
os.makedirs(FIG_DIR, exist_ok=True)

# Path to the already-generated temporal predictive drift LaTeX matrix
TABLE_PATH = os.path.join(ROOT_DIR, 'Thesis_Draft', 'Draft_v1', 'Tables', 'appendices_drift', 'tbl_ch4_14_temporal_drift_analysis_t25.tex')

if not os.path.exists(TABLE_PATH):
    print("Drift table not found.")
    sys.exit(1)

with open(TABLE_PATH, 'r') as f:
    lines = f.readlines()

data = []
in_body = False
for line in lines:
    line = line.strip()
    if line.startswith('\midrule'):
        in_body = True
        continue
    if line.startswith('\\bottomrule'):
        in_body = False
        break
    
    if in_body and line and not line.startswith('%'):
        # Clean latex formatting (like \textbf{0.678} -> 0.678)
        clean = line.replace('\\textbf{', '').replace('}', '').replace('\\\\', '').strip()
        parts = [p.strip() for p in clean.split('&')]
        
        if len(parts) == 9:
            model = parts[0]
            anchor = parts[1]
            for i, year in enumerate([2018, 2019, 2020, 2021, 2022, 2023, 2024]):
                val = parts[i+2]
                if val != '---':
                    try:
                        fval = float(val)
                        data.append({'Model': model, 'Anchor': anchor, 'TestYear': year, 'PR_AUC': fval})
                    except ValueError:
                        pass

df = pd.DataFrame(data)

# Remove the 'Pre-' from anchor string for cleaner titles
df['Anchor'] = df['Anchor'].apply(lambda x: x.replace('Pre-', 'Anchor < '))

# Filter to essential benchmark models to avoid clutter
essential_models = ['CatBoost', 'Logistic (L2)', 'Spatial-FE Logistic', 'TabNet', 'XGBoost', 'Anchor Regression (Causal)']
df = df[df['Model'].isin(essential_models)]

g = sns.relplot(
    data=df, 
    x='TestYear', 
    y='PR_AUC', 
    hue='Model', 
    col='Anchor', 
    col_wrap=3, 
    kind='line',
    marker='o',
    height=3.5, 
    aspect=1.2,
    palette='muted'
)

g.set_axis_labels("Forecasting Horizon (Year)", "Out-Of-Sample PR-AUC")
g.set_titles("{col_name}")
g.set(ylim=(0, 1.05))

# Make x-ticks integers
for ax in g.axes.flat:
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.suptitle("Temporal Predictive Drift by Model Architecture", y=1.05)

out_file = os.path.join(FIG_DIR, "fig_temporal_drift_H0.pdf")
plt.savefig(out_file, bbox_inches='tight')
print(f"Saved: {out_file}")
