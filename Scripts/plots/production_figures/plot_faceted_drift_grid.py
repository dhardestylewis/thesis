import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

sys.path.append(os.path.abspath('Scripts'))
try:
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

ROOT_DIR = os.path.abspath('.')
FIG_DIR = os.path.join(ROOT_DIR, 'Thesis_Draft', 'Draft_v1', 'Figures', 'exhibits')
os.makedirs(FIG_DIR, exist_ok=True)

def parse_drift_table(filepath, metric_name):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return pd.DataFrame()
        
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = []
    in_body = False
    for line in lines:
        line = line.strip()
        if line.startswith('\midrule'):
            in_body = True
            continue
        if line.startswith('\bottomrule'):
            in_body = False
            break
        
        if in_body and line and not line.startswith('%'):
            clean = line.replace('\\textbf{', '').replace('}', '').replace('\\\\', '').strip()
            parts = [p.strip() for p in clean.split('&')]
            
            if len(parts) == 9:
                model = parts[0]
                anchor = parts[1]
                for i, year in enumerate([2018, 2019, 2020, 2021, 2022, 2023, 2024]):
                    val = parts[i+2]
                    if val != '---':
                        try:
                            # Handle '+0.123' -> 0.123
                            fval = float(val)
                            
                            # Parse out AnchorYear
                            anchor_year = int(anchor.replace('Pre-', ''))
                            offset = year - anchor_year
                            
                            data.append({
                                'Model': model, 
                                'Anchor': anchor, 
                                'AnchorYear': anchor_year,
                                'TestYear': year, 
                                'Offset': offset,
                                'Value': fval,
                                'Metric': metric_name
                            })
                        except ValueError:
                            pass
    return pd.DataFrame(data)

df_prauc = parse_drift_table(os.path.join(ROOT_DIR, 'Thesis_Draft', 'Draft_v1', 'Tables', 'chapter4_performance', 'tbl_ch4_14_temporal_drift_analysis.tex'), 'PR-AUC')
df_lift = parse_drift_table(os.path.join(ROOT_DIR, 'Thesis_Draft', 'Draft_v1', 'Tables', 'chapter4_performance', 'tbl_ch4_17_temporal_drift_prauc_lift.tex'), 'Lift')

df = pd.concat([df_prauc, df_lift], ignore_index=True)

# Remove the 'Pre-' from anchor string
df['AnchorLabel'] = df['Anchor'].apply(lambda x: x.replace('Pre-', 'Anchor < '))

# Essential models
essential_models = ['CatBoost', 'Logistic (L2)', 'Spatial-FE Logistic', 'TabNet', 'XGBoost', 'Anchor Regression (Causal)']
df = df[df['Model'].isin(essential_models)]

def plot_grid(data, x_col, x_label, metric, y_lim, out_name):
    sub = data[data['Metric'] == metric].copy()
    if len(sub) == 0:
        return
        
    g = sns.relplot(
        data=sub, 
        x=x_col, 
        y='Value', 
        hue='Model', 
        col='AnchorLabel', 
        col_wrap=3, 
        kind='line',
        marker='o',
        height=3.5, 
        aspect=1.2,
        palette='muted'
    )

    g.set_axis_labels(x_label, f"Out-Of-Sample {metric}")
    g.set_titles("{col_name}")
    if y_lim:
        g.set(ylim=y_lim)

    for ax in g.axes.flat:
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.suptitle(f"Temporal Predictive Drift by Model Architecture ({metric})", y=1.05)
    
    out_file = os.path.join(FIG_DIR, out_name)
    plt.savefig(out_file, bbox_inches='tight')
    print(f"Saved: {out_file}")

# 1. PR-AUC x TestYear
plot_grid(df, 'TestYear', 'Forecasting Horizon (Year)', 'PR-AUC', (0, 1.05), "fig_temporal_drift_prauc_testyear.pdf")

# 2. PR-AUC x Offset
plot_grid(df, 'Offset', 'Out-Of-Distribution Temporal Offset (+Years)', 'PR-AUC', (0, 1.05), "fig_temporal_drift_prauc_offset.pdf")

# 3. Lift x TestYear
# Get base rate to perhaps draw horizontal line if needed, or just let it float (ylim=None since Lift can be huge like 30x)
plot_grid(df, 'TestYear', 'Forecasting Horizon (Year)', 'Lift', None, "fig_temporal_drift_lift_testyear.pdf")

# 4. Lift x Offset
plot_grid(df, 'Offset', 'Out-Of-Distribution Temporal Offset (+Years)', 'Lift', None, "fig_temporal_drift_lift_offset.pdf")

