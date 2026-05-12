import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# Load data
df = pd.read_csv(ROOT / "Data/Zoning_Cases/cate_distribution.csv")

# Define macro generation helper
macros = []
def add_macro(name, value, is_percent=False):
    if is_percent:
        val_str = f"{value * 100:.1f}\\%"
    else:
        val_str = f"{value:.2f}"
    macros.append(f"\\newcommand{{\\metric{name}}}{{{val_str}}}")

# 1. Overall Distribution (Height and Delay)
for metric, col in [('CATEHeight', 'cate_height'), ('CATEDelay', 'cate_delay_days')]:
    add_macro(f"{metric}Median", df[col].median())
    add_macro(f"{metric}Std", df[col].std())
    add_macro(f"{metric}DecileLow", df[col].quantile(0.10))
    add_macro(f"{metric}QuartileLow", df[col].quantile(0.25))
    add_macro(f"{metric}QuartileHigh", df[col].quantile(0.75))
    add_macro(f"{metric}DecileHigh", df[col].quantile(0.90))
    add_macro(f"{metric}Min", df[col].min())
    add_macro(f"{metric}Max", df[col].max())
    add_macro(f"{metric}PctPositive", (df[col] > 0).mean(), is_percent=True)

# 2. Income Quartiles
q_names = {'Q1': 'One', 'Q2': 'Two', 'Q3': 'Three', 'Q4': 'Four'}
df['income_q'] = pd.qcut(df['median_household_income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    sub_df = df[df['income_q'] == q]
    q_str = q_names[q]
    add_macro(f"CATEIncome{q_str}HeightMean", sub_df['cate_height'].mean())
    add_macro(f"CATEIncome{q_str}HeightMedian", sub_df['cate_height'].median())
    add_macro(f"CATEIncome{q_str}DelayMean", sub_df['cate_delay_days'].mean())
    add_macro(f"CATEIncome{q_str}DelayMedian", sub_df['cate_delay_days'].median())

# 3. Renter Share Quartiles
df['renter_q'] = pd.qcut(df['renter_share'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    sub_df = df[df['renter_q'] == q]
    q_str = q_names[q]
    add_macro(f"CATERenter{q_str}HeightMean", sub_df['cate_height'].mean())
    add_macro(f"CATERenter{q_str}HeightMedian", sub_df['cate_height'].median())
    add_macro(f"CATERenter{q_str}DelayMean", sub_df['cate_delay_days'].mean())
    add_macro(f"CATERenter{q_str}DelayMedian", sub_df['cate_delay_days'].median())

# Format macros
macro_str = "\n".join(macros) + "\n"

# Append to config file if they don't already exist
config_path = ROOT / "Thesis_Draft/GSAPP_Final_Submission/Tables/chapter4_performance/tbl_ch4_08_metrics_config.tex"

with open(config_path, 'r') as f:
    existing_content = f.read()

# To avoid duplicates on re-run, we will replace existing dynamic causal macros or just filter
lines = existing_content.splitlines()
new_lines = []
for line in lines:
    if line.startswith(r"\newcommand{\metricCATE"):
        continue # Drop old CATE macros
    new_lines.append(line)

new_content = "\n".join(new_lines) + "\n\n% --- DYNAMIC CAUSAL METRICS ---\n" + macro_str

with open(config_path, 'w') as f:
    f.write(new_content)

print(f"Successfully exported {len(macros)} causal CATE macros to {config_path.name}")
