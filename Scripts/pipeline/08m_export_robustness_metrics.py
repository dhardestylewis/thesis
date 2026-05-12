import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# Load data
df = pd.read_csv(ROOT / "Data/Zoning_Cases/referee_robustness_matrix.csv")

macros = []
def add_macro(name, value):
    if pd.isna(value):
        val_str = "N/A"
    else:
        val_str = f"{value:.2f}"
    # handle negative signs for latex
    val_str = val_str.replace("-", "$-$")
    macros.append(f"\\newcommand{{\\metric{name}}}{{{val_str}}}")

# 1. Nuisance Model Sensitivity
nuisance = df[df['Test_Group'] == 'Nuisance_Sensitivity'].set_index('Variant')
add_macro("RobustNuisanceLasso", nuisance.loc['Linear/Lasso', 'Height_ATE'])
add_macro("RobustNuisanceRF", nuisance.loc['RandomForest', 'Height_ATE'])
add_macro("RobustNuisanceGBM", nuisance.loc['GradBoost', 'Height_ATE'])

# 2. Seed Stability
seed = df[df['Test_Group'] == 'Seed_Stability']
mean_row = seed[seed['Variant'].str.contains('Mean')]
add_macro("RobustSeedMean", mean_row['Height_ATE'].values[0])
add_macro("RobustSeedStd", seed.set_index('Variant').loc['Std Dev', 'Height_ATE'])
# Extract the % positive from the string "Mean (N=20, 45% pos)"
pos_pct = mean_row['Variant'].values[0].split(', ')[1].split('%')[0]
macros.append(f"\\newcommand{{\\metricRobustSeedPosPct}}{{{pos_pct}\\%}}")

# 3. Treatment Robustness
treatment = df[df['Test_Group'] == 'Treatment_Robustness'].set_index('Variant')
add_macro("RobustTreatContinuous", treatment.loc['Continuous (Base)', 'Height_ATE'])
add_macro("RobustTreatLog", treatment.loc['Log-Transformed', 'Height_ATE'])
add_macro("RobustTreatWinsorized", treatment.loc['Winsorized (95th)', 'Height_ATE'])
add_macro("RobustTreatBinary", treatment.loc['Binary (>0.20)', 'Height_ATE'])

# 4. Feature Ablation
ablate = df[df['Test_Group'] == 'Feature_Ablation'].set_index('Variant')
add_macro("RobustAblateBaseline", ablate.loc['Baseline (Full X)', 'Height_ATE'])
add_macro("RobustAblateNoSpatial", ablate.loc['No Spatial', 'Height_ATE'])
add_macro("RobustAblateNoMacro", ablate.loc['No Macro', 'Height_ATE'])
add_macro("RobustAblateNoHistory", ablate.loc['No History/Lags', 'Height_ATE'])
add_macro("RobustAblateNoDemographics", ablate.loc['No Demographics', 'Height_ATE'])

# Format macros
macro_str = "\n".join(macros) + "\n"

# Append to config file
config_path = ROOT / "Thesis_Draft/GSAPP_Final_Submission/Tables/chapter4_performance/tbl_ch4_08_metrics_config.tex"

with open(config_path, 'r') as f:
    existing_content = f.read()

lines = existing_content.splitlines()
new_lines = []
for line in lines:
    if line.startswith(r"\newcommand{\metricRobust"):
        continue # Drop old macros
    new_lines.append(line)

new_content = "\n".join(new_lines) + "\n\n% --- DYNAMIC ROBUSTNESS METRICS ---\n" + macro_str

with open(config_path, 'w') as f:
    f.write(new_content)

print(f"Successfully exported {len(macros)} robustness macros to {config_path.name}")
