import pandas as pd
import os
import io

csv_path = "c:/Users/dhl/data/thesis/thesis/Analysis/Output/Track1_Predictive/Metrics/unclustered_stability_H0.csv"
out_path = "c:/Users/dhl/data/thesis/thesis/Thesis_Draft/Draft_v1/Tables/unclustered_stability.tex"

df = pd.read_csv(csv_path)

# Ensure sorting
df = df.sort_values(by=['Model', 'Anchor', 'Share_Pct'], ascending=[True, True, False])

anchors = sorted(df['Anchor'].unique())

# Define spatiotemporal/structural features to italicize for emphasis
spatial_features = [
    'Year', 'Site Area', 'Latitude', 'Longitude', 
    'Median Structure Age', 'Structure Age', 'Median Sqft', 'Sqft'
]

tex = "\\begin{table}[H]\n\\centering\n"
tex += "\\caption{Top Unclustered Features by Anchor Year: CatBoost vs. LightGBM (Dynamic View)}\n"
tex += "\\label{tab:unclustered_dynamic}\n"
tex += "\\resizebox{\\textwidth}{!}{\n"
tex += "\\begin{tabular}{cp{7cm}p{7cm}}\n"
tex += "\\toprule\n"
tex += "\\textbf{Anchor Year} & \\textbf{CatBoost} & \\textbf{LightGBM} \\\\\n"
tex += "\\midrule\n"

for anchor in anchors:
    cb = df[(df['Anchor'] == anchor) & (df['Model'] == 'CatBoost')].head(4)
    lgbm = df[(df['Anchor'] == anchor) & (df['Model'] == 'LightGBM')].head(4)
    
    # Process CatBoost features
    cb_lines = []
    for i, row in enumerate(cb.iterrows()):
        feat = row[1]['Feature']
        val = row[1]['Share_Pct']
        display_feat = f"\\textit{{{feat}}}" if feat in spatial_features else feat
        # Escape percentage
        cb_lines.append(f"{i+1}. {display_feat} ({val:.1f}\\%)")
    cb_str = " \\newline ".join(cb_lines).replace('&', '\\&')
    
    # Process LightGBM features
    lgbm_lines = []
    for i, row in enumerate(lgbm.iterrows()):
        feat = row[1]['Feature']
        val = row[1]['Share_Pct']
        display_feat = f"\\textit{{{feat}}}" if feat in spatial_features else feat
        # Escape percentage
        lgbm_lines.append(f"{i+1}. {display_feat} ({val:.1f}\\%)")
    lgbm_str = " \\newline ".join(lgbm_lines).replace('&', '\\&')
    
    tex += f"\\textbf{{{anchor}}} & {cb_str} & {lgbm_str} \\\\\n"
    tex += "\\midrule\n" if anchor != anchors[-1] else "\\bottomrule\n"

tex += "\\end{tabular}\n"
tex += "}\n"
tex += "\\vspace{1em}\n"
tex += "\\small{\\textit{Note:} This table presents the freshest (dynamic) unclustered view of feature attribution at each temporal anchor. Spatiotemporal and structural size features are emphasized in \\textit{italics} to highlight how LightGBM consistently relies on them, while CatBoost prioritizes demographics and surrounding neighborhood characteristics.}\n"
tex += "\\end{table}\n"

# Explicitly write as UTF-8 without BOM
with io.open(out_path, "w", encoding="utf-8") as f:
    f.write(tex)

print(f"Successfully generated {out_path}")

