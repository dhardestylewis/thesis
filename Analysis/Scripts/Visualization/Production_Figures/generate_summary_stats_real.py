import pandas as pd
import numpy as np
import os

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA_H0 = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
OUT_DIR = os.path.join(ROOT, "Thesis_Draft", "Draft_v1")
os.makedirs(OUT_DIR, exist_ok=True)

def generate_summary_stats():
    print("==============================================")
    print(" Generating Authentic Table 1: Summary Stats")
    print("==============================================")

    if not os.path.exists(DATA_H0):
        print("[-] Required warehouse data not found.")
        return

    df = pd.read_csv(DATA_H0, low_memory=False)

    # Columns of interest based on the previous LaTeX hardcode
    metrics = {
        'Gross Site Area (Acres)': 'gross_site_area_acres',
        'Requested Height Delta (ft)': 'delta_max_height_ft',
        'Requested FAR Delta': 'delta_max_far',
        'Requested Bldg Cov Delta (\\%)': 'delta_max_bldg_cov_pct',
        'Filing Year': 'year',
        'Organized Opposition (Binary)': 'is_protested'
    }

    results = []
    for label, col in metrics.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            count = len(s)
            mean = s.mean()
            std = s.std()
            min_val = s.min()
            median = s.median()
            max_val = s.max()
            
            # Format counts as strict integers with academic comma separation
            fmt_count = f"{int(count)}"
            
            # Dynamically select significant digits based on metric class
            if label == 'Filing Year':
                fmt = f"{label} & {fmt_count} & {mean:.1f} & {std:.2f} & {int(min_val)} & {int(median)} & {int(max_val)} \\\\"
            elif label == 'Organized Opposition (Binary)':
                fmt = f"{label} & {fmt_count} & {mean:.2f} & {std:.2f} & {int(min_val)} & {int(median)} & {int(max_val)} \\\\"
            elif label == 'Requested Height Delta (ft)':
                fmt = f"{label} & {fmt_count} & {mean:.1f} & {std:.1f} & {int(min_val)} & {int(median)} & {int(max_val)} \\\\"
            elif label == 'Requested Bldg Cov Delta (\\%)':
                fmt = f"{label} & {fmt_count} & {mean:.1f} & {std:.1f} & {int(min_val)} & {int(median)} & {int(max_val)} \\\\"
            else:
                # Default geometric float variables (FAR, Acres)
                fmt = f"{label} & {fmt_count} & {mean:.2f} & {std:.2f} & {min_val:.2f} & {median:.2f} & {max_val:.2f} \\\\"
                
            results.append(fmt)
        else:
            results.append(f"{label} & 0 & 0.00 & 0.00 & 0.0 & 0.0 & 0.0 \\\\")

    tex_content = r"""\begin{table}[ht]
\caption{Historical Panel Descriptive Statistics (V2 Full Dimensional Array)}
\label{tab:desc_stats}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrr}
\toprule
 & count & mean & std & min & Median & max \\
\midrule
"""
    tex_content += "\n".join(results)
    tex_content += r"""
\bottomrule
\end{tabular}
}
\end{table}
"""

    out_path = os.path.join(OUT_DIR, "summary_stats_table.tex")
    with open(out_path, "w") as f:
        f.write(tex_content)
        
    print(f"[+] Successfully built and saved authentic table to {out_path}")

if __name__ == '__main__':
    generate_summary_stats()
