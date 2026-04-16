import pandas as pd
import numpy as np
import os

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA_H0 = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
OUT_DIR = os.path.join(ROOT, "Thesis_Draft", "Draft_v1")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DIR2 = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Tables")
os.makedirs(OUT_DIR2, exist_ok=True)

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

    stats = {
        'Count': [],
        'Mean': [],
        'Std. Dev': [],
        'Min': [],
        'Median': [],
        'Max': []
    }

    header_cols = ["Statistic"]

    for label, col in metrics.items():
        header_cols.append(label)
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            count = len(s)
            mean = s.mean()
            std = s.std()
            min_val = s.min()
            median = s.median()
            max_val = s.max()
            
            # Format count as integer with commas in thesis if necessary, but leaving as is for strict formatting
            # Dynamically select significant digits based on metric class
            if label == 'Filing Year':
                stats['Count'].append(f"{int(count)}")
                stats['Mean'].append(f"{mean:.1f}")
                stats['Std. Dev'].append(f"{std:.2f}")
                stats['Min'].append(f"{int(min_val)}")
                stats['Median'].append(f"{int(median)}")
                stats['Max'].append(f"{int(max_val)}")
            elif label == 'Organized Opposition (Binary)':
                stats['Count'].append(f"{int(count)}")
                stats['Mean'].append(f"{mean:.2f}")
                stats['Std. Dev'].append(f"{std:.2f}")
                stats['Min'].append(f"{int(min_val)}")
                stats['Median'].append(f"{int(median)}")
                stats['Max'].append(f"{int(max_val)}")
            elif label in ['Requested Height Delta (ft)', 'Requested Bldg Cov Delta (\\%)']:
                stats['Count'].append(f"{int(count)}")
                stats['Mean'].append(f"{mean:.1f}")
                stats['Std. Dev'].append(f"{std:.1f}")
                stats['Min'].append(f"{int(min_val)}")
                stats['Median'].append(f"{int(median)}")
                stats['Max'].append(f"{int(max_val)}")
            else:
                # Default geometric float variables (FAR, Acres)
                stats['Count'].append(f"{int(count)}")
                stats['Mean'].append(f"{mean:.2f}")
                stats['Std. Dev'].append(f"{std:.2f}")
                stats['Min'].append(f"{min_val:.2f}")
                stats['Median'].append(f"{median:.2f}")
                stats['Max'].append(f"{max_val:.2f}")
        else:
            stats['Count'].append("0")
            stats['Mean'].append("0.00")
            stats['Std. Dev'].append("0.00")
            stats['Min'].append("0.0")
            stats['Median'].append("0.0")
            stats['Max'].append("0.0")

    cols_str = "l" + "r" * len(metrics)

    tex_content = r"""\begin{table}[ht]
\caption{Historical Panel Descriptive Statistics}
\label{tab:desc_stats}
\resizebox{\textwidth}{!}{%
\begin{tabular}{""" + cols_str + r"""}
\toprule
"""
    tex_content += " & ".join(header_cols) + r" \\" + "\n"
    tex_content += r"\midrule" + "\n"
    
    for stat_name in ['Count', 'Mean', 'Std. Dev', 'Min', 'Median', 'Max']:
        row = [stat_name] + stats[stat_name]
        tex_content += " & ".join(row) + r" \\" + "\n"

    tex_content += r"""\bottomrule
\end{tabular}
}
\end{table}
"""

    out_path = os.path.join(OUT_DIR, "summary_stats_table.tex")
    with open(out_path, "w") as f:
        f.write(tex_content)
        
    out_path2 = os.path.join(OUT_DIR2, "summary_stats_table.tex")
    with open(out_path2, "w") as f:
        f.write(tex_content)
        
    print(f"[+] Successfully built and saved authentic table to {out_path} and {out_path2}")

if __name__ == '__main__':
    generate_summary_stats()
