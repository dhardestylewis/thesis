import os
import pandas as pd

ROOT = r"C:\Users\dhl\data\thesis\thesis"
CSV_PATH = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV", "submission_grade_goldmine_tensor.csv")
OUT_LATEX = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Table_1_Summary_Stats.tex")

def main():
    print("[*] Generating Formal Descriptive Statistics Matrix...")
    if not os.path.exists(CSV_PATH):
        print("[-] Goldmine CSV missing.")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    # Isolate key continuous and binary metrics
    stats_cols = [
        'vote_no', 'valid_petition', 'neighborhood_median_wealth', 
        'neighborhood_density', 'neighborhood_protest_contagion',
        'orig_zoning_density', 'target_zoning_density', 'net_density_change',
        'is_npa', 'acreage'
    ]
    
    # Calculate raw dataframe descriptives
    desc = df[stats_cols].describe().T[['count', 'mean', 'std', 'min', 'max']].fillna(0)
    desc['mean'] = desc['mean'].round(2)
    desc['std'] = desc['std'].round(2)
    desc['min'] = desc['min'].round(2)
    desc['max'] = desc['max'].round(2)
    
    # Rename rows for academic layout
    rename_map = {
        'vote_no': 'Nay Votes per Case',
        'valid_petition': '20\\% Valid Petition Filed (NIMBY)',
        'neighborhood_median_wealth': 'Avg. Neigh. Property Value (\$M)',
        'neighborhood_density': 'Neighborhood Lot Density',
        'neighborhood_protest_contagion': 'Historical Neighborhood Objection Rate',
        'orig_zoning_density': 'Baseline Zoning Ceiling (0-10)',
        'target_zoning_density': 'Target Zoning Ceiling (0-10)',
        'net_density_change': 'Delta Zoning Density Allowed',
        'is_npa': 'Amend Neighborhood Plan Required',
        'acreage': 'Intervention Site Acreage'
    }
    desc.index = desc.index.map(lambda x: rename_map.get(x, x))
    
    # Generate explicit LaTeX code
    latex_str = "\\begin{table}[h]\n\\centering\n"
    latex_str += "\\caption{Table 1: Historical Descriptive Statistics (Austin Zoning Interventions)}\n"
    latex_str += "\\label{tab:summary_stats}\n"
    latex_str += desc.to_latex(escape=False, float_format="%.2f")
    latex_str += "\\end{table}\n"
    
    with open(OUT_LATEX, 'w', encoding='utf-8') as f:
        f.write(latex_str)
        
    print(f"[+] Output LaTeX Table successfully generated: {OUT_LATEX}")

if __name__ == "__main__":
    main()
