import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Paths
ROOT_DIR = r"C:\Users\dhl\data\thesis\thesis"
ICP_PATH = os.path.join(ROOT_DIR, "Data", "Zoning_Cases", "Processed_Data", "CSV", "submission_grade_icp_matrix.csv")
H0_PATH = os.path.join(ROOT_DIR, "Data", "Warehouse_As_Of", "H0_Filing.csv")
OUT_DIR = os.path.join(ROOT_DIR, "Analysis", "Output", "Geographic_Causal")
os.makedirs(OUT_DIR, exist_ok=True)

def safe_regression(formula, data, title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    try:
        model = smf.ols(formula, data=data).fit()
        print(model.summary().tables[1])
        output_str = f"{title}\n" + model.summary().as_text() + "\n\n"
        return output_str, model
    except Exception as e:
        print(f"[-] Regression Failed: {e}")
        return f"[-] Failed: {title}\n", None

def execute_designs():
    all_output = ""
    print("Loading Data Matrices...")
    
    # 1. Load ICP Matrix for deep continuous variables (2009-2020)
    icp_df = pd.read_csv(ICP_PATH)
    icp_df['Meeting_Date'] = pd.to_datetime(icp_df['Meeting_Date'], errors='coerce')
    icp_df['year'] = icp_df['Meeting_Date'].dt.year
    icp_df['is_hd'] = icp_df['target_zoning'].fillna('').str.contains(r'-H($|-)', regex=True).astype(int)
    icp_df['is_tod'] = icp_df['target_zoning'].fillna('').str.contains(r'-TOD|-V($|-)', regex=True).astype(int)
    
    # 2. Load H0 Matrix for broader scope (2007-2024) including council district
    h0_df = pd.read_csv(H0_PATH)
    h0_df['is_protested_binary'] = h0_df['is_protested'].fillna(0).astype(int)
    
    # DESIGN 1: 10-1 Council Transition (2014) | Interrupted Time Series
    icp_df['post_10_1'] = (icp_df['year'] >= 2015).astype(int)
    res_str, m1 = safe_regression("vote_no ~ post_10_1 + acreage + is_npa", icp_df, 
                                  "DESIGN 1: 10-1 Council Transition ITS (Post-2015 Effect on Dissent Votes)")
    all_output += res_str

    # DESIGN 2: Neighborhood Plan Area (NPA) | Spatial Friction
    res_str, m2 = safe_regression("vote_no ~ is_npa + acreage", icp_df, 
                                  "DESIGN 2: Neighborhood Plan Area (NPA) Spatial Friction")
    all_output += res_str

    # DESIGN 3: HOME Geographic Response (Central vs Peripheral)
    print(f"\n{'='*70}\nDESIGN 3: HOME Initiative Central District (D9) Response DiD\n{'='*70}")
    home_df = h0_df.copy()
    # District 9 is the urban core/protest epicenter in your dataset
    home_df['is_central_treatment'] = (home_df['council_district'] == 9).astype(int)
    # HOME window: Jan 2024 onwards
    home_df['post_home_window'] = (home_df['year'] >= 2024).astype(int)
    res_str, m3 = safe_regression("is_protested_binary ~ is_central_treatment * post_home_window", home_df, 
                                  "DESIGN 3: HOME Contextual Response (Central District 9 vs Rest of Austin)")
    all_output += res_str

    # DESIGN 4: Historic District Overlays (HD)
    res_str, m4 = safe_regression("vote_no ~ is_hd + acreage", icp_df, 
                                  "DESIGN 4: Historic District (HD) Overlay Friction")
    all_output += res_str

    # DESIGN 5: Transit-Oriented Development (TOD / VMU)
    res_str, m5 = safe_regression("vote_no ~ is_tod + acreage", icp_df, 
                                  "DESIGN 5: Transit-Oriented Development (TOD/VMU) Density Deregulation Effect")
    all_output += res_str

    # DESIGN 6: 2022 Council Election Geometric Flips
    print(f"\n{'='*70}\nDESIGN 6: 2022 Council Election Geographic Flip DiD\n{'='*70}")
    elec_df = h0_df.dropna(subset=['council_district']).copy()
    elec_df['is_flipped_district'] = elec_df['council_district'].isin([4, 9]).astype(int)
    elec_df['post_2022'] = (elec_df['year'] >= 2023).astype(int)
    res_str, m6 = safe_regression("is_protested_binary ~ is_flipped_district * post_2022", elec_df, 
                                  "DESIGN 6: 2022 Council Flip Effect on Valid Petition Filing Rates")
    all_output += res_str

    # Update LaTeX config with all finding macros
    try:
        tex_path = os.path.join(ROOT_DIR, "Thesis_Draft", "Draft_v1", "Tables", "metrics_config.tex")
        with open(tex_path, "r") as f:
            lines = f.readlines()
        
        # Clean out old macros that match what we're about to write
        macro_names = ["metricFlipDiD", "metricTenOneITS", "metricNPAFriction", "metricHDFriction", "metricTODFriction"]
        new_lines = [line for line in lines if not any(m in line for m in macro_names)]
        
        with open(tex_path, "w") as f:
            for line in new_lines:
                f.write(line)
            f.write("\n% Geographic Causal Design Macros\n")
            
            # Design 1: 10-1
            if m1 is not None:
                f.write(r"\newcommand{\metricTenOneITSCoeff}{" f"{m1.params['post_10_1']:.3f}" "}\n")
                f.write(r"\newcommand{\metricTenOneITSPval}{" f"{m1.pvalues['post_10_1']:.3f}" "}\n")
            # Design 2: NPA
            if m2 is not None:
                f.write(r"\newcommand{\metricNPAFrictionCoeff}{" f"{m2.params['is_npa']:.3f}" "}\n")
                f.write(r"\newcommand{\metricNPAFrictionPval}{" f"{m2.pvalues['is_npa']:.3f}" "}\n")
            # Design 4: HD
            if m4 is not None:
                f.write(r"\newcommand{\metricHDFrictionCoeff}{" f"{m4.params['is_hd']:.3f}" "}\n")
                f.write(r"\newcommand{\metricHDFrictionPval}{" f"{m4.pvalues['is_hd']:.3f}" "}\n")
            # Design 5: TOD
            if m5 is not None:
                f.write(r"\newcommand{\metricTODFrictionCoeff}{" f"{m5.params['is_tod']:.3f}" "}\n")
                f.write(r"\newcommand{\metricTODFrictionPval}{" f"{m5.pvalues['is_tod']:.3f}" "}\n")
            # Design 3: HOME
            if m3 is not None:
                f.write(r"\newcommand{\metricHOMEDiDCoeff}{" f"{m3.params['is_central_treatment:post_home_window']:.3f}" "}\n")
                f.write(r"\newcommand{\metricHOMEDiDPval}{" f"{m3.pvalues['is_central_treatment:post_home_window']:.3f}" "}\n")
            # Design 6: 2022 Flip
            if m6 is not None:
                f.write(r"\newcommand{\metricFlipDiDCoeff}{" f"{m6.params['is_flipped_district:post_2022']:.3f}" "}\n")
                f.write(r"\newcommand{\metricFlipDiDPval}{" f"{m6.pvalues['is_flipped_district:post_2022']:.3f}" "}\n")
                
        print("[+] Exported all LaTeX macros for auxiliary designs to metrics_config.tex")
    except Exception as e:
        print(f"[-] Failed to update LaTeX macros: {e}")

    out_file = os.path.join(OUT_DIR, "geographic_causal_results.txt")
    with open(out_file, "w") as f:
        f.write(all_output)
    print(f"\n[+] All 6 Geographic Causal Designs executed. Full outputs saved to: {out_file}")

if __name__ == "__main__":
    execute_designs()
