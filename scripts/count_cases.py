import pandas as pd
import os

ROOT = r"C:\Users\dhl\data\thesis\thesis"
COA_ZONING_PATH = os.path.join(ROOT, "Data", "CoA_Open_Data", "Zoning", "ZC_current_edir-dcnf.csv")

print("--- Data Counts ---")
try:
    df_raw = pd.read_csv(COA_ZONING_PATH, low_memory=False)
    print("Raw Austin SODA database:", len(df_raw))
    
    # Major targets
    major = df_raw[df_raw['CASE_TYPE'].isin(['Rezoning', 'PUD', 'NPA'])]
    if len(major) == 0:
        # try lowercase or different names
        major = df_raw[df_raw['CASE_TYPE'].str.contains('Zoning|PUD|NPA', case=False, na=False)]
    print("Major targets only:", len(major))
except Exception as e:
    print("Failed to read raw:", e)

try:
    path_complete = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Complete.csv")
    df_comp = pd.read_csv(path_complete, low_memory=False)
    print("H0_Filing_Complete (after ETJ & pre-notice exclusion):", len(df_comp))
except Exception as e:
    print("Failed to read H0_Filing_Complete:", e)

try:
    path_master = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
    df_master = pd.read_csv(path_master, low_memory=False)
    print("H0_Filing_Master_Enriched (Final 518?):", len(df_master))
except Exception as e:
    print("Failed to read H0_Filing_Master_Enriched:", e)

try:
    path_nlp = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H3_Filing_Master_NLP.csv")
    df_nlp = pd.read_csv(path_nlp, low_memory=False)
    print("H3_Filing_Master_NLP:", len(df_nlp))
except Exception as e:
    print("Failed to read H3_Filing_Master_NLP:", e)
