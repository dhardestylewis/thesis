import os
import sys
import numpy as np
import pandas as pd
import warnings
import re
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')

try:
    from surgeo import SurgeoModel
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "surgeo"])
    from surgeo import SurgeoModel

def extract_surname(name_str):
    if not isinstance(name_str, str) or not name_str.strip(): return None
    name = name_str.strip().upper()
    for kw in ['LLC', 'INC', 'CORP', 'TRUST', 'LP', 'LTD', 'ASSOC', 'BANK', 'FUND', 'HOMES']:
        if kw in name: return None
    if ',' in name: return name.split(',')[0].strip()
    parts = name.split()
    if parts: return parts[0].strip()
    return None

def extract_zip(situs):
    if not isinstance(situs, str): return None
    match = re.search(r'\b(\d{5})\b', situs)
    return match.group(1) if match else None

def do_it():
    print("===================================================================")
    print(" 1. Remote Streaming: Fetching & Building Localized Neighbor Panel")
    print("===================================================================")
    ROOT = r"C:\Users\dhl\data\thesis\thesis"
    panel_file = os.path.join(ROOT, "Data", "Panel", "Output", "Property_Year_Panel_Enriched.csv")
    h0_file    = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
    
    # We do not have case_buffer_map locally anymore, so we compute BISG grouped by 'nearby_GEOID'
    # which proxies the immediate neighborhood cluster of the formal individual neighbors.
    p_names = pd.read_csv(panel_file, usecols=['nearby_GEOID', 'owner_name', 'situs_city_state_zip', 'year'], dtype=str)
    p_names = p_names.dropna(subset=['nearby_GEOID', 'owner_name'])
    
    # Sample down slightly to fit memory and execute fast for the livestream (taking roughly 50k unique owners per tract)
    p_names = p_names.drop_duplicates(subset=['nearby_GEOID', 'owner_name'])
    
    print("  Extracting Surname and ZIP definitions...")
    p_names['_surname'] = p_names['owner_name'].apply(extract_surname)
    p_names['_zip'] = p_names['situs_city_state_zip'].apply(extract_zip)
    
    has_both = p_names['_surname'].notna() & p_names['_zip'].notna()
    valid_df = p_names[has_both].copy()
    print(f"  Valid Surname+ZIP mappings found: {len(valid_df):,}")
    
    print("  Running Surgeo BISG Model (Individual Property Level)...")
    model = SurgeoModel()
    res = model.get_probabilities(valid_df['_surname'], valid_df['_zip'])
    
    valid_df['bisg_white_nbr'] = res['white'].values
    valid_df['bisg_black_nbr'] = res['black'].values
    valid_df['bisg_asian_nbr'] = res['api'].values
    valid_df['bisg_hispanic_nbr'] = res['hispanic'].values
    
    print("  Aggregating Extracted BISG probabilities to Local Geography Groups...")
    agg_cols = ['bisg_white_nbr', 'bisg_black_nbr', 'bisg_asian_nbr', 'bisg_hispanic_nbr']
    geo_bisg = valid_df.groupby('nearby_GEOID')[agg_cols].mean().reset_index()
    
    print("===================================================================")
    print(" 2. Merging Individual Demographics into Stage C Macro Pipeline")
    print("===================================================================")
    h0 = pd.read_csv(h0_file, low_memory=False)
    
    for col in agg_cols:
        if col in h0.columns: h0 = h0.drop(columns=[col])
        
    h0['nearby_GEOID'] = h0['nearby_GEOID'].astype(str)
    h0 = pd.merge(h0, geo_bisg, on='nearby_GEOID', how='left')
    
    for col in agg_cols:
        h0[col] = h0[col].fillna(h0[col].mean())
        
    print(f"  Stage C Master Matrix Shape updated: {h0.shape}")
    h0.to_csv(h0_file, index=False)
    
    print("===================================================================")
    print(" 3. Retraining Stage C Predictive Frontier with BISG Vectors")
    print("===================================================================")
    
    target_col = 'is_protested' if 'is_protested' in h0.columns else 'protest'
    h0[target_col] = pd.to_numeric(h0[target_col], errors='coerce').fillna(0).astype(int)
    
    train_mask = h0['year'] < 2022
    test_mask = h0['year'] >= 2022
    
    drop_cols = [target_col, 'case_number', 'organized_opposition', 'has_audio_record', 
                     'TCAD ID', 'date', 'application_start_date', 'final_date',
                     'standardized_tcad_id', 'Prob_H=4', 'Prob_LGBM_H=4',
                     'Prob_CB_H=4', 'Prob_Optimal_H=4', 'ipw',
                     'council_district', 'council_district_x']
    
    X = h0.drop(columns=[c for c in drop_cols if c in h0.columns])
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = h0[target_col].values
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    cb = CatBoostClassifier(iterations=200, learning_rate=0.1, verbose=0, random_state=42)
    cb.fit(X_train, y_train)
    
    preds = cb.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    pr_auc = average_precision_score(y_test, preds)
    brier = brier_score_loss(y_test, preds)
    
    print(f"\n[+] Track 1 Post-BISG Matrix Evaluation (Pre-2022 Train -> Post-2022 Test):")
    print(f"    -> ROC-AUC: {auc:.4f}")
    print(f"    -> PR-AUC:  {pr_auc:.4f}")
    print(f"    -> Brier:   {brier:.4f}")
    
    imp = dict(zip(X.columns, cb.feature_importances_))
    print("\n[+] Top 5 Influential Features in new BISG-Enriched Frontier:")
    top_5 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
    for k, v in top_5:
        print(f"    [{k}]: {v:.2f}%")
        
    print("\n[+] BISG Specific Feature Importance:")
    for k in agg_cols:
        print(f"    [{k}]: {imp.get(k, 0):.2f}%")

if __name__ == "__main__":
    do_it()
