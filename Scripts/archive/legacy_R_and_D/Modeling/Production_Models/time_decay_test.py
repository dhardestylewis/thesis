import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

PATH = r"C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\H0_Filing_Master_Enriched.csv"
df = pd.read_csv(PATH, low_memory=False)

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'has_audio_record', 
            'TCAD ID', 'date', 'application_start_date', 'final_date', 'standardized_tcad_id', 
            'Prob_H=4', 'Prob_LGBM_H=4', 'Prob_CB_H=4', 'Prob_Optimal_H=4', 'ipw', 'council_district']
df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
leak_cols = [c for c in df_clean.columns if c.startswith('tfidf_') or c.startswith('speech_')]
df_clean = df_clean.drop(columns=leak_cols)

X = df_clean.select_dtypes(include=[np.number]).fillna(0)
y = df['is_protested'].fillna(0).astype(int)

# Anchor 2022 (Train <2022, test 2022)
anchor = 2022
tr_mask = df['year'] < anchor
te_mask = df['year'] == anchor

# Base Model (Unweighted)
cb_base = CatBoostClassifier(iterations=300, auto_class_weights='Balanced', depth=6, random_seed=42, verbose=0)
cb_base.fit(X[tr_mask], y[tr_mask])
preds_base = cb_base.predict_proba(X[te_mask])[:, 1]
pr_base = average_precision_score(y[te_mask], preds_base)
print(f"Base PR-AUC (2022): {pr_base:.4f}")

# Time-Decayed Model
years_diff = (anchor - df['year'][tr_mask]).values
for rate in [0.2, 0.4, 0.6, 1.0]:
    decay_weights = np.exp(-rate * years_diff)
    cb_decay = CatBoostClassifier(iterations=300, auto_class_weights='Balanced', depth=6, random_seed=42, verbose=0)
    cb_decay.fit(X[tr_mask], y[tr_mask], sample_weight=decay_weights)
    preds_decay = cb_decay.predict_proba(X[te_mask])[:, 1]
    pr_decay = average_precision_score(y[te_mask], preds_decay)
    print(f"Decayed PR-AUC (Rate={rate}): {pr_decay:.4f}")
