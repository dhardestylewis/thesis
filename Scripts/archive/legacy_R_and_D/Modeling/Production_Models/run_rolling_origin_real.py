import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler
import torch

# Environment Setup
ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data", "Warehouse_As_Of")
OUT_DIR = os.path.join(ROOT, "Analysis", "Output", "Track1_Predictive")
os.makedirs(OUT_DIR, exist_ok=True)

def run_rolling_origin_drift():
    print("Executing COMPREHENSIVE Rolling-Origin Temporal Drift Analysis (Full Roster)...")
    master_path = os.path.join(DATA, 'H0_Filing_Master_Enriched.csv')
    df = pd.read_csv(master_path, low_memory=False)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year', 'is_protested']).sort_values('year')
    
    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date']
    future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
    X_raw = df.drop(columns=[c for c in (drop_cols + future_features) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    y = df['is_protested'].values
    years = df['year'].values

    anchors = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    eval_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    drift_results = []
    
    for anchor in anchors:
        print(f"\n[+] Processing Anchor: Pre-{anchor}")
        train_mask = years < anchor
        if train_mask.sum() < 50: continue
        
        X_train_raw, y_train = X_raw.values[train_mask], y[train_mask]
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_raw)

        models = {
            'CatBoost': CatBoostClassifier(iterations=100, depth=6, verbose=0, random_seed=42),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            'Logistic (L2)': LogisticRegression(class_weight='balanced', random_state=42),
            'TabNet': TabNetClassifier(verbose=0, seed=42)
        }

        # Train models
        for name, m in models.items():
            if name == 'TabNet':
                m.fit(X_train=X_train_sc, y_train=y_train, max_epochs=20)
            elif name == 'Logistic (L2)':
                m.fit(X_train_sc, y_train)
            else:
                m.fit(X_train_raw, y_train)

        for test_year in eval_years:
            if test_year < anchor: continue
            test_mask = years == test_year
            if test_mask.sum() < 5 or y[test_mask].sum() < 1: continue
                
            X_test_raw, y_test = X_raw.values[test_mask], y[test_mask]
            X_test_sc = scaler.transform(X_test_raw)
            
            for name, m in models.items():
                try:
                    p = m.predict_proba(X_test_sc if name in ['TabNet', 'Logistic (L2)'] else X_test_raw)[:, 1]
                    prauc = average_precision_score(y_test, p)
                except: prauc = np.nan
                    
                drift_results.append({
                    'Model': name, 'Anchor': f"Pre-{anchor}",
                    'Evaluate_Year': test_year, 'Offset': f"T+{test_year - anchor}",
                    'PR-AUC': round(prauc, 4) if not np.isnan(prauc) else None
                })

    # Save and Export
    results_df = pd.DataFrame(drift_results)
    results_df.to_json(os.path.join(OUT_DIR, "rolling_origin_drift.json"), orient='records', indent=2)
    
    pivot = results_df.pivot_table(index=['Model', 'Anchor'], columns='Evaluate_Year', values='PR-AUC')
    anchor_max = pivot.groupby('Anchor').max()
    
    tex_lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{\textbf{Comprehensive Temporal Predictive Drift: PR-AUC Decay by Algorithm}}",
        r"\label{tab:temporal_drift}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c"*len(eval_years) + "}", r"\toprule",
        r"\textbf{Anchor Training} & " + " & ".join([f"\\textbf{{{y}}}" for y in eval_years]) + r" \\",
        r"\midrule"
    ]
    
    for idx in pivot.index:
        model, anchor = idx; row = pivot.loc[idx]
        r = []
        for y in eval_years:
            val = row.get(y, np.nan)
            if pd.notnull(val):
                s = f"{val:.3f}"
                if val == anchor_max.loc[anchor, y]: s = f"\\textbf{{{s}}}"
                r.append(s)
            else: r.append("---")
        tex_lines.append(f"{model} {anchor} & {' & '.join(r)} \\\\" )
        
    tex_lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    with open(os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Tables", "temporal_drift_analysis.tex"), 'w') as f:
        f.write('\n'.join(tex_lines))

if __name__ == "__main__":
    run_rolling_origin_drift()
