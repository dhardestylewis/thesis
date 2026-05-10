import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

def get_prauc_lift():
    df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
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

        for name, m in models.items():
            if name == 'TabNet': m.fit(X_train=X_train_sc, y_train=y_train, max_epochs=20)
            elif name == 'Logistic (L2)': m.fit(X_train_sc, y_train)
            else: m.fit(X_train_raw, y_train)

        for test_year in eval_years:
            if test_year < anchor: continue
            test_mask = years == test_year
            if test_mask.sum() < 5 or y[test_mask].sum() < 1: continue
                
            X_test_raw, y_test = X_raw.values[test_mask], y[test_mask]
            X_test_sc = scaler.transform(X_test_raw)
            
            for name, m in models.items():
                try:
                    p = m.predict_proba(X_test_sc if name in ['TabNet', 'Logistic (L2)'] else X_test_raw)[:, 1]
                    base_rate = y_test.sum() / len(y_test)
                    prauc = average_precision_score(y_test, p)
                    lift = prauc / base_rate
                    drift_results.append({
                        'Model': name, 'Anchor': f'Pre-{anchor}',
                        'Evaluate_Year': test_year, 'PR-AUC Lift': round(lift, 3)
                    })
                except:
                    pass

    results_df = pd.DataFrame(drift_results)
    pivot = results_df.pivot_table(index=['Model', 'Anchor'], columns='Evaluate_Year', values='PR-AUC Lift')
    anchor_max = pivot.groupby('Anchor').max()

    tex_lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption[Temporal Drift (PR-AUC Lift)]{\textbf{Temporal predictive drift: PR-AUC lift by algorithm.}}',
        r'\label{tab:temporal_drift_prauc_lift}', r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{l' + 'c'*len(eval_years) + '}', r'\toprule',
        r'\textbf{Anchor Training} & ' + ' & '.join([f'\textbf{{{y}}}' for y in eval_years]) + r' \\',
        r'\midrule'
    ]
    
    for idx in pivot.index:
        model, anchor = idx; row = pivot.loc[idx]
        r = []
        for y in eval_years:
            val = row.get(y, np.nan)
            if pd.notnull(val):
                s = f'{val:.3f}'
                if val == anchor_max.loc[anchor, y]: s = '\\textbf{' + s + '}'
                r.append(s)
            else: r.append('---')
        tex_lines.append(f'{model} {anchor} & {" & ".join(r)} \\\\' )
        
    tex_lines.extend([r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}'])
    
    OUT_DIR = os.path.join(ROOT, 'Thesis_Draft', 'Draft_v1', 'Tables')
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'temporal_drift_prauc_lift.tex'), 'w') as f:
        f.write('\n'.join(tex_lines))

get_prauc_lift()
