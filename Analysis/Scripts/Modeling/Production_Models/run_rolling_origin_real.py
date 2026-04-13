import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler
import torch

# Environment Setup
ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data", "Warehouse_As_Of")
OUT_DIR = os.path.join(ROOT, "Analysis", "Output", "Track1_Predictive")
os.makedirs(OUT_DIR, exist_ok=True)

def run_rolling_origin_drift():
    print("Executing Real Rolling-Origin Temporal Drift Analysis...")
    master_path = os.path.join(DATA, 'H0_Filing_Master_Enriched.csv')
    df = pd.read_csv(master_path, low_memory=False)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year', 'is_protested']).sort_values('year')
    
    # Feature list (Consistent with H0 Filing logic)
    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date']
    # Explicitly strip future features to prevent H0 leakage
    future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
    X = df.drop(columns=[c for c in (drop_cols + future_features) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    y = df['is_protested'].values
    years = df['year'].values

    anchors = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    eval_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    drift_results = []
    
    for anchor in anchors:
        print(f"\n[+] Processing Anchor: Pre-{anchor} Training Window")
        train_mask = years < anchor
        if train_mask.sum() < 50: continue
        
        X_train, y_train = X.values[train_mask], y[train_mask]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        cb_model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, verbose=0, auto_class_weights='Balanced', random_seed=42)
        cb_model.fit(X_train, y_train)

        tabnet_model = TabNetClassifier(verbose=0)
        tabnet_sparse = TabNetClassifier(
            lambda_sparse=0.1,  # Extreme structural feature pruning
            n_steps=3,
            gamma=1.5,
            verbose=0
        )
        from torch.nn import CrossEntropyLoss
        # Split train for eval
        split_idx = int(len(X_train_scaled)*0.8)
        X_tr_tab, y_tr_tab = X_train_scaled[:split_idx], y_train[:split_idx]
        X_val_tab, y_val_tab = X_train_scaled[split_idx:], y_train[split_idx:]
        
        try:
            tabnet_model.fit(
                X_train=X_tr_tab, y_train=y_tr_tab,
                eval_set=[(X_val_tab, y_val_tab)],
                patience=5, max_epochs=30,
                loss_fn=CrossEntropyLoss(label_smoothing=0.1)
            )
            tabnet_ok = True
        except Exception as e:
            print("TabNet failed to train:", e)
            tabnet_ok = False

        try:
            tabnet_sparse.fit(
                X_train=X_tr_tab, y_train=y_tr_tab,
                eval_set=[(X_val_tab, y_val_tab)],
                patience=5, max_epochs=30,
                loss_fn=CrossEntropyLoss(label_smoothing=0.1)
            )
            sparse_ok = True
        except Exception as e:
            print("TabNet Sparse failed to train:", e)
            sparse_ok = False

        models_to_eval = [('CatBoost', cb_model, False)]
        if tabnet_ok: models_to_eval.append(('TabNet(LS=0.1)', tabnet_model, True))
        if sparse_ok: models_to_eval.append(('TabNet(LS+Pruning)', tabnet_sparse, True))
        
        for test_year in eval_years:
            if test_year < anchor:
                continue
                
            test_mask = years == test_year
            offset = test_year - anchor
            
            if test_mask.sum() < 5 or y[test_mask].sum() < 1:
                continue
                
            X_test, y_test = X.values[test_mask], y[test_mask]
            X_test_scaled = scaler.transform(X_test)
            
            for m_name, m_inst, needs_scale in models_to_eval:
                try:
                    preds = m_inst.predict_proba(X_test_scaled if needs_scale else X_test)[:, 1]
                    prauc = average_precision_score(y_test, preds)
                except Exception:
                    prauc = np.nan
                    
                drift_results.append({
                    'Model': m_name,
                    'Anchor': f"Pre-{anchor}",
                    'Evaluate_Year': test_year,
                    'Offset': f"T+{offset}",
                    'PR-AUC': round(prauc, 4) if not np.isnan(prauc) else None
                })
                if not np.isnan(prauc):
                    print(f"    -> [{m_name}] Evaluated on {test_year} (T+{offset}): PR-AUC = {prauc:.4f}")

    # Generate Pivot Table for LaTeX
    results_df = pd.DataFrame(drift_results)
    pivot = results_df.pivot_table(index=['Model', 'Anchor'], columns='Evaluate_Year', values='PR-AUC')
    
    # Save to JSON
    results_df.to_json(os.path.join(OUT_DIR, "rolling_origin_drift.json"), orient='records', indent=2)
    
    # Generate LaTeX Table
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Temporal Predictive Drift: H0 Filing Performance Decay by Forecast Year}}",
        r"\label{tab:temporal_drift}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c"*len(eval_years) + "}",
        r"\toprule",
        r"\textbf{Anchor Training} & " + " & ".join([f"\\textbf{{{y}}}" for y in eval_years]) + r" \\",
        r"\midrule"
    ]
    
    for idx in pivot.index:
        row = pivot.loc[idx]
        idx_str = f"{idx[0]} {idx[1]}"
        r = [f"{row[y]:.3f}" if (y in row.index and pd.notnull(row[y])) else "---" for y in eval_years]
        tex_lines.append(f"{idx_str} & {' & '.join(r)} \\\\")
        
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}"
    ])
    
    tex_path = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Tables", "temporal_drift_analysis.tex")
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex_lines))
    print(f"\n[+] Temporal Drift LaTeX Table saved to {tex_path}")

if __name__ == "__main__":
    run_rolling_origin_drift()
