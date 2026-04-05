import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
except: pass

from pytorch_tabnet.tab_model import TabNetClassifier
import torch

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data", "Warehouse_As_Of")

def compute_ece(y_true, y_prob, n_bins=10):
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        if len(prob_true) == 0: return np.nan
        return float(np.mean(np.abs(prob_true - prob_pred)))
    except:
        return np.nan

def benchmark_horizon(path, horizon_name, master_df=None):
    if not os.path.exists(path):
        return []
    
    df = pd.read_csv(path, low_memory=False)
    if master_df is not None and ('Notice' in horizon_name or 'Commission' in horizon_name):
        df['case_number'] = df['case_number'].astype(str).str.strip().str.upper()
        h0_cols = set(master_df.columns)
        stub_cols = set(df.columns)
        new_cols = list(stub_cols - h0_cols)
        stub_clean = df[['case_number'] + new_cols].drop_duplicates(subset=['case_number'])
        df = master_df.merge(stub_clean, on='case_number', how='left')
        for col in new_cols: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    target_col = next((col for col in ['is_protested', 'organized_opposition', 'opposition'] if col in df.columns), None)
    if not target_col or 'year' not in df.columns: return []
    
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year']).sort_values('year')
    
    drop_cols = [target_col, 'case_number', 'organized_opposition', 'is_protested',
                 'has_audio_record', 'TCAD ID', 'date', 'application_start_date', 
                 'final_date', 'year', 'signers', 'signer_pct']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    y = df[target_col].values
    years = df['year'].values
    
    train_mask = years < 2022
    test_mask = years >= 2022
    if train_mask.sum() < 20 or test_mask.sum() < 5 or y[test_mask].sum() < 1: return []
    
    X_train, y_train = X.values[train_mask], y[train_mask]
    X_test, y_test = X.values[test_mask], y[test_mask]
    
    # Scale Data for TabNet/SMOTE bounds
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    
    models = {
        'Logistic Regression (L2)': LogisticRegression(penalty='l2', max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1),
        'CatBoost': CatBoostClassifier(iterations=100, depth=6, learning_rate=0.05, verbose=0, auto_class_weights='Balanced', random_seed=42)
    }

    # Create a calibration holdout from training data for Isotonic fitting
    X_tr_fit, X_tr_cal, y_tr_fit, y_tr_cal = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    X_tr_fit_sc, X_tr_cal_sc = scaler.fit_transform(X_tr_fit), scaler.transform(X_tr_cal)
    X_test_scaled = scaler.transform(X_test)
    X_tr_fit_raw, X_tr_cal_raw = X_tr_fit, X_tr_cal  # Unscaled for tree models

    for model_name, model in models.items():
        # Baseline
        try:
            if 'CatBoost' in model_name:
                model.fit(X_tr_fit_raw, y_tr_fit, eval_set=(X_tr_cal_raw, y_tr_cal), early_stopping_rounds=20)
            else:
                model.fit(X_tr_fit_raw, y_tr_fit)
            preds_raw = model.predict_proba(X_test)[:, 1]
            ece_pre = compute_ece(y_test, preds_raw)
            # Isotonic Calibration
            cal_preds = model.predict_proba(X_tr_cal_raw)[:, 1]
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso.fit(cal_preds, y_tr_cal)
            preds_cal = iso.predict(preds_raw)
            ece_post = compute_ece(y_test, preds_cal)
            results.append({
                'Horizon': horizon_name, 'Architecture': f"{model_name} (Base)",
                'PR-AUC': average_precision_score(y_test, preds_cal),
                'ROC-AUC': roc_auc_score(y_test, preds_cal),
                'ECE (Raw)': ece_pre,
                'ECE (Cal)': ece_post
            })
        except Exception as e: print(f"{model_name} Base failed: {e}")
        
        # SMOTE
        try:
            smote = SMOTE(random_state=42)
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
            # Split SMOTE data for calibration holdout
            X_sm_fit, X_sm_cal, y_sm_fit, y_sm_cal = train_test_split(X_train_sm, y_train_sm, test_size=0.2, random_state=42, stratify=y_train_sm)
            
            # Reinstantiate unweighted model for SMOTE data
            if 'Logistic' in model_name: m_smote = LogisticRegression(penalty='l2', max_iter=1000, random_state=42)
            elif 'Random Forest' in model_name: m_smote = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
            elif 'LightGBM' in model_name: m_smote = LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
            else: m_smote = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
            
            m_smote.fit(X_sm_fit, y_sm_fit)
            preds_smote_raw = m_smote.predict_proba(X_test)[:, 1]
            ece_pre_sm = compute_ece(y_test, preds_smote_raw)
            # Isotonic Calibration
            cal_preds_sm = m_smote.predict_proba(X_sm_cal)[:, 1]
            iso_sm = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso_sm.fit(cal_preds_sm, y_sm_cal)
            preds_smote_cal = iso_sm.predict(preds_smote_raw)
            ece_post_sm = compute_ece(y_test, preds_smote_cal)
            results.append({
                'Horizon': horizon_name, 'Architecture': f"SMOTE + {model_name}",
                'PR-AUC': average_precision_score(y_test, preds_smote_cal),
                'ROC-AUC': roc_auc_score(y_test, preds_smote_cal),
                'ECE (Raw)': ece_pre_sm,
                'ECE (Cal)': ece_post_sm
            })
        except Exception as e: print(f"SMOTE + {model_name} failed: {e}")

    # Deep Learning Benchmark (TabNet)
    try:
        tabnet = TabNetClassifier(verbose=0)
        tabnet.fit(X_train=X_tr_fit_sc, y_train=y_tr_fit, eval_set=[(X_tr_cal_sc, y_tr_cal)], patience=20, max_epochs=100)
        preds_tabnet_raw = tabnet.predict_proba(X_test_scaled)[:, 1]
        ece_pre_tab = compute_ece(y_test, preds_tabnet_raw)
        # Isotonic Calibration
        cal_preds_tab = tabnet.predict_proba(X_tr_cal_sc)[:, 1]
        iso_tab = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso_tab.fit(cal_preds_tab, y_tr_cal)
        preds_tabnet_cal = iso_tab.predict(preds_tabnet_raw)
        ece_post_tab = compute_ece(y_test, preds_tabnet_cal)
        results.append({
            'Horizon': horizon_name, 'Architecture': 'TabNet',
            'PR-AUC': average_precision_score(y_test, preds_tabnet_cal),
            'ROC-AUC': roc_auc_score(y_test, preds_tabnet_cal),
            'ECE (Raw)': ece_pre_tab,
            'ECE (Cal)': ece_post_tab
        })
    except Exception as e: print(f"TabNet failed: {e}")

    return results

def main():
    print("Running Alternative Architectures Benchmark...")
    master_df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
    master_df['case_number'] = master_df['case_number'].astype(str).str.strip().str.upper()
    
    horizons = {
        'H0 (Filing)': 'H0_Filing_Master_Enriched.csv',
        'H3 (Pre-Council)': 'H3_Pre_Council.csv',
    }
    
    all_res = []
    for h_name, f_name in horizons.items():
        res = benchmark_horizon(os.path.join(DATA, f_name), h_name, master_df)
        all_res.extend(res)
        
    df_res = pd.DataFrame(all_res)
    print("\n--- ALTERNATIVE ARCHITECTURES PERFORMANCE ---")
    print(df_res.to_string(index=False))

    # Generate LaTeX
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Alternative Architectures Benchmark (SMOTE \& TabNet)}",
        r"\label{tab:alt_architectures}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Horizon} & \textbf{Architecture} & \textbf{PR-AUC} & \textbf{ROC-AUC} & \textbf{ECE (Raw)} & \textbf{ECE (Cal)} \\",
        r"\midrule"
    ]
    
    for _, row in df_res.iterrows():
        horizon_clean = row['Horizon']
        arch_clean = row['Architecture']
        pr = f"{row['PR-AUC']:.3f}" if pd.notna(row['PR-AUC']) else "---"
        roc = f"{row['ROC-AUC']:.3f}" if pd.notna(row['ROC-AUC']) else "---"
        ece_raw = f"{row['ECE (Raw)']:.3f}" if pd.notna(row.get('ECE (Raw)')) else "---"
        ece_cal = f"{row['ECE (Cal)']:.3f}" if pd.notna(row.get('ECE (Cal)')) else "---"
        tex_lines.append(f"{horizon_clean} & {arch_clean} & {pr} & {roc} & {ece_raw} & {ece_cal} \\\\")
        
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    tex_path = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Tables", "alternative_architectures.tex")
    os.makedirs(os.path.dirname(tex_path), exist_ok=True)
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines))
    print(f"LaTeX table saved to {tex_path}")

if __name__ == '__main__':
    main()
