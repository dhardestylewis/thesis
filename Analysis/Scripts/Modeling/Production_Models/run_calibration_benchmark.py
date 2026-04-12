"""Calibration Method Benchmark: Isotonic vs Platt vs Venn-Abers across model architectures."""
import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
try:
    from venn_abers import VennAbersCalibrator
    HAS_VENN_ABERS = True
except ImportError:
    HAS_VENN_ABERS = False

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data", "Warehouse_As_Of")

def compute_ece(y_true, y_prob, n_bins=10):
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        if len(prob_true) == 0: return np.nan
        return float(np.mean(np.abs(prob_true - prob_pred)))
    except:
        return np.nan

def compute_ace(y_true, y_prob, n_bins=10):
    try:
        sorted_idx = np.argsort(y_prob)
        y_prob_sorted = y_prob[sorted_idx]
        y_true_sorted = y_true[sorted_idx]
        bin_size = max(1, len(y_prob) // n_bins)
        ace = 0.0
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else len(y_prob)
            bin_prob = y_prob_sorted[start:end]
            bin_true = y_true_sorted[start:end]
            if len(bin_prob) > 0:
                ace += (len(bin_prob) / len(y_prob)) * abs(bin_prob.mean() - bin_true.mean())
        return float(ace)
    except:
        return np.nan

def main():
    print("=" * 60)
    print(" CALIBRATION METHOD BENCHMARK")
    print("=" * 60)
    
    df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
    
    target_col = next((c for c in ['is_protested', 'organized_opposition', 'opposition'] if c in df.columns), None)
    if not target_col: 
        print("[!] No target column found"); return
    
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)
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
    X_train, y_train = X.values[train_mask], y[train_mask]
    X_test, y_test = X.values[test_mask], y[test_mask]
    
    print(f"Train: {len(y_train)} | Test: {len(y_test)} | Prevalence: {y.mean():.3f}")
    
    # Base models to calibrate
    base_models = {
        'CatBoost': CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, verbose=0, 
                                        auto_class_weights='Balanced', random_seed=42),
        'LightGBM': LGBMClassifier(n_estimators=200, max_depth=6, class_weight='balanced', 
                                    random_state=42, verbose=-1, n_jobs=-1),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced',
                                                 random_state=42, n_jobs=-1),
    }
    
    results = []
    
    for model_name, base_model in base_models.items():
        print(f"\n--- {model_name} ---")
        
        # Fit base model
        if 'CatBoost' in model_name:
            base_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=30)
        else:
            base_model.fit(X_train, y_train)
        
        preds_raw = base_model.predict_proba(X_test)[:, 1]
        pr_raw = average_precision_score(y_test, preds_raw)
        ece_raw = compute_ece(y_test, preds_raw)
        ace_raw = compute_ace(y_test, preds_raw)
        brier_raw = brier_score_loss(y_test, preds_raw)
        print(f"  Uncalibrated:  PR-AUC={pr_raw:.4f}  ECE={ece_raw:.4f}  ACE={ace_raw:.4f}  Brier={brier_raw:.4f}")
        results.append({'Model': model_name, 'Calibrator': 'Uncalibrated',
                        'PR-AUC': pr_raw, 'ECE': ece_raw, 'ACE': ace_raw, 'Brier': brier_raw})
        
        # 1. Isotonic Regression
        try:
            iso_model = CalibratedClassifierCV(estimator=base_model.__class__(**base_model.get_params()) if hasattr(base_model, 'get_params') else base_model,
                                               method='isotonic', cv=3)
            iso_model.fit(X_train, y_train)
            preds_iso = iso_model.predict_proba(X_test)[:, 1]
            pr_iso = average_precision_score(y_test, preds_iso)
            ece_iso = compute_ece(y_test, preds_iso)
            ace_iso = compute_ace(y_test, preds_iso)
            brier_iso = brier_score_loss(y_test, preds_iso)
            print(f"  Isotonic:      PR-AUC={pr_iso:.4f}  ECE={ece_iso:.4f}  ACE={ace_iso:.4f}  Brier={brier_iso:.4f}")
            results.append({'Model': model_name, 'Calibrator': 'Isotonic',
                            'PR-AUC': pr_iso, 'ECE': ece_iso, 'ACE': ace_iso, 'Brier': brier_iso})
        except Exception as e:
            print(f"  Isotonic failed: {e}")
        
        # 2. Platt Scaling (Sigmoid)
        try:
            platt_model = CalibratedClassifierCV(estimator=base_model.__class__(**base_model.get_params()) if hasattr(base_model, 'get_params') else base_model,
                                                  method='sigmoid', cv=3)
            platt_model.fit(X_train, y_train)
            preds_platt = platt_model.predict_proba(X_test)[:, 1]
            pr_platt = average_precision_score(y_test, preds_platt)
            ece_platt = compute_ece(y_test, preds_platt)
            ace_platt = compute_ace(y_test, preds_platt)
            brier_platt = brier_score_loss(y_test, preds_platt)
            print(f"  Platt:         PR-AUC={pr_platt:.4f}  ECE={ece_platt:.4f}  ACE={ace_platt:.4f}  Brier={brier_platt:.4f}")
            results.append({'Model': model_name, 'Calibrator': 'Platt (Sigmoid)',
                            'PR-AUC': pr_platt, 'ECE': ece_platt, 'ACE': ace_platt, 'Brier': brier_platt})
        except Exception as e:
            print(f"  Platt failed: {e}")
        
        # 3. Venn-Abers
        if HAS_VENN_ABERS:
            try:
                va = VennAbersCalibrator()
                va.fit(preds_raw.reshape(-1, 1), y_test)  # VA calibrates on held-out scores
                # For VA we use the train predictions to calibrate
                preds_train_raw = base_model.predict_proba(X_train)[:, 1]
                va2 = VennAbersCalibrator()
                va2.fit(preds_train_raw.reshape(-1, 1), y_train)
                p0, p1 = va2.predict_proba(preds_raw.reshape(-1, 1))
                preds_va = (p0 + p1) / 2  # midpoint estimate
                pr_va = average_precision_score(y_test, preds_va)
                ece_va = compute_ece(y_test, preds_va)
                ace_va = compute_ace(y_test, preds_va)
                brier_va = brier_score_loss(y_test, preds_va)
                print(f"  Venn-Abers:    PR-AUC={pr_va:.4f}  ECE={ece_va:.4f}  ACE={ace_va:.4f}  Brier={brier_va:.4f}")
                results.append({'Model': model_name, 'Calibrator': 'Venn-Abers',
                                'PR-AUC': pr_va, 'ECE': ece_va, 'ACE': ace_va, 'Brier': brier_va})
            except Exception as e:
                print(f"  Venn-Abers failed: {e}")
    
    # Print summary
    df_res = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(" CALIBRATION BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(df_res.to_string(index=False))
    
    # Generate LaTeX table
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Calibration Method Comparison: ECE and Brier Score Across Architectures}",
        r"\label{tab:calibration_benchmark}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Base Model} & \textbf{Calibrator} & \textbf{PR-AUC} & \textbf{ECE $\downarrow$} & \textbf{ACE $\downarrow$} & \textbf{Brier $\downarrow$} \\",
        r"\midrule",
    ]
    
    for _, row in df_res.iterrows():
        pr = f"{row['PR-AUC']:.3f}"
        ece = f"{row['ECE']:.3f}"
        ace = f"{row['ACE']:.3f}" if row.get('ACE') is not None else "---"
        brier = f"{row['Brier']:.4f}"
        tex_lines.append(f"{row['Model']} & {row['Calibrator']} & {pr} & {ece} & {ace} & {brier} \\\\")
    
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    
    tex_path = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Tables", "calibration_benchmark.tex")
    os.makedirs(os.path.dirname(tex_path), exist_ok=True)
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines))
    print(f"\nLaTeX table saved to {tex_path}")

if __name__ == '__main__':
    main()
