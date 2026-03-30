"""Phase 2: Multi-horizon opposition model with bootstrap CIs."""
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[!] CatBoost not available, using LogisticRegression only")

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data", "Warehouse_As_Of")
OUT_DIR = os.path.join(ROOT, "Analysis", "Output", "Track1_Predictive")
os.makedirs(OUT_DIR, exist_ok=True)

def compute_ece(y_true, y_prob, n_bins=10):
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        if len(prob_true) == 0: return np.nan
        return float(np.mean(np.abs(prob_true - prob_pred)))
    except:
        return np.nan

def bootstrap_metric(y_true, y_pred, metric_fn, n_boot=500, seed=42):
    """Compute metric with 95% bootstrap CI."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2: continue
        try:
            vals.append(metric_fn(yt, yp))
        except:
            continue
    if len(vals) < 10:
        return np.nan, np.nan, np.nan
    point = metric_fn(y_true, y_pred)
    lo = np.percentile(vals, 2.5)
    hi = np.percentile(vals, 97.5)
    return float(point), float(lo), float(hi)

def run_horizon(path, horizon_name, results_collector):
    """Run opposition model for a single horizon."""
    if not os.path.exists(path):
        print(f"[!] {horizon_name}: Data file not found at {path}")
        return
    
    df = pd.read_csv(path, low_memory=False)
    print(f"\n{'='*60}")
    print(f" {horizon_name}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"{'='*60}")
    
    # Find target column
    target_col = None
    for col in ['is_protested', 'organized_opposition', 'opposition']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        print(f"[!] No target column found. Available: {list(df.columns)}")
        return
    
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
    
    # Temporal column
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year']).sort_values('year')
    else:
        print(f"[!] No 'year' column found")
        return
    
    # Feature selection
    drop_cols = [target_col, 'case_number', 'organized_opposition', 'is_protested',
                 'has_audio_record', 'TCAD ID', 'date', 'application_start_date', 
                 'final_date', 'year']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df[target_col].values
    years = df['year'].values
    
    prevalence = y.mean()
    print(f"Target: {target_col}, Prevalence: {prevalence:.3f} ({y.sum()}/{len(y)})")
    print(f"Feature count: {X.shape[1]}")
    
    # Temporal split: train on years < 2022, test on 2022+
    train_mask = years < 2022
    test_mask = years >= 2022
    
    if train_mask.sum() < 20 or test_mask.sum() < 5:
        print(f"[!] Insufficient data for temporal split (train={train_mask.sum()}, test={test_mask.sum()})")
        return
    
    X_train, y_train = X.values[train_mask], y[train_mask]
    X_test, y_test = X.values[test_mask], y[test_mask]
    
    if y_test.sum() < 1 or y_train.sum() < 1:
        print(f"[!] No positive cases in {'test' if y_test.sum() < 1 else 'train'}")
        return
    
    # Model: CatBoost if available, else Logistic
    if HAS_CATBOOST:
        model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, 
                                    verbose=0, auto_class_weights='Balanced')
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        model_name = "CatBoost"
    else:
        model = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        model_name = "ElasticNet"
    
    # Compute metrics with bootstrap CIs
    pr_auc, pr_lo, pr_hi = bootstrap_metric(y_test, preds, average_precision_score)
    roc_auc, roc_lo, roc_hi = bootstrap_metric(y_test, preds, roc_auc_score)
    brier, brier_lo, brier_hi = bootstrap_metric(y_test, preds, brier_score_loss)
    ece = compute_ece(y_test, preds)
    
    result = {
        'horizon': horizon_name,
        'model': model_name,
        'n_train': int(train_mask.sum()),
        'n_test': int(test_mask.sum()),
        'prevalence': round(prevalence, 4),
        'PR-AUC': round(pr_auc, 4) if not np.isnan(pr_auc) else None,
        'PR-AUC_CI': f"[{pr_lo:.4f}, {pr_hi:.4f}]" if not np.isnan(pr_lo) else None,
        'ROC-AUC': round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        'ROC-AUC_CI': f"[{roc_lo:.4f}, {roc_hi:.4f}]" if not np.isnan(roc_lo) else None,
        'Brier': round(brier, 4) if not np.isnan(brier) else None,
        'ECE': round(ece, 4) if not np.isnan(ece) else None,
    }
    
    results_collector.append(result)
    
    print(f"\nResults ({model_name}):")
    print(f"  PR-AUC:  {result['PR-AUC']}  {result['PR-AUC_CI']}")
    print(f"  ROC-AUC: {result['ROC-AUC']}  {result['ROC-AUC_CI']}")
    print(f"  Brier:   {result['Brier']}")
    print(f"  ECE:     {result['ECE']}")

def main():
    horizons = {
        'H0 (Filing)': 'H0_Filing_Master_Enriched.csv',
        'H1 (Notice)': 'H1_Notice.csv',
        'H2 (Pre-Commission)': 'H2_Pre_Commission.csv',
        'H3 (Pre-Council)': 'H3_Pre_Council.csv',
    }
    
    results = []
    for name, filename in horizons.items():
        path = os.path.join(DATA, filename)
        run_horizon(path, name, results)
    
    # Save results
    out_path = os.path.join(OUT_DIR, "multi_horizon_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")
    
    # Generate LaTeX table
    tex_lines = []
    tex_lines.append(r"\begin{table}[htbp]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{Multi-Horizon Opposition Model Performance with 95\% Bootstrap CIs}")
    tex_lines.append(r"\label{tab:multi_horizon}")
    tex_lines.append(r"\renewcommand{\arraystretch}{1.2}")
    tex_lines.append(r"\begin{tabular}{lcccc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r"\textbf{Horizon} & \textbf{PR-AUC [95\% CI]} & \textbf{ROC-AUC} & \textbf{Brier} & \textbf{ECE} \\")
    tex_lines.append(r"\midrule")
    
    for r in results:
        if r['PR-AUC'] is not None:
            pr_str = f"{r['PR-AUC']:.3f} {r['PR-AUC_CI']}"
        else:
            pr_str = "---"
        roc_str = f"{r['ROC-AUC']:.3f}" if r['ROC-AUC'] else "---"
        brier_str = f"{r['Brier']:.3f}" if r['Brier'] else "---"
        ece_str = f"{r['ECE']:.3f}" if r['ECE'] else "---"
        tex_lines.append(f"{r['horizon']} & {pr_str} & {roc_str} & {brier_str} & {ece_str} \\\\")
    
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"\end{table}")
    
    tex_path = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Tables", "multi_horizon_results.tex")
    os.makedirs(os.path.dirname(tex_path), exist_ok=True)
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines))
    print(f"LaTeX table saved to {tex_path}")

if __name__ == '__main__':
    main()
