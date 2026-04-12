"""Phase 2: Multi-horizon opposition model with bootstrap CIs."""
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[!] CatBoost not available, using LogisticRegression only")

ROOT = r"C:\Users\dhl\data\thesis\thesis"
import sys
_scripts_dir = os.path.join(ROOT, "Analysis", "Scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
try:
    from Modeling.Utilities_and_Logs import lib_metrics
except ImportError:
    lib_metrics = None

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

def run_horizon(path, horizon_name, results_collector, master_df=None):
    """Run opposition model for a single horizon. Dynamically joins stubs to master spine."""
    if not os.path.exists(path):
        print(f"[!] {horizon_name}: Data file not found at {path}")
        return
    
    df = pd.read_csv(path, low_memory=False)
    
    # STUB REINTEGRATION: If this is H1, H2, or H3, it's a lightweight stub. 
    # We MUST dynamically join it onto the robust 141-col Master baseline to prevent data loss.
    if master_df is not None and ('Notice' in horizon_name or 'Commission' in horizon_name or 'Council' in horizon_name):
        print(f"[+] Reintegrating stub '{horizon_name}' (cols={df.shape[1]}) dynamically against Master Spine...")
        df['case_number'] = df['case_number'].astype(str).str.strip().str.upper()
        
        # Isolate exactly the new columns to merge to prevent duplication
        h0_cols = set(master_df.columns)
        stub_cols = set(df.columns)
        new_cols = list(stub_cols - h0_cols)
        
        # Deduplicate the stub to prevent Cartesian row explosions 
        stub_clean = df[['case_number'] + new_cols].drop_duplicates(subset=['case_number'])
        
        # Left-join ensures we preserve all 7k rows and 141 columns from the Master
        df = master_df.merge(stub_clean, on='case_number', how='left')
        print(f"    -> Reintegrated {len(new_cols)} new columns: {new_cols}")
        
        # Fill missing downstream event characteristics (e.g., cases w/o petitions get 0 signers)
        for col in new_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
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
    
    # Safely convert target, but DO NOT fillna(0) since we built explicit petition_record_found NaNs
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    # Now it is safe to coerce into integers
    df[target_col] = df[target_col].astype(int)
    
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
                 'final_date', 'year', 'signers', 'signer_pct']
    
    # -------------------------------------------------------------------------
    # TEMPORAL FEATURE GUARD: ENFORCE STAGE-C SEQUENCE (Filing -> Notice -> Comm -> Council)
    # This ensures that early horizons cannot 'cheat' by using downstream features.
    # -------------------------------------------------------------------------
    future_features = []
    if 'Filing' in horizon_name:
        # Filing has ZERO neighbor context (H1), zero staff feedback (H2), zero text (H3)
        future_features = [
            'median_appraised_value', 'mean_appraised_value', 'std_appraised_value',
            'median_sqft', 'median_structure_age', 'median_neighbor_acreage',
            'median_neighbor_far', 'owner_occupancy_share', 'senior_share',
            'neighbor_sf_share', 'neighbor_mf_share', 'neighbor_comm_share',
            'renter_share', 'median_household_income', 'staff_recommendation_cat',
            'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr'
        ]
    elif 'Notice' in horizon_name:
        # Notice has Neighbor context (H1), but zero staff feedback (H2) or text (H3)
        future_features = ['staff_recommendation_cat', 'agenda_text_raw']
    elif 'Commission' in horizon_name:
        # Commission has staff feedback (H2), but zero text (H3)
        future_features = ['agenda_text_raw']
        
    X = df.drop(columns=[c for c in (drop_cols + future_features) if c in df.columns], errors='ignore')
    print(f"    [Guard] Stripped {len([c for c in future_features if c in df.columns])} future columns to enforce {horizon_name} temporal closure.")
    
    # Pluck out text column for separate dynamic rolling processing before coercion
    has_text = 'agenda_text_raw' in X.columns
    if has_text:
        text_series = X['agenda_text_raw'].reset_index(drop=True)
        X = X.drop(columns=['agenda_text_raw'])
    
    # One-hot encode string categoricals (like staff_recommendation_cat)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
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
    
    # -------------------------------------------------------------------------
    # DYNAMIC NLP EMBEDDING
    # Here, we prevent future leakage by fitting text representation strictly
    # on the training window and transforming the held-out validation.
    # -------------------------------------------------------------------------
    if has_text:
        print("[+] Integrating rolling Time-Aware NLP...")
        import importlib.util
        nlp_path = os.path.join(_scripts_dir, "Pipeline", "02_Transcription_and_NLP", "build_tfidf_embeddings.py")
        spec = importlib.util.spec_from_file_location("build_tfidf_embeddings", nlp_path)
        nlp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nlp_module)
        
        embedder = nlp_module.TimeAwareTextEmbedder()
        # Ensure we feed the correct subsets since we reindexed earlier
        train_nlp = embedder.fit_transform(text_series[train_mask])
        test_nlp = embedder.transform(text_series[test_mask])
        
        # Concat numeric
        X_train = np.hstack([X_train, train_nlp.values])
        X_test = np.hstack([X_test, test_nlp.values])
        print(f"    -> Text dynamically encoded: {train_nlp.shape[1]} components added.")
    
    if y_test.sum() < 1 or y_train.sum() < 1:
        print(f"[!] No positive cases in {'test' if y_test.sum() < 1 else 'train'}")
        return
    
    # Model: CatBoost if available, else Logistic
    if HAS_CATBOOST:
        base_model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, 
                                    verbose=0, auto_class_weights='Balanced')
        base_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=40)
        preds_uncalibrated = base_model.predict_proba(X_test)[:, 1]
        
        calibrated_model = CalibratedClassifierCV(
            estimator=CatBoostClassifier(iterations=100, depth=6, learning_rate=0.05, verbose=0, auto_class_weights='Balanced', random_seed=42),
            method='sigmoid', cv=3
        )
        calibrated_model.fit(X_train, y_train)
        preds_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
        model_name = "CatBoost"
    else:
        base_model = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)
        base_model.fit(X_train, y_train)
        preds_uncalibrated = base_model.predict_proba(X_test)[:, 1]
        
        calibrated_model = CalibratedClassifierCV(
            estimator=LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0),
            method='sigmoid', cv=3
        )
        calibrated_model.fit(X_train, y_train)
        preds_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
        model_name = "ElasticNet"
    
    # Compute metrics with bootstrap CIs on Calibrated output
    pr_auc, pr_lo, pr_hi = bootstrap_metric(y_test, preds_calibrated, average_precision_score)
    roc_auc, roc_lo, roc_hi = bootstrap_metric(y_test, preds_calibrated, roc_auc_score)
    brier, brier_lo, brier_hi = bootstrap_metric(y_test, preds_calibrated, brier_score_loss)
    ece_pre = compute_ece(y_test, preds_uncalibrated)
    ace_pre = compute_ace(y_test, preds_uncalibrated)
    ece_post = compute_ece(y_test, preds_calibrated)
    ace_post = compute_ace(y_test, preds_calibrated)
    
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
        'ECE_Pre': round(ece_pre, 4) if not np.isnan(ece_pre) else None,
        'ACE_Pre': round(ace_pre, 4) if not np.isnan(ace_pre) else None,
        'ECE_Post': round(ece_post, 4) if not np.isnan(ece_post) else None,
        'ACE_Post': round(ace_post, 4) if not np.isnan(ace_post) else None,
    }
    
    results_collector.append(result)
    
    print(f"\nResults ({model_name}):")
    print(f"  PR-AUC:  {result['PR-AUC']}  {result['PR-AUC_CI']}")
    print(f"  ROC-AUC: {result['ROC-AUC']}  {result['ROC-AUC_CI']}")
    print(f"  Brier:   {result['Brier']}")
    print(f"  ECE(Pre):{result['ECE_Pre']}")
    print(f"  ECE(Pos):{result['ECE_Post']}")

def main():
    horizons = {
        'H0 (Filing)': 'H0_Filing_Master_Enriched.csv',
        'H1 (Notice)': 'H1_Notice.csv',
        'H2 (Pre-Commission)': 'H2_Pre_Commission.csv',
        'H3 (Pre-Council)': 'H3_Pre_Council.csv',
    }
    
    master_path = os.path.join(DATA, 'H0_Filing_Master_Enriched.csv')
    if not os.path.exists(master_path):
        print(f"[!] Critical: Master spine {master_path} not found.")
        return
        
    print(f"Loading 141-column master baseline from {master_path}...")
    master_df = pd.read_csv(master_path, low_memory=False)
    if 'case_number' in master_df.columns:
        master_df['case_number'] = master_df['case_number'].astype(str).str.strip().str.upper()
    
    results = []
    for name, filename in horizons.items():
        path = os.path.join(DATA, filename)
        run_horizon(path, name, results, master_df)
    
    # Save results
    out_path = os.path.join(OUT_DIR, "multi_horizon_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")
    
    # Identify maximums and minimums for bolding
    valid_pr = [r['PR-AUC'] for r in results if r['PR-AUC'] is not None]
    max_pr = max(valid_pr) if valid_pr else None
    
    valid_roc = [r['ROC-AUC'] for r in results if r['ROC-AUC'] is not None]
    max_roc = max(valid_roc) if valid_roc else None
    
    valid_brier = [r['Brier'] for r in results if r['Brier'] is not None]
    min_brier = min(valid_brier) if valid_brier else None
    
    valid_ece_pre = [r['ECE_Pre'] for r in results if r['ECE_Pre'] is not None]
    min_ece_pre = min(valid_ece_pre) if valid_ece_pre else None
    
    valid_ece_post = [r['ECE_Post'] for r in results if r['ECE_Post'] is not None]
    min_ece_post = min(valid_ece_post) if valid_ece_post else None
    
    valid_ace = [r.get('ACE_Post') for r in results if r.get('ACE_Post') is not None]
    min_ace = min(valid_ace) if valid_ace else None

    tex_lines = []
    tex_lines.append(r"\begin{table}[htbp]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{\textbf{Stage C: Multi-Horizon Opposition Model Performance with 95\% Bootstrap CIs}}")
    tex_lines.append(r"\label{tab:multi_horizon}")
    tex_lines.append(r"\renewcommand{\arraystretch}{1.2}")
    tex_lines.append(r"\resizebox{\columnwidth}{!}{%")
    tex_lines.append(r"\begin{tabular}{lcccccc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r"\textbf{Horizon} & \textbf{PR-AUC [95\% CI]} & \textbf{ROC-AUC} & \textbf{Brier} & \textbf{ECE (Pre)} & \textbf{ECE (Post)} & \textbf{ACE (Post)} \\")
    tex_lines.append(r"\midrule")
    
    for r in results:
        if r['PR-AUC'] is not None:
            raw_pr = f"{r['PR-AUC']:.3f}"
            if r['PR-AUC'] == max_pr: raw_pr = f"\\textbf{{{raw_pr}}}"
            pr_str = f"{raw_pr} {r['PR-AUC_CI']}"
        else:
            pr_str = "---"
            
        roc_str = "---"
        if r['ROC-AUC']:
            roc_str = f"\\textbf{{{r['ROC-AUC']:.3f}}}" if r['ROC-AUC'] == max_roc else f"{r['ROC-AUC']:.3f}"
            
        brier_str = "---"
        if r['Brier']:
            brier_str = f"\\textbf{{{r['Brier']:.3f}}}" if r['Brier'] == min_brier else f"{r['Brier']:.3f}"
            
        ece_pre_str = "---"
        if r['ECE_Pre'] is not None:
            ece_pre_str = f"\\textbf{{{r['ECE_Pre']:.3f}}}" if r['ECE_Pre'] == min_ece_pre else f"{r['ECE_Pre']:.3f}"
            
        ece_post_str = "---"
        if r['ECE_Post'] is not None:
            ece_post_str = f"\\textbf{{{r['ECE_Post']:.3f}}}" if r['ECE_Post'] == min_ece_post else f"{r['ECE_Post']:.3f}"
            
        ace_post_str = "---"
        if r.get('ACE_Post') is not None:
            ace_post_str = f"\\textbf{{{r['ACE_Post']:.3f}}}" if r['ACE_Post'] == min_ace else f"{r['ACE_Post']:.3f}"
            
        tex_lines.append(f"{r['horizon']} & {pr_str} & {roc_str} & {brier_str} & {ece_pre_str} & {ece_post_str} & {ace_post_str} \\\\")
        
        # Export inline macro variables for text integration
        if lib_metrics and r['PR-AUC'] is not None:
            if 'Filing' in r['horizon']:
                lib_metrics.update_metric('metricBootstrapFiling', f"{r['PR-AUC']:.3f}")
                lib_metrics.update_metric('metricBootstrapFilingCI', f"[{r['PR-AUC_CI'].replace('[','').replace(']','')}]")
                lib_metrics.update_metric('metricBootstrapFilingECE', f"{r['ECE_Pre']:.3f}")
            elif 'Notice' in r['horizon']:
                lib_metrics.update_metric('metricBootstrapNotice', f"{r['PR-AUC']:.3f}")
            elif 'Commission' in r['horizon']:
                lib_metrics.update_metric('metricBootstrapPreComm', f"{r['PR-AUC']:.3f}")
            elif 'Council' in r['horizon']:
                lib_metrics.update_metric('metricBootstrapPreCouncil', f"{r['PR-AUC']:.3f}")
                lib_metrics.update_metric('metricBootstrapPreCouncilECE', f"{r['ECE_Pre']:.3f}")
    
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"}")
    tex_lines.append(r"\end{table}")
    
    tex_path = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Tables", "multi_horizon_results.tex")
    os.makedirs(os.path.dirname(tex_path), exist_ok=True)
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex_lines))
    print(f"LaTeX table saved to {tex_path}")

if __name__ == '__main__':
    main()
