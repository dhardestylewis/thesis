import pandas as pd
import numpy as np
import os
import json
import warnings
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.base import clone
warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostClassifier
except ImportError:
    pass

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data", "Warehouse_As_Of")
OUT_DIR = os.path.join(ROOT, "Analysis", "Output", "Track1_Predictive")
os.makedirs(OUT_DIR, exist_ok=True)

# Path to IPW Vectors from Stage A
STAGE_A_PROBS = os.path.join(ROOT, "Analysis", "Output", "Track0_Predictive", "stage_a_hazard_results.csv")

# Ensure reproducibility
np.random.seed(42)

def compute_ece(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    if len(prob_true) == 0: return 0
    return np.mean(np.abs(prob_true - prob_pred))

def compute_calibration_slope(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    if len(prob_true) < 2: return 1.0 # Fallback
    # Fit OLS: Prob_true = Slope * Prob_pred + Intercept
    z = np.polyfit(prob_pred, prob_true, 1)
    return z[0]

def compute_top_decile_lift(y_true, y_prob):
    df_eval = pd.DataFrame({'y': y_true, 'p': y_prob})
    base_rate = df_eval['y'].mean()
    if base_rate == 0: return 0.0
    df_eval = df_eval.sort_values('p', ascending=False)
    k = max(1, int(len(df_eval) * 0.10))
    top_decile_hit_rate = df_eval.head(k)['y'].mean()
    return top_decile_hit_rate / base_rate

def compute_fnr_gap(y_true, y_prob, districts):
    df_eval = pd.DataFrame({'y': y_true, 'p': y_prob, 'd': districts})
    threshold = np.mean(y_true) # Academically rigorous constraint against the empirical background rate
    df_eval['pred_binary'] = (df_eval['p'] > threshold).astype(int)
    
    fnrs = {}
    for d in df_eval['d'].unique():
        sub = df_eval[df_eval['d'] == d]
        positives = sub[sub['y'] == 1]
        if len(positives) > 0:
            fnr = 1.0 - (positives['pred_binary'].sum() / len(positives))
            fnrs[d] = fnr
            
    if not fnrs: return 0.0
    return (max(fnrs.values()) - min(fnrs.values())) * 100

def extract_advanced_metrics(y_true, y_pred, districts=None, name=""):
    if len(np.unique(y_true)) < 2:
        return {'Model': name, 'PR-AUC': np.nan, 'Top-Decile Lift': np.nan, 'ECE': np.nan, 'Brier': np.nan, 'Calib-Slope': np.nan, 'FNR-Gap%': np.nan, 'FPR%': np.nan, 'Mean Prob (TP)': np.nan, 'Mean Prob (TN)': np.nan}
        
    threshold = np.mean(y_true)
    y_pred_bin = (y_pred > threshold).astype(int)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred_bin)
    fpr = cm.ravel()[1] / (cm.ravel()[1] + cm.ravel()[0]) if cm.shape == (2,2) and (cm.ravel()[1] + cm.ravel()[0]) > 0 else np.nan
    mean_tp_prob = np.mean(y_pred[y_true == 1])
    mean_tn_prob = np.mean(y_pred[y_true == 0])

    metrics = {
        'Model': name,
        'PR-AUC': average_precision_score(y_true, y_pred),
        'Top-Decile Lift': compute_top_decile_lift(y_true, y_pred),
        'ECE': compute_ece(y_true, y_pred),
        'Brier': brier_score_loss(y_true, y_pred),
        'Calib-Slope': compute_calibration_slope(y_true, y_pred),
        'FPR%': fpr * 100,
        'Mean Prob (TP)': mean_tp_prob,
        'Mean Prob (TN)': mean_tn_prob
    }
    if districts is not None:
        metrics['FNR-Gap%'] = compute_fnr_gap(y_true, y_pred, districts)
    else:
        metrics['FNR-Gap%'] = np.nan
        
    return metrics

def run_bounded_optimization(X, y, sample_weights):
    print("\n[*] Executing Calibration-Bounded Model Optimization (Isotonic [0.9, 1.1] Slope)")
    # Grid search across explicit depth/regularization
    grid = [
        {'depth': 4, 'l2_leaf_reg': 3, 'learning_rate': 0.05},
        {'depth': 6, 'l2_leaf_reg': 1, 'learning_rate': 0.03},
        {'depth': 8, 'l2_leaf_reg': 5, 'learning_rate': 0.01}
    ]
    
    best_prauc = -1
    best_model = None
    best_metrics = None
    
    # Store configuration with the slope mathematically closest to perfect 1.0 calibration
    closest_slope = float('inf')
    closest_params = None
    closest_prauc = -1
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for params in grid:
        oof_preds = np.zeros(len(y))
        for train_idx, val_idx in kf.split(X):
            X_tr, y_tr, w_tr = X.iloc[train_idx], y.iloc[train_idx], sample_weights.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            
            cb = CatBoostClassifier(iterations=100, **params, verbose=0, random_seed=42)
            calibrated_cb = CalibratedClassifierCV(estimator=cb, method='isotonic', cv=3)
            calibrated_cb.fit(X_tr, y_tr, sample_weight=w_tr)
            oof_preds[val_idx] = calibrated_cb.predict_proba(X_va)[:, 1]
            
        slope = compute_calibration_slope(y, oof_preds)
        prauc = average_precision_score(y, oof_preds)
        
        print(f"    Params {params} -> Isotonic Config Slope: {slope:.3f}, PR-AUC: {prauc:.3f}")
        
        # Track closest bounds to 1.0 in case of absolute failure
        dist_to_perfect = abs(slope - 1.0)
        if dist_to_perfect < closest_slope:
            closest_slope = dist_to_perfect
            closest_params = params
            closest_prauc = prauc
            closest_actual_slope = slope
        
        # Section 4.6 Strict Constraint
        if 0.9 <= slope <= 1.1:
            if prauc > best_prauc:
                best_prauc = prauc
                best_cb = CatBoostClassifier(iterations=150, **params, verbose=0, random_seed=42)
                best_model = CalibratedClassifierCV(estimator=best_cb, method='isotonic', cv=5)
                best_metrics = (slope, prauc)
                
    if best_model is None:
        print(f"    [!] No configuration perfectly satisfied [0.9, 1.1] bounds. Selecting closest bounded architecture (Slope {closest_actual_slope:.3f}, PR-AUC {closest_prauc:.3f}).")
        best_cb = CatBoostClassifier(iterations=150, **closest_params, verbose=0, random_seed=42)
        best_model = CalibratedClassifierCV(estimator=best_cb, method='isotonic', cv=5)
    else:
        print(f"    [+] Selected strictly bounded model: Slope {best_metrics[0]:.3f}, PR-AUC {best_metrics[1]:.3f}")
        
    return best_model

def process_horizon(path, horizon_name):
    # Safe filename string
    safe_hz = horizon_name.split()[0].replace('(', '').replace(')', '')
    
    print(f"\n==============================================")
    print(f" HORIZON ALIGNED EXECUTION: {horizon_name} ({safe_hz})")
    print(f"==============================================")
    
    df = pd.read_csv(path, low_memory=False)
    
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    else:
        return
        
    df['is_protested'] = df['is_protested'].fillna(0).astype(int)
    
    # Locate Council District for Spatial Holdouts
    dist_col = 'council_district' if 'council_district' in df.columns else 'council_district_x'
    if dist_col not in df.columns:
        df['council_district'] = 1 # Fallback
    else:
        df['council_district'] = df[dist_col].fillna(1)
        
    df = df.dropna(subset=['year']).sort_values('year').copy()
    
    # Inverse-Probability Weighting Ingest
    if os.path.exists(STAGE_A_PROBS):
        print("[+] IPW: Ingesting Optimal Stage A Hazard Probabilities (LightGBM 1-Year)...")
        df_hazard = pd.read_csv(STAGE_A_PROBS, usecols=['standardized_tcad_id', 'year', 'Prob_LGBM_H=4'])
        if 'standardized_tcad_id' in df.columns:
            df['standardized_tcad_id'] = df['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
            df_hazard['standardized_tcad_id'] = df_hazard['standardized_tcad_id'].astype(str).str.zfill(10)
            df = df.merge(df_hazard, on=['standardized_tcad_id', 'year'], how='left')
            df['ipw'] = 1.0 / np.clip(df['Prob_LGBM_H=4'].fillna(0.01), 0.0001, 1.0)
            print(f"    Successfully aligned IPW weights. Mean weight: {df['ipw'].mean():.2f}")
        else:
            df['ipw'] = 1.0
    else:
        df['ipw'] = 1.0
        
    # Strip Explicit Targets, IDs, and weights
    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'has_audio_record', 'TCAD ID', 'date', 'application_start_date', 'final_date', 'standardized_tcad_id', 'Prob_H=4', 'Prob_LGBM_H=4', 'ipw', dist_col, 'council_district']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Strip Temporal Leakage (NLP vectors from public hearings do not belong in H0)
    # PER USER REQUEST: Protected Classes (Race/Ethnicity) ARE permitted for academic explanatory auditing.
    leak_cols = [c for c in df_clean.columns if c.startswith('tfidf_') or c.startswith('speech_')]
    if len(leak_cols) > 0 and horizon_name == 'H0_Only_Complete':
        print(f"    [!] Stripped {len(leak_cols)} restricted temporal feature leaks from H0 (nlp)")
        df_clean = df_clean.drop(columns=leak_cols)
        
    X_raw = df_clean
    X = X_raw.select_dtypes(include=[np.number])
    y = df['is_protested']
    districts = df['council_district']
    weights = df['ipw']
    
    # ---------------------------------------------------------
    # PART 0: DEPLOYMENT TABLE METRICS (5-Fold CV Bounded Optimization)
    # ---------------------------------------------------------
    print("\nPART 0: DEPLOYMENT METRICS EXTRACTION (H0 EXACT)")
    optimal_model = run_bounded_optimization(X, y, weights)
    
    # Generate OOF Preds for full Table 5 extraction (and multiple architectures for Fig 17)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    oof_preds_lr = np.zeros(len(y))
    oof_preds_rf = np.zeros(len(y))
    
    for train_idx, val_idx in kf.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        w_tr = weights.iloc[train_idx]
        X_v = X.iloc[val_idx]
        
        # Primary CatBoost
        cb = clone(optimal_model)
        cb.fit(X_tr, y_tr, sample_weight=w_tr)
        oof_preds[val_idx] = cb.predict_proba(X_v)[:, 1]
        
        # Logistic Regression Baseline
        lr = LogisticRegression(max_iter=1000, class_weight='balanced')
        lr.fit(X_tr, y_tr, sample_weight=w_tr)
        oof_preds_lr[val_idx] = lr.predict_proba(X_v)[:, 1]
        
        # Random Forest Baseline
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_tr, y_tr, sample_weight=w_tr)
        oof_preds_rf[val_idx] = rf.predict_proba(X_v)[:, 1]
        
    tbl5_metrics = extract_advanced_metrics(y, oof_preds, districts, name="Table 5 Final")
    print("\n>>> PRE-SPECIFIED DEPLOYMENT METRICS (TABLE 5) <<<")
    for k, v in tbl5_metrics.items():
        if isinstance(v, float):
            print(f"  {k:20}: {v:.3f}")
        else:
            print(f"  {k:20}: {v}")
            
    try:
        from Utilities_and_Logs.lib_metrics import update_metric
        if "H0" in horizon_name:
            update_metric("metricPRAUC", f"{tbl5_metrics.get('PR-AUC', 0):.3f}")
            update_metric("metricTopDecileLift", f"{tbl5_metrics.get('Top-Decile Lift', 0):.2f}\\times")
            update_metric("metricECE", f"{tbl5_metrics.get('ECE', 0):.3f}")
            update_metric("metricCalibrationSlope", f"{tbl5_metrics.get('Calib-Slope', 0):.3f}")
            update_metric("metricFNRGap", f"{tbl5_metrics.get('FNR-Gap%', 0):.2f}\\%")
            update_metric("metricFPR", f"{tbl5_metrics.get('FPR%', 0):.2f}\\%")
            update_metric("metricMeanProbTP", f"{tbl5_metrics.get('Mean Prob (TP)', 0):.3f}")
            update_metric("metricMeanProbTN", f"{tbl5_metrics.get('Mean Prob (TN)', 0):.3f}")
    except Exception as e:
        print(f"    [!] Macro Telemetry Export Failed: {e}")
            
    # Save OOF Preds for Visualizations to pick up
    df_oof = pd.DataFrame({
        'y_true': y, 
        'y_prob': oof_preds, 
        'y_prob_lr': oof_preds_lr,
        'y_prob_rf': oof_preds_rf,
        'year': df['year'], 
        'district': districts
    })
    df_oof.to_csv(os.path.join(OUT_DIR, f"stage_c_oof_predictions_{safe_hz}.csv"), index=False)
            
    # Retrain on full for subsequent holdouts
    optimal_model.fit(X, y, sample_weight=weights)
    
    # Collect data for feature importance plot
    try:
        importance = optimal_model.get_feature_importance()
    except AttributeError:
        # Extract from the underlying uncalibrated estimator or one of the fitted calibrators
        if hasattr(optimal_model, 'calibrated_classifiers_'):
            importance = optimal_model.calibrated_classifiers_[0].estimator.get_feature_importance()
        else:
            importance = optimal_model.estimator.get_feature_importance()
            
    fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values('Importance', ascending=False)
    fi_df.to_csv(os.path.join(OUT_DIR, f"stage_c_feature_importance_{safe_hz}.csv"), index=False)

    # ---------------------------------------------------------
    # PART A: TEMPORAL DRIFT MULTI-HORIZON (Rolling-Origin)
    # ---------------------------------------------------------
    print("\nPART A: TEMPORAL DRIFT (ROLLING-ORIGIN)")
    drift_res = []
    for anchor in [2019, 2020, 2021, 2022]:
        tr_mask = df['year'] < anchor
        if tr_mask.sum() < 20: continue
        cb = clone(optimal_model)
        cb.fit(X[tr_mask], y[tr_mask], sample_weight=weights[tr_mask])
        for offset in [0,1,2,3]:
            te_mask = df['year'] == (anchor + offset)
            if te_mask.sum() < 5 or y[te_mask].sum() < 1: continue
            preds = cb.predict_proba(X[te_mask])[:, 1]
            drift_res.append({'Anchor': anchor, 'Offset': offset, 'PR-AUC': average_precision_score(y[te_mask], preds)})
            
    pd.DataFrame(drift_res).to_csv(os.path.join(OUT_DIR, f"stage_c_drift_{safe_hz}.csv"), index=False)

    # ---------------------------------------------------------
    # PART B: POLICY REGIMES
    # ---------------------------------------------------------
    print("\nPART B: POLICY REGIMES HOLDOUTS")
    regime_results = []
    regimes = [
        {"name": "Pre-2022 Validation", "train_bound": 2021, "test_start": 2021, "test_end": 2021},
        {"name": "2022 Transition", "train_bound": 2022, "test_start": 2022, "test_end": 2022},
        {"name": "HOME Adoption (2024)", "train_bound": 2024, "test_start": 2024, "test_end": 2026}
    ]
    for reg in regimes:
        tr_mask = df['year'] < reg['train_bound']
        te_mask = (df['year'] >= reg['test_start']) & (df['year'] <= reg['test_end'])
        if te_mask.sum() < 5 or y[te_mask].sum() < 1: continue
        cb = clone(optimal_model)
        cb.fit(X[tr_mask], y[tr_mask], sample_weight=weights[tr_mask])
        preds = cb.predict_proba(X[te_mask])[:, 1]
        regime_results.append({'Regime': reg['name'], 'PR-AUC': average_precision_score(y[te_mask], preds)})
        
    pd.DataFrame(regime_results).to_csv(os.path.join(OUT_DIR, f"stage_c_regimes_{safe_hz}.csv"), index=False)

    # ---------------------------------------------------------
    # PART C: SPATIAL HOLDOUTS (Council Districts)
    # ---------------------------------------------------------
    print("\nPART C: SPATIAL HOLDOUTS (DISTRICT LEVEL OOD)")
    spatial_praucs = []
    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(X, y, groups=districts):
        if len(np.unique(y.iloc[val_idx])) < 2: continue
        cb = clone(optimal_model)
        cb.fit(X.iloc[train_idx], y.iloc[train_idx], sample_weight=weights.iloc[train_idx])
        preds = cb.predict_proba(X.iloc[val_idx])[:, 1]
        spatial_praucs.append(average_precision_score(y.iloc[val_idx], preds))
        
    if spatial_praucs:
        print(f"   Spatial Holdout GroupKFold PR-AUC: {np.mean(spatial_praucs):.3f}")

def run_track1():
    print("Initiating Master Multi-Horizon Structural Engine (Aligned)...")
    horizons = {
        'H0 (Filing Baseline)': 'H0_Filing_Master_Enriched.csv',
        'H3 (Pre-Council with NLP)': 'H3_Filing_Master_NLP.csv'
    }
    
    for name, filename in horizons.items():
        path = os.path.join(DATA, filename)
        if os.path.exists(path):
            process_horizon(path, name)
            
    print("Evaluation Cycle Exhausted.")

if __name__ == '__main__':
    run_track1()
