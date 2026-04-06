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

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression

class AnchorRegressionLPM(BaseEstimator, ClassifierMixin):
    """
    True Anchor Regression formulated as a Linear Probability Model (LPM) 
    for out-of-distribution causal inference (Rothenhausler et al. 2021).
    X_anc = X + (sqrt(gamma) - 1) * P_A X
    y_anc = y + (sqrt(gamma) - 1) * P_A y
    """
    _estimator_type = "classifier"
    
    def __init__(self, gamma=10.0, n_anchors=None):
        self.gamma = gamma
        self.n_anchors = n_anchors
        self.model = LinearRegression(fit_intercept=True)
        self.proj_X = LinearRegression(fit_intercept=False)
        self.proj_y = LinearRegression(fit_intercept=False)
        
    def fit(self, X_transformed, y, sample_weight=None):
        import pandas as pd
        if isinstance(X_transformed, pd.DataFrame): X_transformed = X_transformed.values
        if isinstance(y, (pd.Series, pd.DataFrame)): y = y.values
            
        A = X_transformed[:, :self.n_anchors]
        
        self.proj_X.fit(A, X_transformed, sample_weight=sample_weight)
        self.proj_y.fit(A, y, sample_weight=sample_weight)
        
        X_P = self.proj_X.predict(A)
        y_P = self.proj_y.predict(A)
        
        factor = np.sqrt(self.gamma) - 1.0
        X_anc = X_transformed + factor * X_P
        y_anc = y + factor * y_P
        
        self.model.fit(X_anc, y_anc, sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X_transformed):
        import pandas as pd
        if isinstance(X_transformed, pd.DataFrame): X_transformed = X_transformed.values
        preds = self.model.predict(X_transformed)
        preds = np.clip(preds, 0, 1) # Bound probability 0-1
        return np.vstack([1 - preds, preds]).T
        
    def predict(self, X_transformed):
        return (self.predict_proba(X_transformed)[:, 1] > 0.5).astype(int)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from artifact_registry import ROOT_DIR, DATA_WAREHOUSE_DIR, TRACK1_DIR, TraceabilityRegistry as AR

ROOT = str(ROOT_DIR)
DATA = str(DATA_WAREHOUSE_DIR)
OUT_DIR = str(TRACK1_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# Path to IPW Vectors from Stage A
STAGE_A_PROBS = str(AR.STAGE_A_HAZARD_RESULTS)

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

def compute_fpr_gap(y_true, y_prob, districts):
    """Max−Min False Positive Rate gap across council districts (Chouldechova complement to FNR gap)."""
    df_eval = pd.DataFrame({'y': y_true, 'p': y_prob, 'd': districts})
    threshold = np.mean(y_true)
    df_eval['pred_binary'] = (df_eval['p'] > threshold).astype(int)
    
    fprs = {}
    for d in df_eval['d'].unique():
        sub = df_eval[df_eval['d'] == d]
        negatives = sub[sub['y'] == 0]
        if len(negatives) > 0:
            fpr = negatives['pred_binary'].sum() / len(negatives)
            fprs[d] = fpr
            
    if not fprs: return 0.0
    return (max(fprs.values()) - min(fprs.values())) * 100

def extract_advanced_metrics(y_true, y_pred, districts=None, name=""):
    if len(np.unique(y_true)) < 2:
        return {'Model': name, 'PR-AUC': np.nan, 'Top-Decile Lift': np.nan, 'ECE': np.nan, 'Brier': np.nan, 'Calib-Slope': np.nan, 'FNR-Gap%': np.nan, 'FPR-Gap%': np.nan, 'FPR%': np.nan, 'Mean Prob (TP)': np.nan, 'Mean Prob (TN)': np.nan}
        
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
        metrics['FPR-Gap%'] = compute_fpr_gap(y_true, y_pred, districts)
    else:
        metrics['FNR-Gap%'] = np.nan
        metrics['FPR-Gap%'] = np.nan
        
    return metrics

def run_bounded_optimization(X, y, sample_weights):
    print("\n[*] Executing Calibration-Bounded Model Optimization (Adaptive Search for [0.9, 1.1] Slope)")
    
    # Starting point
    current_params = {'depth': 6, 'l2_leaf_reg': 3, 'learning_rate': 0.03}
    
    best_prauc = -1
    best_model = None
    best_metrics = None
    
    closest_slope_dist = float('inf')
    closest_params = None
    closest_prauc = -1
    closest_actual_slope = None
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Adaptive Search parameters
    max_steps = 10
    step_count = 0
    
    while step_count < max_steps:
        step_count += 1
        oof_preds = np.zeros(len(y))
        
        for train_idx, val_idx in kf.split(X):
            X_tr, y_tr, w_tr = X.iloc[train_idx], y.iloc[train_idx], sample_weights.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            
            cb = CatBoostClassifier(iterations=100, **current_params, verbose=0, random_seed=42)
            calibrated_cb = CalibratedClassifierCV(estimator=cb, method='sigmoid', cv=3)
            calibrated_cb.fit(X_tr, y_tr, sample_weight=w_tr)
            oof_preds[val_idx] = calibrated_cb.predict_proba(X_va)[:, 1]
            
        slope = compute_calibration_slope(y, oof_preds)
        prauc = average_precision_score(y, oof_preds)
        
        print(f"    Step {step_count}: Params {current_params} -> Slope: {slope:.3f}, PR-AUC: {prauc:.3f}")
        
        dist_to_perfect = abs(slope - 1.0)
        if dist_to_perfect < closest_slope_dist:
            closest_slope_dist = dist_to_perfect
            closest_params = current_params.copy()
            closest_prauc = prauc
            closest_actual_slope = slope
        
        # Section 4.6 Strict Constraint Validation
        if 0.9 <= slope <= 1.1:
            print(f"    [+] Strict calibration bound satisfied at step {step_count}!")
            best_prauc = prauc
            best_cb = CatBoostClassifier(iterations=150, **current_params, verbose=0, random_seed=42)
            best_model = CalibratedClassifierCV(estimator=best_cb, method='sigmoid', cv=5)
            best_metrics = (slope, prauc)
            break # Early stopping! We landed in the desired range!
            
        # Adaptive Directional Shift
        # If slope > 1.1, the model is underconfident (probabilities are too squashed) -> Decrease regularization, increase depth
        # If slope < 0.9, the model is overconfident (probabilities are too extreme) -> Increase regularization, decrease depth
        new_params = current_params.copy()
        if slope > 1.1:
            new_params['l2_leaf_reg'] = max(1, current_params['l2_leaf_reg'] - 1)
            new_params['depth'] = min(8, current_params['depth'] + 1)
        else:
            new_params['l2_leaf_reg'] = current_params['l2_leaf_reg'] + 2
            new_params['depth'] = max(3, current_params['depth'] - 1)
            
        # Prevent infinite loops if params hit boundaries and stop changing
        if new_params == current_params:
            print("    [!] Parameter boundaries reached. Halting adaptive search.")
            break
            
        current_params = new_params
                
    if best_model is None:
        print(f"    [!] Search exhausted. Selecting closest bounded architecture (Slope {closest_actual_slope:.3f}, PR-AUC {closest_prauc:.3f}).")
        best_cb = CatBoostClassifier(iterations=150, **closest_params, verbose=0, random_seed=42)
        best_model = CalibratedClassifierCV(estimator=best_cb, method='sigmoid', cv=5)
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
        print("[+] IPW: Ingesting Optimal Stage A Hazard Probabilities...")
        df_hazard = pd.read_csv(STAGE_A_PROBS, usecols=['standardized_tcad_id', 'year', 'Prob_Optimal_H=4'])
        if 'standardized_tcad_id' in df.columns:
            df['standardized_tcad_id'] = df['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
            df_hazard['standardized_tcad_id'] = df_hazard['standardized_tcad_id'].astype(str).str.zfill(10)
            df = df.merge(df_hazard, on=['standardized_tcad_id', 'year'], how='left')
            df['ipw'] = 1.0 / np.clip(df['Prob_Optimal_H=4'].fillna(0.01), 0.0001, 1.0)
            print(f"    Successfully aligned IPW weights. Mean weight: {df['ipw'].mean():.2f}")
        else:
            df['ipw'] = 1.0
    else:
        df['ipw'] = 1.0
        
    # Strip Explicit Targets, IDs, and weights
    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'has_audio_record', 'TCAD ID', 'date', 'application_start_date', 'final_date', 'standardized_tcad_id', 'Prob_H=4', 'Prob_LGBM_H=4', 'Prob_CB_H=4', 'Prob_Optimal_H=4', 'ipw', dist_col, 'council_district']
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
    from lightgbm import LGBMClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    oof_preds_lr = np.zeros(len(y))
    oof_preds_rf = np.zeros(len(y))
    oof_preds_lgbm = np.zeros(len(y))
    oof_preds_spatial_lr = np.zeros(len(y))
    oof_preds_anchor = np.zeros(len(y))
    
    # Pre-configure explicit environment dataframes for the benchmark models
    X_spatial = X.copy()
    X_spatial['council_district'] = districts.astype(str)
    
    X_anchor = X.copy()
    X_anchor['council_district'] = districts.astype(str)
    X_anchor['year'] = df['year'].astype(str)
    
    spatial_prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['council_district']),
        ('num', SimpleImputer(strategy='median'), X.columns)
    ])
    
    anchor_prep = ColumnTransformer([
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['council_district', 'year']),
        ('num', SimpleImputer(strategy='median'), X.columns)
    ])

    for train_idx, val_idx in kf.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        w_tr = weights.iloc[train_idx]
        X_v = X.iloc[val_idx]
        
        X_sp_tr, X_sp_v = X_spatial.iloc[train_idx], X_spatial.iloc[val_idx]
        X_anc_tr, X_anc_v = X_anchor.iloc[train_idx], X_anchor.iloc[val_idx]
        
        # Primary CatBoost (Natively handles missing data)
        cb = clone(optimal_model)
        cb.fit(X_tr, y_tr, sample_weight=w_tr)
        oof_preds[val_idx] = cb.predict_proba(X_v)[:, 1]
        
        # Logistic Regression Baseline (Requires Imputation)
        lr = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        lr_cal = CalibratedClassifierCV(estimator=lr, method='sigmoid', cv=3)
        lr_cal.fit(X_tr, y_tr, model__sample_weight=w_tr)
        oof_preds_lr[val_idx] = lr_cal.predict_proba(X_v)[:, 1]
        
        # Random Forest Baseline (Requires Imputation)
        rf = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42))
        ])
        rf_cal = CalibratedClassifierCV(estimator=rf, method='sigmoid', cv=3)
        rf_cal.fit(X_tr, y_tr, model__sample_weight=w_tr)
        oof_preds_rf[val_idx] = rf_cal.predict_proba(X_v)[:, 1]
        
        # LightGBM Baseline (Requires Imputation)
        lgbm = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1))
        ])
        lgbm_cal = CalibratedClassifierCV(estimator=lgbm, method='sigmoid', cv=3)
        lgbm_cal.fit(X_tr, y_tr, model__sample_weight=w_tr)
        oof_preds_lgbm[val_idx] = lgbm_cal.predict_proba(X_v)[:, 1]
        
        # Spatial Fixed-Effects Logistic (Domain Benchmark)
        splr = Pipeline([
            ('prep', spatial_prep),
            ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        splr_cal = CalibratedClassifierCV(estimator=splr, method='sigmoid', cv=3)
        splr_cal.fit(X_sp_tr, y_tr, model__sample_weight=w_tr)
        oof_preds_spatial_lr[val_idx] = splr_cal.predict_proba(X_sp_v)[:, 1]
        
        # True Anchor Regression (Causal Invariance Benchmark)
        # Manual Isotonic calibration because CalibratedClassifierCV
        # cannot introspect the internal LinearRegression estimator type.
        from sklearn.isotonic import IsotonicRegression
        X_anc_tr_prep = anchor_prep.fit_transform(X_anc_tr)
        num_anchors = len(anchor_prep.named_transformers_['cat'].get_feature_names_out())
        
        anc = AnchorRegressionLPM(gamma=10.0, n_anchors=num_anchors)
        anc.fit(X_anc_tr_prep, y_tr, sample_weight=w_tr)
        
        # Get raw LPM predictions on training fold for isotonic fit
        raw_train_preds = anc.predict_proba(X_anc_tr_prep)[:, 1]
        iso_anc = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso_anc.fit(raw_train_preds, y_tr)
        
        X_anc_v_prep = anchor_prep.transform(X_anc_v)
        raw_val_preds = anc.predict_proba(X_anc_v_prep)[:, 1]
        oof_preds_anchor[val_idx] = iso_anc.predict(raw_val_preds)
        
    tbl5_metrics = extract_advanced_metrics(y, oof_preds, districts, name="Table 5 Final")
    print("\n>>> PRE-SPECIFIED DEPLOYMENT METRICS (TABLE 5) <<<")
    for k, v in tbl5_metrics.items():
        if isinstance(v, float):
            print(f"  {k:20}: {v:.3f}")
        else:
            print(f"  {k:20}: {v}")
            
    try:
        import sys
        module_path = os.path.join(ROOT, 'Analysis', 'Scripts', 'Modeling')
        if module_path not in sys.path:
            sys.path.append(module_path)
            
        from Utilities_and_Logs.lib_metrics import update_metric
        if "H0" in horizon_name:
            update_metric("metricPRAUC", f"{tbl5_metrics.get('PR-AUC', 0):.3f}")
            update_metric("metricTopDecileLift", f"{tbl5_metrics.get('Top-Decile Lift', 0):.2f}$\\times$")
            update_metric("metricECE", f"{tbl5_metrics.get('ECE', 0):.3f}")
            update_metric("metricCalibrationSlope", f"{tbl5_metrics.get('Calib-Slope', 0):.3f}")
            update_metric("metricFNRGap", f"{tbl5_metrics.get('FNR-Gap%', 0):.2f}\\%")
            update_metric("metricFPRGap", f"{tbl5_metrics.get('FPR-Gap%', 0):.2f}\\%")
            update_metric("metricFPR", f"{tbl5_metrics.get('FPR%', 0):.2f}\\%")
            update_metric("metricMeanProbTP", f"{tbl5_metrics.get('Mean Prob (TP)', 0):.3f}")
            update_metric("metricMeanProbTN", f"{tbl5_metrics.get('Mean Prob (TN)', 0):.3f}")
    except Exception as e:
        print(f"    [!] Macro Telemetry Export Failed: {e}")
            
    # Save OOF Preds for Visualizations to pick up
    df_oof = pd.DataFrame({
        'standardized_tcad_id': df.get('standardized_tcad_id', None),
        'y_true': y, 
        'y_prob': oof_preds, 
        'y_prob_lr': oof_preds_lr,
        'y_prob_rf': oof_preds_rf,
        'y_prob_lgbm': oof_preds_lgbm,
        'y_prob_spatial_lr': oof_preds_spatial_lr,
        'y_prob_anchor': oof_preds_anchor,
        'year': df['year'], 
        'district': districts
    })
    df_oof.to_csv(str(AR.stage_c_oof(safe_hz)), index=False)
            
    # Retrain on full for subsequent holdouts
    optimal_model.fit(X, y, sample_weight=weights)
    
    import joblib
    out_joblib = str(AR.stage_c_model(safe_hz))
    try:
        joblib.dump(optimal_model, out_joblib)
        print(f"    [+] Saved fully trained model artifact to {out_joblib}")
    except Exception as e:
        print(f"    [-] Failed to save model artifact: {e}")
    
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
    fi_df.to_csv(str(AR.stage_c_feature_importance(safe_hz)), index=False)

    # ---------------------------------------------------------
    # PART A: TEMPORAL DRIFT MULTI-HORIZON (Rolling-Origin)
    # ---------------------------------------------------------
    print("\nPART A: TEMPORAL DRIFT (ROLLING-ORIGIN)")
    drift_res = []
    for anchor in [2019, 2020, 2021, 2022, 2023]:
        tr_mask = df['year'] < anchor
        if tr_mask.sum() < 20: continue
        cb = clone(optimal_model)
        cb.fit(X[tr_mask], y[tr_mask], sample_weight=weights[tr_mask])
        for offset in [0,1,2,3]:
            te_mask = df['year'] == (anchor + offset)
            if te_mask.sum() < 5 or y[te_mask].sum() < 1: continue
            preds = cb.predict_proba(X[te_mask])[:, 1]
            drift_res.append({'Anchor': anchor, 'Offset': offset, 'PR-AUC': average_precision_score(y[te_mask], preds)})
            
    pd.DataFrame(drift_res).to_csv(str(AR.stage_c_drift(safe_hz)), index=False)

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
        
    pd.DataFrame(regime_results).to_csv(str(AR.stage_c_regimes(safe_hz)), index=False)

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

    # ---------------------------------------------------------
    # PART D: DEMOGRAPHIC HOLDOUTS (INCOME & RACE)
    # ---------------------------------------------------------
    print("\nPART D: DEMOGRAPHIC HOLDOUTS (OOD Fairness Audit)")
    if 'acs_median_household_income' in df.columns:
        med_inc = df['acs_median_household_income'].median()
        tr_mask = df['acs_median_household_income'] >= med_inc # Train exclusively on High-Wealth properties
        te_mask = df['acs_median_household_income'] < med_inc  # Test strictly on lower-income demographics
        if te_mask.sum() > 5 and y[te_mask].sum() > 0 and tr_mask.sum() > 5:
            cb_demo = clone(optimal_model)
            cb_demo.fit(X[tr_mask], y[tr_mask], sample_weight=weights[tr_mask])
            preds_demo = cb_demo.predict_proba(X[te_mask])[:, 1]
            print(f"   Demographic Holdout (High-to-Low Wealth) PR-AUC: {average_precision_score(y[te_mask], preds_demo):.3f}")
    else:
        print("   Demographic Holdout skipped: 'acs_median_household_income' scalar absent in explicit projection.")

    # ---------------------------------------------------------
    # PART E: MORPHOLOGICAL HOLDOUTS (CORE VS PERIPHERY)
    # ---------------------------------------------------------
    print("\nPART E: MORPHOLOGICAL HOLDOUTS (CORE VS SUBURBS)")
    if 'council_district' in df.columns:
        # Use District 9 (Downtown/Central) as Core, remaining as Suburbs
        tr_mask = df['council_district'] == 9  
        te_mask = df['council_district'] != 9  
        if te_mask.sum() > 5 and y[te_mask].sum() > 0 and tr_mask.sum() > 5:
            cb_morph = clone(optimal_model)
            cb_morph.fit(X[tr_mask], y[tr_mask], sample_weight=weights[tr_mask])
            preds_morph = cb_morph.predict_proba(X[te_mask])[:, 1]
            print(f"   Morphological Holdout (Core-to-Suburb) PR-AUC: {average_precision_score(y[te_mask], preds_morph):.3f}")
        else:
            print("   Morphological Holdout skipped: Insufficient target density.")
    else:
        print("   Morphological Holdout skipped: 'council_district' absent.")

    # ---------------------------------------------------------
    # PART F: ZONING TYPOLOGY HOLDOUTS
    # ---------------------------------------------------------
    print("\nPART F: ZONING TYPOLOGY HOLDOUTS (RESIDENTIAL VS COMMERCIAL)")
    if 'property_category_code' in df.columns:
        tr_mask = df['property_category_code'].astype(str).str.startswith('A', na=False) # Res
        te_mask = df['property_category_code'].astype(str).str.startswith('F', na=False) # Comm
        if te_mask.sum() > 5 and y[te_mask].sum() > 0 and tr_mask.sum() > 5:
            cb_zone = clone(optimal_model)
            cb_zone.fit(X[tr_mask], y[tr_mask], sample_weight=weights[tr_mask])
            preds_zone = cb_zone.predict_proba(X[te_mask])[:, 1]
            print(f"   Zoning Typology (Residential-to-Commercial) PR-AUC: {average_precision_score(y[te_mask], preds_zone):.3f}")
    else:
        print("   Zoning Typology skipped: 'property_category_code' string absent.")

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
