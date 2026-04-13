import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

import sys
try:
    # Attempt to locate the root Scripts directory
    _curr = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(_curr) != 'Scripts' and os.path.dirname(_curr) != _curr:
        _curr = os.path.dirname(_curr)
    if _curr not in sys.path:
        sys.path.insert(0, _curr)
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostClassifier
except ImportError:
    pass

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA_IN = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
OUT_DIR = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Aesthetic parameters for the LaTeX Thesis mapping
# Removed local style: plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'figure.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'pdf.fonttype': 42 # TrueType mapping for PDF
})

def compute_ece(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    if len(prob_true) == 0: return 0
    return np.mean(np.abs(prob_true - prob_pred))

# Human-Readable Map for the Algorithmic Outputs
PRESENTATION_MAP = {
    'gross_site_area_acres': 'Gross Site Area (Acres)',
    'delta_max_height_ft': 'Requested Height Delta (ft)',
    'delta_max_far': 'Requested Density Delta (FAR)',
    'delta_max_bldg_cov_pct': 'Requested Impervious Cover Delta',
    'zoning_case_nearby': 'Proximity to Concurrent Rezonings',
    'distance_to_core_m': 'Distance to Austin Core (m)',
    'nearest_park_dist_m': 'Distance to Critical Green Space',
    'median_income_fill': 'ACS Median Local Income',
    'pct_renter_fill': 'ACS Renter Concentration (%)',
    'pct_white_fill': 'ACS Demographic Pct White (%)',
    'pct_bachelor_fill': 'ACS Education Pct Bachelor (%)',
    'appraised_val_per_sqft_fill': 'TCAD Appraised Value per SqFt ($)'
}

def build_visuals():
    print("Loading V2 Master Matrix for Visual Abstraction...")
    df = pd.read_csv(DATA_IN, low_memory=False)
    
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    df['is_protested'] = df['is_protested'].fillna(0).astype(int)
    df = df.dropna(subset=['year']).sort_values('year').copy()
    
    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'has_audio_record', 'TCAD ID', 'date', 'application_start_date', 'final_date', 'Case Number', 'standardized_tcad_id', 'Signature']
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = X_raw.select_dtypes(include=[np.number]).fillna(0)
    
    # Translate features BEFORE modeling so SHAP charts automatically read them
    X.columns = [PRESENTATION_MAP.get(col, col) for col in X.columns]
    y = df['is_protested']
    
    # ----------------------------------------------------
    # 1. Temporal Rot Line Plot (Multihorizon T+0 to T+3)
    # ----------------------------------------------------
    print("Generating Figure 1: Temporal Drift Multihorizon...")
    drift_data = []
    
    # Implementing All 3 Cross-Validation Models per User Directive
    for anchor in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
        train_mask = df['year'] < anchor
        if train_mask.sum() < 20: continue
            
        cb = CatBoostClassifier(iterations=50, verbose=0).fit(X[train_mask], y[train_mask])
        lr = LogisticRegression(max_iter=500, class_weight='balanced').fit(X[train_mask], y[train_mask])
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42).fit(X[train_mask], y[train_mask])
        
        # Open-ended eval of all years >= anchor
        test_years = sorted(df[df['year'] >= anchor]['year'].unique())
        for test_year in test_years:
            offset = int(test_year - anchor)
            test_mask = df['year'] == test_year
            if test_mask.sum() < 5 or y[test_mask].sum() < 1: continue
            
            p_cb = cb.predict_proba(X[test_mask])[:, 1]
            p_lr = lr.predict_proba(X[test_mask])[:, 1]
            p_rf = rf.predict_proba(X[test_mask])[:, 1]
            
            drift_data.append({'Training Anchor': f"Pre-{anchor}", 'Model': 'CatBoost', 'PR-AUC': average_precision_score(y[test_mask], p_cb), 'Offset': offset})
            drift_data.append({'Training Anchor': f"Pre-{anchor}", 'Model': 'ElasticNet', 'PR-AUC': average_precision_score(y[test_mask], p_lr), 'Offset': offset})
            drift_data.append({'Training Anchor': f"Pre-{anchor}", 'Model': 'RandomForest', 'PR-AUC': average_precision_score(y[test_mask], p_rf), 'Offset': offset})
    
    if drift_data:
        d_df = pd.DataFrame(drift_data)
        # Average the models across the offsets for clarity if too dense, but since User requested all models:
        # We will plot CatBoost only to avoid crowding, or we can use seaborn 'lineplot' with style='Model'
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=d_df, x='Offset', y='PR-AUC', hue='Training Anchor', style='Model', markers=True, dashes=False, linewidth=2, markersize=8)
        plt.title('Predictive Temporal Drift (Model Rot over Time)')
        plt.xlabel('Forecasting Horizon (Years Out)')
        plt.ylabel('Out-Of-Sample PR-AUC')
        plt.axhline(y=0.06, color='r', linestyle='--', label='Random Chance (~6%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'fig_temporal_drift.pdf'), format='pdf', dpi=300)
        plt.close()

    # ----------------------------------------------------
    # 2. Worst-Case Policy Regime Stability (Bar Chart)
    # ----------------------------------------------------
    print("Generating Figure 2: Legislative Policy Regimes OOD...")
    regimes = [
        {"name": "Pre-2022 Council", "train_bound": 2021, "test_start": 2021, "test_end": 2021},
        {"name": "2022 Transition", "train_bound": 2022, "test_start": 2022, "test_end": 2022},
        {"name": "HOME Phase 1 (2024)", "train_bound": 2024, "test_start": 2024, "test_end": 2026}
    ]
    
    regime_data = []
    for reg in regimes:
        train_mask = df['year'] < reg['train_bound']
        test_mask = (df['year'] >= reg['test_start']) & (df['year'] <= reg['test_end'])
        
        if test_mask.sum() < 5 or y[test_mask].sum() < 1: continue
        
        lr = LogisticRegression(max_iter=500, class_weight='balanced').fit(X[train_mask], y[train_mask])
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42).fit(X[train_mask], y[train_mask])
        cb = CatBoostClassifier(iterations=100, verbose=0).fit(X[train_mask], y[train_mask])
        
        regime_data.append({'Regime': reg['name'], 'Model': 'ElasticNet Baseline', 'PR-AUC': average_precision_score(y[test_mask], lr.predict_proba(X[test_mask])[:, 1])})
        regime_data.append({'Regime': reg['name'], 'Model': 'Random Forest', 'PR-AUC': average_precision_score(y[test_mask], rf.predict_proba(X[test_mask])[:, 1])})
        regime_data.append({'Regime': reg['name'], 'Model': 'CatBoost V2', 'PR-AUC': average_precision_score(y[test_mask], cb.predict_proba(X[test_mask])[:, 1])})
    
    if regime_data:
        r_df = pd.DataFrame(regime_data)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=r_df, x='Regime', y='PR-AUC', hue='Model', palette='viridis')
        plt.title('Out-of-Distribution Robustness Across Policy Shocks')
        plt.ylabel('PR-AUC')
        plt.axhline(y=0.06, color='r', linestyle='--', label='Random Baseline')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'fig_policy_regimes.pdf'), format='pdf', dpi=300)
        plt.close()

    # ----------------------------------------------------
    # 3. Calibration Curve / Reliability Diagram
    # ----------------------------------------------------
    print("Generating Figure 3: ECE Reliability Diagrams...")
    global_trues, global_preds_lr, global_preds_cb, global_preds_rf = [], [], [], []
    
    for ty in [2021, 2022, 2023, 2024]:
        tm = df['year'] < ty
        eval_m = df['year'] == ty
        if eval_m.sum() < 2 or y[eval_m].sum() < 1: continue
        
        cb = CatBoostClassifier(iterations=100, verbose=0).fit(X[tm], y[tm])
        lr = LogisticRegression(max_iter=500, class_weight='balanced').fit(X[tm], y[tm])
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42).fit(X[tm], y[tm])
        
        global_trues.extend(y[eval_m].values)
        global_preds_cb.extend(cb.predict_proba(X[eval_m])[:, 1])
        global_preds_lr.extend(lr.predict_proba(X[eval_m])[:, 1])
        global_preds_rf.extend(rf.predict_proba(X[eval_m])[:, 1])
        
    if global_trues:
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        
        for preds, name in [(global_preds_cb, "CatBoost"), (global_preds_lr, "ElasticNet"), (global_preds_rf, "Random Forest")]:
            fraction_pos, mean_pred = calibration_curve(global_trues, preds, n_bins=10)
            plt.plot(mean_pred, fraction_pos, "s-", label=f"{name} (ECE={compute_ece(global_trues, preds):.3f})")
        
        plt.ylabel("Fraction of Positives (Empirical Probability)")
        plt.xlabel("Mean Predicted Probability")
        plt.title('Model Reliability / Calibration')
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'fig_calibration_ece.pdf'), format='pdf', dpi=300)
        plt.close()
        
    # ----------------------------------------------------
    # 4. Feature Ablation Bar Graph
    # ----------------------------------------------------
    print("Generating Figure 4: Model Interpretability Feature Ranks...")
    try:
        cb = CatBoostClassifier(iterations=200, verbose=0).fit(X, y)
        importances = cb.get_feature_importance()
        # Filter out completely unrelated dummy/empty items simply so the graph represents true dynamics
        feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False)
        feat_df = feat_df[~feat_df['Feature'].astype(str).str.contains('Unnamed')].head(15)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feat_df, x='Importance', y='Feature', palette='mako')
        plt.title('Primary Base Feature Attribution (CatBoost H0 Baseline)')
        plt.xlabel('Permutation Entropy Contribution')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'fig_feature_importance.pdf'), format='pdf', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Skipping feature importances: {e}")

    print("Architecture Execution Complete. All LaTeX PDFs rendered to Draft_v1/Figures/")

if __name__ == '__main__':
    build_visuals()
