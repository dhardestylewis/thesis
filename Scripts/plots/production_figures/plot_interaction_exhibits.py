import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

ROOT = Path(r"c:\Users\dhl\data\Thesis\thesis")
FIG_DIR = ROOT / "Thesis_Draft" / "GSAPP_Final_Submission" / "Figures" / "exhibits"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def clean_name(f):
    return (f.replace('acs2_', '').replace('acs_', '').replace('ldb_', '')
             .replace('_lag_6yr', '').replace('_', ' ').title())


def save_beeswarm(shap_matrix, X_display, title, filename):
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_matrix, X_display, max_display=15, show=False, plot_size=None)
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [+] Saved {filename}")


def save_interaction_beeswarm_grid(iv, X_sample, feature_names, title, filename, n=8):
    """
    Beeswarm grid: n×n array of scatter subplots.
    Cell (i, j) shows the distribution of SHAP interaction values Phi[i,j] across
    all cases, with points colored by the value of feature j.
    Diagonal = main effects. Off-diagonal = pairwise interactions.
    Top n features selected by total off-diagonal interaction strength.
    """
    mean_abs_int = np.abs(iv).mean(axis=0)
    off_diag = mean_abs_int.copy(); np.fill_diagonal(off_diag, 0)
    top_idx = np.argsort(off_diag.sum(axis=0))[-n:]
    top_names = [feature_names[i] for i in top_idx]

    fig, axes = plt.subplots(n, n, figsize=(n * 2.2, n * 2.0))
    fig.suptitle(title, fontsize=13, y=1.01)

    cmap = plt.cm.coolwarm

    for row, i in enumerate(top_idx):
        for col, j in enumerate(top_idx):
            ax = axes[row, col]
            y_vals = iv[:, i, j]           # interaction values for pair (i,j)
            x_jitter = np.random.default_rng(42).uniform(-0.3, 0.3, size=len(y_vals))
            color_vals = X_sample.iloc[:, j]
            norm_c = (color_vals - color_vals.min()) / (color_vals.max() - color_vals.min() + 1e-9)
            ax.scatter(x_jitter, y_vals, c=norm_c, cmap=cmap, alpha=0.4, s=4, linewidths=0)
            ax.axhline(0, color='gray', linewidth=0.4, linestyle='--')
            ax.set_xticks([]); ax.set_yticks([])
            if row == n - 1:
                ax.set_xlabel(top_names[col], fontsize=7, rotation=30, ha='right')
            if col == 0:
                ax.set_ylabel(top_names[row], fontsize=7, rotation=0, ha='right', labelpad=40)

    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  [+] Saved {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# CATBOOST: Beeswarm (main effects) + Pairwise interaction grid
# ─────────────────────────────────────────────────────────────────────────────
def run_forecasting_interaction():
    print("\n--- CatBoost Interaction Exhibits ---")
    model_path = ROOT / "Analysis" / "Output" / "Track1_Predictive" / "Models" / "stage_c_model_H0.joblib"
    data_path  = ROOT / "Data" / "Warehouse_As_Of" / "canonical" / "H0_Filing_Master_Enriched_v2.csv"
    if not model_path.exists():
        print(f"  [!] Missing: {model_path}"); return

    model = joblib.load(model_path)
    df = pd.read_csv(data_path, low_memory=False)
    drop_cols = ['year', 'is_protested', 'case_number', 'reconstructed_petition_share', 'area_pct']
    X = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    X = X.drop(columns=[c for c in drop_cols if c in X.columns]).fillna(0)
    X_sample = X.sample(n=min(600, len(X)), random_state=42)

    print("  [*] Computing CatBoost interaction values (N=600)...")
    explainer = shap.TreeExplainer(model)
    iv = explainer.shap_interaction_values(X_sample)
    if isinstance(iv, list): iv = iv[1]

    feat_names = [clean_name(c) for c in X_sample.columns]
    X_disp = X_sample.copy()
    X_disp.columns = feat_names

    # Main effects beeswarm (diagonal)
    main_effects = np.diagonal(iv, axis1=1, axis2=2)
    save_beeswarm(main_effects, X_disp,
                  "Interaction TreeSHAP: Forecasting Main Effects (Filing Date)",
                  "fig_ch4_14_forecasting_interaction_shap.pdf")

    # Pairwise interaction grid (off-diagonal)
    mean_abs_int = np.abs(iv).mean(axis=0)
    save_interaction_grid(mean_abs_int, feat_names,
                          "Interaction TreeSHAP: Pairwise Feature Interactions (Filing Date)",
                          "fig_ch4_14b_forecasting_interaction_grid.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL DML: Beeswarm + Grid via internal GRF forest (TreeSHAP)
# Falls back to PermutationExplainer (beeswarm only) if GRF not accessible
# ─────────────────────────────────────────────────────────────────────────────
def run_causal_interaction():
    print("\n--- Causal DML Interaction Exhibits ---")
    model_path = ROOT / "Data" / "Zoning_Cases" / "causal_models_production.pkl"
    if not model_path.exists():
        print(f"  [!] Missing: {model_path}"); return

    m_dict   = joblib.load(model_path)
    cf       = m_dict['cf_joint']
    hurdle   = m_dict['hurdle_model']
    features = m_dict['features']
    ex_ante  = [f for f in features if f != 'P_withdraw']
    feat_names = [clean_name(f) for f in features]

    panel_path = ROOT / "Data" / "Panel" / "cross_sectional_dml_panel.csv"
    df = pd.read_csv(panel_path, low_memory=False)
    df['P_withdraw'] = hurdle.predict_proba(df[ex_ante].fillna(0).values)[:, 1]
    X = df[features].fillna(0)
    X_sample = X.sample(n=min(200, len(X)), random_state=42)

    # Direct path confirmed via inspection:
    # cf.rlearner_model_final_._model.estimators_[0] is an econml.grf.classes.CausalForest
    # which is sklearn-compatible and accepted by shap.TreeExplainer
    causal_forest = cf.rlearner_model_final_._model.estimators_[0]

    print("  [*] Computing DML interaction values via GRF CausalForest (TreeSHAP, N=200)...")
    try:
        exp_grf = shap.TreeExplainer(causal_forest)
        iv_dml  = exp_grf.shap_interaction_values(X_sample.values)
        if isinstance(iv_dml, list): iv_dml = iv_dml[0]

        X_disp = X_sample.copy()
        X_disp.columns = feat_names

        # Main effects beeswarm
        save_beeswarm(np.diagonal(iv_dml, axis1=1, axis2=2), X_disp,
                      "Interaction TreeSHAP: DML Treatment Effect Heterogeneity",
                      "fig_ch5_14_causal_dml_interaction_shap.pdf")

        # Pairwise interaction grid (beeswarm grid)
        save_interaction_beeswarm_grid(iv_dml, X_disp, feat_names,
                                       "Interaction TreeSHAP: Pairwise Feature Interactions (DML)",
                                       "fig_ch5_14b_causal_dml_interaction_grid.pdf", n=8)
        return
    except Exception as e:
        print(f"  [!] GRF TreeSHAP failed ({e}), falling back to PermutationExplainer...")

    # Fallback: PermutationExplainer — beeswarm only
    print("  [*] Fallback: PermutationExplainer (N=50)...")
    def effect_fn(X_in):
        return cf.const_marginal_effect(X_in)[:, 0]
    X_small = X_sample.sample(n=min(50, len(X_sample)), random_state=42)
    exp_perm = shap.Explainer(effect_fn, X_small, max_evals=100)
    sv = exp_perm(X_small)
    X_disp = X_small.copy()
    X_disp.columns = feat_names
    save_beeswarm(sv.values, X_disp,
                  "Causal DML: Treatment Effect Attribution",
                  "fig_ch5_14_causal_dml_interaction_shap.pdf")
    print("  [!] Interaction grid not available via PermutationExplainer")


if __name__ == "__main__":
    # CatBoost exhibits are generated inside drift_and_archetypes.py
    # (model requires exact feature pipeline context — can't reload from CSV standalone)
    run_causal_interaction()
