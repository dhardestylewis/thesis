"""
plot_stage_b_seed_audit.py — Stage B 6-Tier Typology Multi-Seed Stability
=========================================================================
Runs the Stage B multi-class zoning typology classifier across 10 seeds
and 4 model architectures, reporting macro-F1, weighted-F1, accuracy,
and per-class F1 stability.
"""
import os, sys, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import lightgbm as lgb_lib

warnings.filterwarnings("ignore")

try:
    _curr = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(_curr) != 'Scripts' and os.path.dirname(_curr) != _curr:
        _curr = os.path.dirname(_curr)
    if _curr not in sys.path:
        sys.path.insert(0, _curr)
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

ROOT = r"C:\Users\dhl\data\thesis\thesis"
SEEDS = list(range(100))
TIER_LABELS = ["PUD / large negotiated project", "mixed-use", "multifamily",
               "missing-middle", "discretionary rezoning", "by-right infill"]
SHORT_LABELS = ["PUD", "Mixed-Use", "Multi-\nfamily", "Missing\nMiddle", "Rezon-\ning", "By-Right"]

def derive_6_tier(row):
    far = row['delta_max_far']
    acres = row['gross_site_area_acres']
    if acres > 3 and far > 1.5: return "PUD / large negotiated project"
    if far > 1.0: return "mixed-use"
    if far > 0.5: return "multifamily"
    if far > 0.1 and acres < 1.0: return "missing-middle"
    if far > 0: return "discretionary rezoning"
    return "by-right infill"

def run():
    out_dir = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter4")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv"),
                     low_memory=False)
    df = df.dropna(subset=['delta_max_far', 'gross_site_area_acres', 'year'])
    df['6_tier_class'] = df.apply(derive_6_tier, axis=1)

    # Use more features than the original (which only used 2)
    feature_candidates = ['gross_site_area_acres', 'year', 
                          'latitude', 'longitude', 'ldb_appraised_value',
                          'std_appraised_value', 'median_appraised_value',
                          'acs_median_home_value', 'acs_median_rent', 'acs_owner_occupied_pct']
    features = [f for f in feature_candidates if f in df.columns]
    X = df[features].fillna(0)
    y = df['6_tier_class']

    models_cfg = ['CatBoost', 'Random Forest', 'LightGBM', 'Logistic Regression']
    colors = {'CatBoost': '#8B0000', 'Random Forest': '#888888',
              'LightGBM': '#006400', 'Logistic Regression': '#FF8C00'}

    results = {m: {'macro_f1': [], 'weighted_f1': [], 'accuracy': [], 'log_loss': []} for m in models_cfg}
    per_class = {m: {t: [] for t in TIER_LABELS} for m in models_cfg}

    print("=" * 65)
    print(" Stage B: 6-Tier Zoning Typology — 10 Seeds x 4 Architectures")
    print("=" * 65)

    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)

        # CatBoost
        cb = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05,
                                loss_function='MultiClass', verbose=0, random_state=seed)
        cb.fit(X_train, y_train)
        p_cb = cb.predict(X_test).flatten()
        pp_cb = cb.predict_proba(X_test)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed,
                                    class_weight='balanced', n_jobs=-1)
        rf.fit(X_train, y_train)
        p_rf = rf.predict(X_test)
        pp_rf = rf.predict_proba(X_test)

        # LightGBM
        lgbm = lgb_lib.LGBMClassifier(n_estimators=200, random_state=seed, verbose=-1, importance_type='gain')
        lgbm.fit(X_train, y_train)
        p_lgb = lgbm.predict(X_test)
        pp_lgb = lgbm.predict_proba(X_test)

        # Logistic Regression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X_train); Xte_s = sc.transform(X_test)
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)
        lr.fit(Xtr_s, y_train)
        p_lr = lr.predict(Xte_s); pp_lr = lr.predict_proba(Xte_s)

        for mname, preds, probs in [('CatBoost', p_cb, pp_cb), ('Random Forest', p_rf, pp_rf),
                                     ('LightGBM', p_lgb, pp_lgb), ('Logistic Regression', p_lr, pp_lr)]:
            results[mname]['macro_f1'].append(f1_score(y_test, preds, average='macro', zero_division=0))
            results[mname]['weighted_f1'].append(f1_score(y_test, preds, average='weighted', zero_division=0))
            results[mname]['accuracy'].append(accuracy_score(y_test, preds))
            try: results[mname]['log_loss'].append(log_loss(y_test, probs))
            except: results[mname]['log_loss'].append(np.nan)
            for tier in TIER_LABELS:
                per_class[mname][tier].append(f1_score(y_test == tier, preds == tier, zero_division=0))

        print(f"  Seed {seed}: " + " | ".join(f"{m}={results[m]['macro_f1'][-1]:.3f}" for m in models_cfg))
        gc.collect()

    # Plots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for ax, ml, mk in zip(axes, ['Macro-F1', 'Weighted-F1', 'Accuracy', 'Log-Loss'], ['macro_f1', 'weighted_f1', 'accuracy', 'log_loss']):
        bp = ax.boxplot([results[m][mk] for m in models_cfg], labels=[m.replace(' ', '\n') for m in models_cfg], patch_artist=True, widths=0.6)
        for patch, m in zip(bp['boxes'], models_cfg): patch.set_facecolor(colors[m]); patch.set_alpha(0.7)
        for med in bp['medians']: med.set_color('white'); med.set_linewidth(2)
        ax.set_title(ml, fontsize=13, pad=8); ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.suptitle("Stage B: 6-Tier Zoning Typology Seed Stability (10 Seeds x 4 Models)", fontsize=14, y=1.01)
    plt.tight_layout(); p1 = os.path.join(out_dir, "fig_stage_b_seed.pdf")
    plt.savefig(p1, bbox_inches='tight', dpi=150); plt.close()

    # Per-Class Boxplot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10)); axes = axes.flatten()
    for i, tier in enumerate(TIER_LABELS):
        ax = axes[i]; data = [per_class[m][tier] for m in models_cfg]
        bp = ax.boxplot(data, labels=[m.replace(' ', '\n') for m in models_cfg], patch_artist=True, widths=0.5)
        for patch, m in zip(bp['boxes'], models_cfg): patch.set_facecolor(colors[m]); patch.set_alpha(0.7)
        for med in bp['medians']: med.set_color('white'); med.set_linewidth(1.5)
        for k, m in enumerate(models_cfg):
            jitter = np.random.normal(0, 0.04, size=len(per_class[m][tier]))
            ax.scatter(np.ones(len(per_class[m][tier]))*(k+1)+jitter, per_class[m][tier], color='black', alpha=0.3, s=10, zorder=5)
        ax.set_title(f"Typology: {tier}\nF1 stability", fontsize=11, fontweight='bold')
        ax.set_ylim(-0.05, 1.05); ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.suptitle("Stage B Per-Class F1 Stability: 10 Seeds x 4 Architectures", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); p3 = os.path.join(out_dir, "fig_stage_b_perclass_boxplot.pdf")
    plt.savefig(p3, bbox_inches='tight', dpi=150); plt.close()

    # Heatmap (Legacy/Appendix)
    fig, axes = plt.subplots(1, len(models_cfg), figsize=(5*len(models_cfg), 5))
    for ax, mname in zip(axes, models_cfg):
        matrix = np.array([per_class[mname][t] for t in TIER_LABELS])
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_yticks(range(len(TIER_LABELS))); ax.set_yticklabels(SHORT_LABELS, fontsize=9)
        ax.set_xticks(range(len(SEEDS))); ax.set_xticklabels([f"S{s}" for s in SEEDS], fontsize=8)
        ax.set_xlabel("Seed"); ax.set_title(mname, fontsize=12, fontweight='bold')
        for i in range(len(TIER_LABELS)):
            for j in range(len(SEEDS)):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha='center', va='center', fontsize=7, color='white' if matrix[i,j] < 0.4 else 'black')
    plt.tight_layout(); p2 = os.path.join(out_dir, "fig_stage_b_perclass.pdf")
    plt.savefig(p2, bbox_inches='tight', dpi=150); plt.close()

if __name__ == "__main__":
    run()
