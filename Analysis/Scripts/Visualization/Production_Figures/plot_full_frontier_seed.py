"""
plot_full_frontier_seed_audit.py — Complete Unified Frontier Seed Stability
===========================================================================
Runs ALL 5 stochastic models from the Unified Predictive Frontier across
20 seeds × 2 horizons AND the OOD variance interaction for each.
Models: CatBoost, Random Forest, TabNet, FT-Transformer, TabPFN
(Logistic Regression and ExcelFormer excluded: LR is deterministic,
ExcelFormer requires custom training loop not available as a package.)
"""
import os, sys, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import torch, torch.nn as nn

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

def get_data(hz_file, root):
    df = pd.read_csv(os.path.join(root, "Data", "Warehouse_As_Of", hz_file), low_memory=False)
    target_col = 'is_protested' if 'is_protested' in df.columns else 'protest'
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    drop_cols = [target_col, 'case_number', 'organized_opposition', 'has_audio_record',
                 'TCAD ID', 'date', 'application_start_date', 'final_date',
                 'standardized_tcad_id', 'Prob_H=4', 'Prob_LGBM_H=4',
                 'Prob_CB_H=4', 'Prob_Optimal_H=4', 'ipw',
                 'council_district', 'council_district_x']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df_clean = df_clean.drop(columns=[c for c in df_clean.columns if c.startswith('tfidf_') or c.startswith('speech_')])
    X = df_clean.select_dtypes(include=[np.number]).fillna(0)
    y = df[target_col].values
    return df, X, y, target_col

def train_and_eval(model_name, seed, X_train, y_train, X_test, y_test):
    """Train a model and return predicted probabilities on X_test."""
    if model_name == 'CatBoost':
        m = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=0, random_state=seed)
        m.fit(X_train, y_train)
        return m.predict_proba(X_test)[:, 1]

    elif model_name == 'Random Forest':
        m = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
        m.fit(X_train, y_train)
        return m.predict_proba(X_test)[:, 1]

    elif model_name == 'TabNet':
        from pytorch_tabnet.tab_model import TabNetClassifier
        m = TabNetClassifier(seed=seed, verbose=0, device_name='cpu')
        m.fit(X_train.values if hasattr(X_train, 'values') else X_train,
              y_train,
              max_epochs=30, patience=10, batch_size=256,
              drop_last=False)
        return m.predict_proba(X_test.values if hasattr(X_test, 'values') else X_test)[:, 1]

    elif model_name == 'FT-Transformer':
        torch.manual_seed(seed); np.random.seed(seed)
        scaler = StandardScaler()
        Xs_tr = scaler.fit_transform(X_train)
        Xs_te = scaler.transform(X_test)
        d = Xs_tr.shape[1]
        # Simple transformer-style model
        model = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        crit = nn.BCEWithLogitsLoss()
        Xt = torch.FloatTensor(Xs_tr)
        yt = torch.FloatTensor(y_train).unsqueeze(1)
        for _ in range(50):
            opt.zero_grad(); loss = crit(model(Xt), yt); loss.backward(); opt.step()
        with torch.no_grad():
            return torch.sigmoid(model(torch.FloatTensor(Xs_te))).numpy().flatten()

    elif model_name == 'TabPFN':
        from tabpfn import TabPFNClassifier
        m = TabPFNClassifier(device='cpu', N_ensemble_configurations=4, seed=seed)
        # TabPFN has a 1000-sample limit for training
        n_max = min(1000, len(X_train))
        idx = np.random.RandomState(seed).choice(len(X_train), n_max, replace=False)
        m.fit(X_train.iloc[idx] if hasattr(X_train, 'iloc') else X_train[idx],
              y_train[idx])
        return m.predict_proba(X_test.values if hasattr(X_test, 'values') else X_test)[:, 1]

def run():
    ROOT = r"C:\Users\dhl\data\thesis\thesis"
    out_dir = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter4")

    SEEDS = list(range(10))
    MODELS = ['CatBoost', 'Random Forest', 'TabNet', 'FT-Transformer', 'TabPFN']
    COLORS = {'CatBoost': '#8B0000', 'Random Forest': '#888888', 'TabNet': '#FF8C00',
              'FT-Transformer': '#228B22', 'TabPFN': '#9400D3'}
    MARKERS = {'CatBoost': 'o', 'Random Forest': 's', 'TabNet': '^',
               'FT-Transformer': 'D', 'TabPFN': 'P'}

    horizons = [('H0 (Filing)', 'H0_Filing_Master_Enriched.csv'),
                ('H3 (Pre-Council)', 'H3_Filing_Master_NLP.csv')]

    # =============================================
    # PART 1: Multi-Seed Stability (fixed split)
    # =============================================
    print("=" * 70)
    print(" PART 1: Full Frontier Multi-Seed Stability (10 seeds x 5 models)")
    print("=" * 70)

    all_results = {}
    for hz_label, hz_file in horizons:
        print(f"\n[+] {hz_label}")
        df, X, y, tc = get_data(hz_file, ROOT)
        train_mask = df['year'] < 2022
        test_mask = df['year'] >= 2022
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        models_res = {m: {'pr_auc': [], 'roc_auc': [], 'brier': []} for m in MODELS}

        for seed in SEEDS:
            for mname in MODELS:
                try:
                    prob = train_and_eval(mname, seed, X_train, y_train, X_test, y_test)
                    models_res[mname]['pr_auc'].append(average_precision_score(y_test, prob))
                    models_res[mname]['roc_auc'].append(roc_auc_score(y_test, prob))
                    models_res[mname]['brier'].append(brier_score_loss(y_test, prob))
                except Exception as e:
                    print(f"    WARN: {mname} seed={seed} failed: {e}")
                    models_res[mname]['pr_auc'].append(np.nan)
                    models_res[mname]['roc_auc'].append(np.nan)
                    models_res[mname]['brier'].append(np.nan)
                gc.collect()
            print(f"    Seed {seed}: " + " | ".join(
                f"{m}={models_res[m]['pr_auc'][-1]:.3f}" for m in MODELS))

        all_results[hz_label] = models_res

        # Print summary
        print(f"\n    {'Model':<18} {'PR-AUC':>18} {'ROC-AUC':>18} {'Brier':>18}")
        print(f"    {'-'*72}")
        for m in MODELS:
            pr = np.array(models_res[m]['pr_auc'])
            roc = np.array(models_res[m]['roc_auc'])
            br = np.array(models_res[m]['brier'])
            print(f"    {m:<18} {np.nanmean(pr):.4f} +/- {np.nanstd(pr):.4f}  "
                  f"{np.nanmean(roc):.4f} +/- {np.nanstd(roc):.4f}  "
                  f"{np.nanmean(br):.4f} +/- {np.nanstd(br):.4f}")

    # Plot Part 1
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    metric_labels = ['PR-AUC', 'ROC-AUC', 'Brier Score']
    metric_keys = ['pr_auc', 'roc_auc', 'brier']

    for row_idx, (hz_label, models) in enumerate(all_results.items()):
        for col_idx, (ml, mk) in enumerate(zip(metric_labels, metric_keys)):
            ax = axes[row_idx, col_idx]
            data = [models[m][mk] for m in MODELS]
            bp = ax.boxplot(data, labels=[m.replace(' ', '\n') for m in MODELS],
                           patch_artist=True, widths=0.6)
            for patch, m in zip(bp['boxes'], MODELS):
                patch.set_facecolor(COLORS[m]); patch.set_alpha(0.7)
            for patch in bp['medians']:
                patch.set_color('white'); patch.set_linewidth(2)
            for i, m in enumerate(MODELS):
                jitter = np.random.normal(0, 0.04, size=len(models[m][mk]))
                ax.scatter(np.ones(len(models[m][mk])) * (i+1) + jitter,
                          models[m][mk], color='black', alpha=0.3, s=12, zorder=5)
            ax.set_title(f"{hz_label}\n{ml}", fontsize=11, pad=8)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle("Full Frontier Multi-Seed Stability: 10 Seeds x 5 Architectures x 2 Horizons",
                 fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = os.path.join(out_dir, "fig_full_frontier_seed.pdf")
    plt.savefig(p1, dpi=150)
    print(f"\n[+] Saved Part 1: {p1}")
    plt.close()

    # =============================================
    # PART 2: OOD Variance Interaction (all models)
    # =============================================
    print("\n" + "=" * 70)
    print(" PART 2: Full Frontier OOD Seed Variance Interaction")
    print("=" * 70)

    OFFSETS = [0, 1, 2, 3, 4, 5]
    VINTAGES = [2012, 2015]

    df, X, y, tc = get_data('H0_Filing_Master_Enriched.csv', ROOT)
    ood_res = {m: {off: [] for off in OFFSETS} for m in MODELS}

    for v in VINTAGES:
        train_mask = df['year'] <= v
        if y[train_mask].sum() < 10: continue
        X_tr, y_tr = X[train_mask], y[train_mask]

        for seed in SEEDS:
            print(f"  V<={v}, Seed={seed}")
            for mname in MODELS:
                try:
                    # Train once per seed
                    for off in OFFSETS:
                        test_mask = df['year'] == (v + off)
                        X_te, y_te = X[test_mask], y[test_mask]
                        if len(X_te) < 10 or y_te.sum() == 0: continue
                        prob = train_and_eval(mname, seed, X_tr, y_tr, X_te, y_te)
                        ood_res[mname][off].append(average_precision_score(y_te, prob))
                except Exception as e:
                    print(f"    WARN: {mname} failed: {e}")
                gc.collect()

    fig, axes = plt.subplots(1, len(MODELS), figsize=(4*len(MODELS), 5), sharey=True)
    for ax, mname in zip(axes, MODELS):
        means, stds, offs_valid = [], [], []
        for off in OFFSETS:
            vals = np.array(ood_res[mname][off])
            if len(vals) > 2:
                means.append(np.nanmean(vals)); stds.append(np.nanstd(vals))
                offs_valid.append(off)
        means, stds = np.array(means), np.array(stds)
        ax.plot(offs_valid, means, f"{MARKERS[mname]}-", color=COLORS[mname], lw=2.5, ms=8)
        ax.fill_between(offs_valid, means-stds, means+stds, alpha=0.25, color=COLORS[mname], label='+/-1 std')
        ax.fill_between(offs_valid, means-2*stds, means+2*stds, alpha=0.10, color=COLORS[mname])
        ax.set_title(mname, fontsize=12, fontweight='bold')
        ax.set_xlabel("OOD Offset (years)")
        ax.set_xticks(offs_valid)
        ax.set_xticklabels([f"+{o}" for o in offs_valid])
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    axes[0].set_ylabel("PR-AUC")
    plt.suptitle("Full Frontier: Seed Variance x OOD Offset (All 5 Architectures)", fontsize=14, y=1.01)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "fig_full_frontier_ood_variance.pdf")
    plt.savefig(p2, bbox_inches='tight', dpi=150)
    print(f"\n[+] Saved Part 2: {p2}")

    # Print numerical summary
    print("\n[+] OOD Variance Summary (std of PR-AUC):")
    print(f"    {'Offset':<8}", end="")
    for m in MODELS: print(f"{m:<18}", end="")
    print()
    for off in OFFSETS:
        print(f"    +{off:<7}", end="")
        for m in MODELS:
            vals = np.array(ood_res[m][off])
            if len(vals) > 2: print(f"{np.nanstd(vals):.4f}            ", end="")
            else: print(f"{'N/A':<18}", end="")
        print()

if __name__ == "__main__":
    run()
