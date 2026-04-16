"""
plot_ood_seed_variance.py — Seed Variance × OOD Offset Interaction
===================================================================
Tests whether model instability (seed variance) increases with
temporal deployment distance, as would be expected if the model
is genuinely degrading out-of-distribution.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler

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

class SimpleDeepTabular(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x): return self.net(x)

def run():
    ROOT = r"C:\Users\dhl\data\thesis\thesis"
    out_dir = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter4")
    os.makedirs(out_dir, exist_ok=True)

    SEEDS = list(range(10))
    OFFSETS = [0, 1, 2, 3, 4, 5]
    # Use a few representative origin vintages to average over
    VINTAGES = [2012, 2014, 2016]

    hz_file = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
    df = pd.read_csv(hz_file, low_memory=False)
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

    models_cfg = {
        'CatBoost': {'color': '#8B0000', 'marker': 'o'},
        'Random Forest': {'color': '#888888', 'marker': 's'},
        'Deep Surrogate': {'color': '#1E90FF', 'marker': 'D'}
    }

    # For each offset, collect PR-AUC across all seeds × all vintages
    results = {m: {off: [] for off in OFFSETS} for m in models_cfg}

    for v in VINTAGES:
        train_mask = df['year'] <= v
        if y[train_mask].sum() < 10:
            continue
        X_train_base, y_train_base = X[train_mask], y[train_mask]

        for seed in SEEDS:
            print(f"  Vintage<={v}, Seed={seed}")

            # CatBoost
            cb = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=0, random_state=seed)
            cb.fit(X_train_base, y_train_base)

            # Random Forest
            rf = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
            rf.fit(X_train_base, y_train_base)

            # Deep Surrogate
            torch.manual_seed(seed); np.random.seed(seed)
            scaler = StandardScaler()
            Xs_tr = scaler.fit_transform(X_train_base)
            dp = SimpleDeepTabular(Xs_tr.shape[1])
            opt = torch.optim.Adam(dp.parameters(), lr=0.005)
            crit = nn.BCEWithLogitsLoss()
            Xt = torch.FloatTensor(Xs_tr)
            yt = torch.FloatTensor(y_train_base).unsqueeze(1)
            for _ in range(40):
                opt.zero_grad(); loss = crit(dp(Xt), yt); loss.backward(); opt.step()

            for off in OFFSETS:
                test_year = v + off
                test_mask = df['year'] == test_year
                X_test, y_test = X[test_mask], y[test_mask]
                if len(X_test) < 10 or y_test.sum() == 0:
                    continue

                cb_pr = average_precision_score(y_test, cb.predict_proba(X_test)[:, 1])
                rf_pr = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
                with torch.no_grad():
                    dp_pr = average_precision_score(y_test, torch.sigmoid(dp(torch.FloatTensor(scaler.transform(X_test)))).numpy().flatten())

                results['CatBoost'][off].append(cb_pr)
                results['Random Forest'][off].append(rf_pr)
                results['Deep Surrogate'][off].append(dp_pr)

    # Plot: mean ± std bands
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, (model, cfg) in zip(axes, models_cfg.items()):
        means, stds, offsets_valid = [], [], []
        for off in OFFSETS:
            vals = np.array(results[model][off])
            if len(vals) > 2:
                means.append(vals.mean())
                stds.append(vals.std())
                offsets_valid.append(off)

        means, stds = np.array(means), np.array(stds)
        ax.plot(offsets_valid, means, f"{cfg['marker']}-", color=cfg['color'], linewidth=2.5, markersize=8, label='Mean PR-AUC')
        ax.fill_between(offsets_valid, means - stds, means + stds, alpha=0.25, color=cfg['color'], label='±1 σ (seed variance)')
        ax.fill_between(offsets_valid, means - 2*stds, means + 2*stds, alpha=0.10, color=cfg['color'], label='±2 σ')

        ax.set_title(model, fontsize=14, fontweight='bold')
        ax.set_xlabel("OOD Deployment Offset (years)")
        ax.set_xticks(offsets_valid)
        ax.set_xticklabels([f"+{o}" for o in offsets_valid])
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='lower left', fontsize=9)

    axes[0].set_ylabel("PR-AUC")
    plt.suptitle("Seed Variance × OOD Offset: Do Confidence Intervals Widen With Deployment Distance?",
                 fontsize=15, y=1.0)
    plt.tight_layout()

    out_pdf = os.path.join(out_dir, "fig_ood_seed_variance.pdf")
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"\n[+] Saved: {out_pdf}")

    # Print the numerical evidence
    print("\n[+] Numerical Summary: std(PR-AUC) by offset")
    print(f"    {'Offset':<10}", end="")
    for m in models_cfg: print(f"{m:<20}", end="")
    print()
    for off in OFFSETS:
        print(f"    +{off:<9}", end="")
        for m in models_cfg:
            vals = np.array(results[m][off])
            if len(vals) > 2:
                print(f"{vals.std():.4f}              ", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()

if __name__ == "__main__":
    run()
