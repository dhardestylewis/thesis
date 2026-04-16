"""
seed_stability_audit.py — Multi-Seed Robustness Validation
============================================================
Runs the primary Stage C models across 20 different random seeds
and reports mean ± std of key metrics to prove results are not
artifacts of a single seed selection.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
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
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

def run_seed_audit():
    print("================================================================")
    print(" Multi-Seed Stability Audit: 20 Seeds × 3 Models × 2 Horizons  ")
    print("================================================================")
    
    ROOT = r"C:\Users\dhl\data\thesis\thesis"
    out_dir = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter4")
    os.makedirs(out_dir, exist_ok=True)
    
    SEEDS = list(range(100))
    
    horizons = [
        ('H0 (Filing Date)', 'H0_Filing_Master_Enriched.csv'),
        ('H3 (Pre-Council)', 'H3_Filing_Master_NLP.csv')
    ]
    
    all_results = {}
    
    for hz_label, hz_file in horizons:
        print(f"\n[+] Horizon: {hz_label}")
        df = pd.read_csv(os.path.join(ROOT, "Data", "Warehouse_As_Of", hz_file), low_memory=False)
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
        
        train_mask = df['year'] < 2022
        test_mask = df['year'] >= 2022
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        models = {
            'CatBoost':       {'pr_auc': [], 'roc_auc': [], 'brier': []},
            'Random Forest':  {'pr_auc': [], 'roc_auc': [], 'brier': []},
            'Deep Surrogate': {'pr_auc': [], 'roc_auc': [], 'brier': []}
        }
        
        for seed in SEEDS:
            # CatBoost
            cb = CatBoostClassifier(iterations=200, learning_rate=0.1, verbose=0, random_state=seed)
            cb.fit(X_train, y_train)
            cb_prob = cb.predict_proba(X_test)[:, 1]
            models['CatBoost']['pr_auc'].append(average_precision_score(y_test, cb_prob))
            models['CatBoost']['roc_auc'].append(roc_auc_score(y_test, cb_prob))
            models['CatBoost']['brier'].append(brier_score_loss(y_test, cb_prob))
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_prob = rf.predict_proba(X_test)[:, 1]
            models['Random Forest']['pr_auc'].append(average_precision_score(y_test, rf_prob))
            models['Random Forest']['roc_auc'].append(roc_auc_score(y_test, rf_prob))
            models['Random Forest']['brier'].append(brier_score_loss(y_test, rf_prob))
            
            # Deep Surrogate
            torch.manual_seed(seed)
            np.random.seed(seed)
            scaler = StandardScaler()
            X_s_train = scaler.fit_transform(X_train)
            X_s_test = scaler.transform(X_test)
            
            dp = SimpleDeepTabular(X_s_train.shape[1])
            opt = torch.optim.Adam(dp.parameters(), lr=0.005)
            crit = nn.BCEWithLogitsLoss()
            X_t = torch.FloatTensor(X_s_train)
            y_t = torch.FloatTensor(y_train).unsqueeze(1)
            
            for _ in range(50):
                opt.zero_grad()
                loss = crit(dp(X_t), y_t)
                loss.backward()
                opt.step()
                
            with torch.no_grad():
                dp_prob = torch.sigmoid(dp(torch.FloatTensor(X_s_test))).numpy().flatten()
            
            models['Deep Surrogate']['pr_auc'].append(average_precision_score(y_test, dp_prob))
            models['Deep Surrogate']['roc_auc'].append(roc_auc_score(y_test, dp_prob))
            models['Deep Surrogate']['brier'].append(brier_score_loss(y_test, dp_prob))
            
            print(f"    Seed {seed:2d}: CB PR-AUC={models['CatBoost']['pr_auc'][-1]:.4f} | "
                  f"RF={models['Random Forest']['pr_auc'][-1]:.4f} | "
                  f"Deep={models['Deep Surrogate']['pr_auc'][-1]:.4f}")
        
        all_results[hz_label] = models
        
        # Print summary
        print(f"\n    {'Model':<17} {'PR-AUC':>18} {'ROC-AUC':>18} {'Brier':>18}")
        print(f"    {'-'*71}")
        for m in models:
            pr = np.array(models[m]['pr_auc'])
            roc = np.array(models[m]['roc_auc'])
            br = np.array(models[m]['brier'])
            print(f"    {m:<17} {pr.mean():.4f} ± {pr.std():.4f}  "
                  f"{roc.mean():.4f} ± {roc.std():.4f}  "
                  f"{br.mean():.4f} ± {br.std():.4f}")
    
    # ---- Visualization: Box plots ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metric_labels = ['PR-AUC', 'ROC-AUC', 'Brier Score']
    metric_keys = ['pr_auc', 'roc_auc', 'brier']
    colors = {'CatBoost': '#8B0000', 'Random Forest': '#888888', 'Deep Surrogate': '#1E90FF'}
    
    for row_idx, (hz_label, models) in enumerate(all_results.items()):
        for col_idx, (ml, mk) in enumerate(zip(metric_labels, metric_keys)):
            ax = axes[row_idx, col_idx]
            data = [models[m][mk] for m in models]
            bp = ax.boxplot(data, labels=list(models.keys()), patch_artist=True, widths=0.6)
            for patch, m in zip(bp['boxes'], models.keys()):
                patch.set_facecolor(colors[m])
                patch.set_alpha(0.7)
            for patch in bp['medians']:
                patch.set_color('white')
                patch.set_linewidth(2)
            
            # Overlay individual seed points
            for i, m in enumerate(models.keys()):
                jitter = np.random.normal(0, 0.04, size=len(models[m][mk]))
                ax.scatter(np.ones(len(models[m][mk])) * (i + 1) + jitter,
                          models[m][mk], color='black', alpha=0.3, s=15, zorder=5)
            
            ax.set_title(f"{hz_label}\n{ml}", fontsize=12, pad=8)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            if mk == 'brier':
                ax.set_ylabel("Score (Lower is Better)")
            else:
                ax.set_ylabel("Score (Higher is Better)")
    
    plt.suptitle("Multi-Seed Stability Audit: 20 Random Seeds × 3 Architectures × 2 Horizons",
                 fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_pdf = os.path.join(out_dir, "fig_seed_stability.pdf")
    plt.savefig(out_pdf, dpi=150)
    print(f"\n[+] Saved: {out_pdf}")

if __name__ == "__main__":
    run_seed_audit()
