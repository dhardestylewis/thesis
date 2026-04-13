import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings

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
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def calculate_rates(y_true, y_prob, threshold=0.15):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    return fpr, fnr

def plot_fnr_fpr():
    print("==========================================================")
    print(" Rendering Matrix: FNR / FPR Longitudinal Volatility Audit")
    print("==========================================================")
    
    ROOT = r"C:\Users\dhl\data\thesis\thesis"
    out_dir = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter4")
    os.makedirs(out_dir, exist_ok=True)
    
    horizons = [('H0 (Filing Date)', 'H0_Filing_Master_Enriched.csv'), 
                ('H3 (Pre-Council)', 'H3_Filing_Master_NLP.csv')]
                
    years = list(range(2008, 2024))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    
    for row_idx, (hz_label, hz_file) in enumerate(horizons):
        print(f"[+] Processing {hz_label}...")
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
        
        # Train baseline models on older data to predict rolling years, 
        # or train per window. We simulate cross-validation by fitting on the rest of the dataset.
        X = df_clean.select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col].values
        
        results = {
            'CatBoost': {'fpr': [], 'fnr': []},
            'Random Forest': {'fpr': [], 'fnr': []},
            'Deep Surrogate': {'fpr': [], 'fnr': []}
        }
        
        for tgt_year in years:
            train_mask = (df['year'] != tgt_year)
            test_mask = (df['year'] == tgt_year)
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            
            if len(X_test) < 10 or y_test.sum() == 0:
                for k in results.keys():
                    results[k]['fpr'].append(np.nan)
                    results[k]['fnr'].append(np.nan)
                continue
                
            # Random Forest
            rf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_prob = rf.predict_proba(X_test)[:, 1]
            rf_fpr, rf_fnr = calculate_rates(y_test, rf_prob, threshold=0.15)
            
            # CatBoost
            cb = CatBoostClassifier(iterations=50, learning_rate=0.1, verbose=0, random_state=42)
            cb.fit(X_train, y_train)
            cb_prob = cb.predict_proba(X_test)[:, 1]
            cb_fpr, cb_fnr = calculate_rates(y_test, cb_prob, threshold=0.15)
            
            # Deep Surrogate
            scaler = StandardScaler()
            X_s_train = scaler.fit_transform(X_train)
            X_s_test = scaler.transform(X_test)
            
            X_t = torch.FloatTensor(X_s_train)
            y_t = torch.FloatTensor(y_train).unsqueeze(1)
            X_v = torch.FloatTensor(X_s_test)
            
            dp = SimpleDeepTabular(X_s_train.shape[1])
            opt = torch.optim.Adam(dp.parameters(), lr=0.01)
            crit = nn.BCEWithLogitsLoss()
            
            for _ in range(30):
                opt.zero_grad()
                out = dp(X_t)
                loss = crit(out, y_t)
                loss.backward()
                opt.step()
                
            with torch.no_grad():
                dp_out = dp(X_v)
                dp_prob = torch.sigmoid(dp_out).numpy().flatten()
            
            dp_fpr, dp_fnr = calculate_rates(y_test, dp_prob, threshold=0.15)
            
            results['Random Forest']['fpr'].append(rf_fpr)
            results['Random Forest']['fnr'].append(rf_fnr)
            results['CatBoost']['fpr'].append(cb_fpr)
            results['CatBoost']['fnr'].append(cb_fnr)
            results['Deep Surrogate']['fpr'].append(dp_fpr)
            results['Deep Surrogate']['fnr'].append(dp_fnr)

        # Plot FPR
        ax_fpr = axes[row_idx, 0]
        ax_fpr.plot(years, results['CatBoost']['fpr'], 'o-', label='CatBoost', color='darkred', alpha=0.8, linewidth=2)
        ax_fpr.plot(years, results['Random Forest']['fpr'], 'x--', label='Random Forest', color='gray', alpha=0.7)
        ax_fpr.plot(years, results['Deep Surrogate']['fpr'], 'D-', label='Deep Surrogate', color='dodgerblue', alpha=0.8, linewidth=2)
        ax_fpr.set_title(f"{hz_label} - False Positive Rate (FPR)", fontsize=13, pad=10)
        ax_fpr.set_ylabel("FPR (Lower is Better)")
        ax_fpr.set_ylim(-0.05, 1.05)
        ax_fpr.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Plot FNR
        ax_fnr = axes[row_idx, 1]
        ax_fnr.plot(years, results['CatBoost']['fnr'], 'o-', label='CatBoost', color='darkred', alpha=0.8, linewidth=2)
        ax_fnr.plot(years, results['Random Forest']['fnr'], 'x--', label='Random Forest', color='gray', alpha=0.7)
        ax_fnr.plot(years, results['Deep Surrogate']['fnr'], 'D-', label='Deep Surrogate', color='dodgerblue', alpha=0.8, linewidth=2)
        ax_fnr.set_title(f"{hz_label} - False Negative Rate (FNR)", fontsize=13, pad=10)
        ax_fnr.set_ylabel("FNR (Lower is Better)")
        ax_fnr.set_ylim(-0.05, 1.05)
        ax_fnr.grid(axis='y', linestyle='--', alpha=0.6)
        
        if row_idx == 0:
            ax_fpr.legend(loc='upper left', fontsize=10)

    for i in range(2):
        axes[1, i].set_xlabel("Filing Year (Out of Distribution Target)", fontsize=12)

    plt.suptitle("Algorithmic Error Volatility: Evaluation of FNR and FPR Across Time, Horizons, and Models", fontsize=16, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    out_pdf = os.path.join(out_dir, "fig_fpr_fnr_longitudinal.pdf")
    plt.savefig(out_pdf)
    print(f"[+] Saved Longitudinal Error Matrix: {out_pdf}")

if __name__ == "__main__":
    plot_fnr_fpr()
