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
            nn.Linear(in_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

def calculate_rates(y_true, y_prob, threshold=0.15):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    return fpr, fnr

def run_ood_decay_audit():
    print("==========================================================================")
    print(" Executing OOD Deployment Offset Decay Audit (Cumulative Origin Training) ")
    print("==========================================================================")
    
    ROOT = r"C:\Users\dhl\data\thesis\thesis"
    out_dir = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter4")
    os.makedirs(out_dir, exist_ok=True)
    
    horizons = [('H0 (Filing Date)', 'H0_Filing_Master_Enriched.csv'), 
                ('H3 (Pre-Council)', 'H3_Filing_Master_NLP.csv')]
                
    vintages = list(range(2009, 2025))  # Train up to V, deploy on V+1 to V+6
    offsets = [1, 2, 3, 4, 5, 6]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    
    for row_idx, (hz_label, hz_file) in enumerate(horizons):
        print(f"\n[+] Processing Horizon: {hz_label}")
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
        
        # We store list of metrics for each model, for each offset, across all Vintages
        metrics = {
            'CatBoost': {o: {'fpr': [], 'fnr': []} for o in offsets},
            'Random Forest': {o: {'fpr': [], 'fnr': []} for o in offsets},
            'Deep Surrogate': {o: {'fpr': [], 'fnr': []} for o in offsets}
        }
        
        for v in vintages:
            print(f"    -> Auditing Base Origin Vintage <= {v}")
            train_mask = (df['year'] <= v)
            if train_mask.sum() < 500 or y[train_mask].sum() < 10: 
                continue # Skip if training class is hopelessly sparse
                
            X_train, y_train = X[train_mask], y[train_mask]
            
            rf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            
            cb = CatBoostClassifier(iterations=50, learning_rate=0.1, verbose=0, random_state=42)
            cb.fit(X_train, y_train)
            
            scaler = StandardScaler()
            X_s_train = scaler.fit_transform(X_train)
            X_t = torch.FloatTensor(X_s_train)
            y_t = torch.FloatTensor(y_train).unsqueeze(1)
            
            dp = SimpleDeepTabular(X_s_train.shape[1])
            opt = torch.optim.Adam(dp.parameters(), lr=0.01)
            crit = nn.BCEWithLogitsLoss()
            for _ in range(30):
                opt.zero_grad()
                loss = crit(dp(X_t), y_t)
                loss.backward()
                opt.step()
                
            for off in offsets:
                test_mask = (df['year'] == (v + off))
                X_test, y_test = X[test_mask], y[test_mask]
                
                if y_test.sum() == 0 or len(X_test) < 10:
                    continue # Sparsity trap on the test year
                    
                # Evaluate
                rf_prob = rf.predict_proba(X_test)[:, 1]
                cb_prob = cb.predict_proba(X_test)[:, 1]
                
                X_s_test = scaler.transform(X_test)
                with torch.no_grad(): dp_prob = torch.sigmoid(dp(torch.FloatTensor(X_s_test))).numpy().flatten()
                
                rf_f, rf_n = calculate_rates(y_test, rf_prob, threshold=0.15)
                cb_f, cb_n = calculate_rates(y_test, cb_prob, threshold=0.15)
                dp_f, dp_n = calculate_rates(y_test, dp_prob, threshold=0.15)
                
                metrics['Random Forest'][off]['fpr'].append(rf_f)
                metrics['Random Forest'][off]['fnr'].append(rf_n)
                metrics['CatBoost'][off]['fpr'].append(cb_f)
                metrics['CatBoost'][off]['fnr'].append(cb_n)
                metrics['Deep Surrogate'][off]['fpr'].append(dp_f)
                metrics['Deep Surrogate'][off]['fnr'].append(dp_n)
                
        # Aggregate Expected Error over all vintages
        final_plot = {m: {'fpr': [], 'fnr': []} for m in metrics.keys()}
        for m in metrics.keys():
            for off in offsets:
                f_arr = np.array(metrics[m][off]['fpr'])
                n_arr = np.array(metrics[m][off]['fnr'])
                final_plot[m]['fpr'].append(np.nanmean(f_arr))
                final_plot[m]['fnr'].append(np.nanmean(n_arr))
                
        # Plotting
        ax_fpr = axes[row_idx, 0]
        ax_fpr.plot(offsets, final_plot['CatBoost']['fpr'], 'o-', label='CatBoost', color='darkred', linewidth=3)
        ax_fpr.plot(offsets, final_plot['Random Forest']['fpr'], 's--', label='Random Forest', color='gray', alpha=0.7)
        ax_fpr.plot(offsets, final_plot['Deep Surrogate']['fpr'], 'D-', label='Deep Surrogate', color='dodgerblue', linewidth=3)
        ax_fpr.set_title(f"{hz_label} - Expected FPR OOD Decay", fontsize=14, pad=10)
        ax_fpr.set_ylabel("Average FPR (across all origin vintages)")
        ax_fpr.set_xticks(offsets)
        ax_fpr.set_xticklabels([f"+{o} Yrs" for o in offsets])
        ax_fpr.grid(axis='y', linestyle='--', alpha=0.6)
        
        ax_fnr = axes[row_idx, 1]
        ax_fnr.plot(offsets, final_plot['CatBoost']['fnr'], 'o-', label='CatBoost', color='darkred', linewidth=3)
        ax_fnr.plot(offsets, final_plot['Random Forest']['fnr'], 's--', label='Random Forest', color='gray', alpha=0.7)
        ax_fnr.plot(offsets, final_plot['Deep Surrogate']['fnr'], 'D-', label='Deep Surrogate', color='dodgerblue', linewidth=3)
        ax_fnr.set_title(f"{hz_label} - Expected FNR OOD Decay", fontsize=14, pad=10)
        ax_fnr.set_ylabel("Average FNR (across all origin vintages)")
        ax_fnr.set_xticks(offsets)
        ax_fnr.set_xticklabels([f"+{o} Yrs" for o in offsets])
        ax_fnr.grid(axis='y', linestyle='--', alpha=0.6)
        
        if row_idx == 0: ax_fpr.legend(loc='best', fontsize=11)

    plt.suptitle("Out-of-Distribution Error Decay: Evaluating Architectural Drift Across Temporal Deployment Offsets", fontsize=16, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    out_pdf = os.path.join(out_dir, "fig_ood_offset_decay.pdf")
    plt.savefig(out_pdf)
    print(f"\n[+] Saved Matrix: {out_pdf}")

if __name__ == "__main__":
    run_ood_decay_audit()
