import os, sys, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

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
OUT_DIR = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter4")
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS = 100

def train_and_predict(model_name, seed, X_train, y_train, X_test):
    if model_name == 'Logistic Regression':
        m = LogisticRegression(random_state=seed, class_weight='balanced', max_iter=200)
        m.fit(X_train, y_train)
        return m.predict_proba(X_test)[:, 1]
    
    elif model_name == 'Random Forest':
        m = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1, class_weight='balanced')
        m.fit(X_train, y_train)
        return m.predict_proba(X_test)[:, 1]
    
    elif model_name == 'CatBoost':
        m = CatBoostClassifier(iterations=150, depth=6, learning_rate=0.05, verbose=0, random_seed=seed, auto_class_weights='Balanced')
        m.fit(X_train, y_train)
        return m.predict_proba(X_test)[:, 1]
    
    elif model_name == 'TabNet':
        from pytorch_tabnet.tab_model import TabNetClassifier
        m = TabNetClassifier(seed=seed, verbose=0, device_name='cpu')
        m.fit(X_train.values if hasattr(X_train, 'values') else X_train,
              y_train, max_epochs=20, patience=5, batch_size=256, drop_last=False)
        return m.predict_proba(X_test.values if hasattr(X_test, 'values') else X_test)[:, 1]

    elif model_name == 'FT-Transformer':
        torch.manual_seed(seed); np.random.seed(seed)
        d = X_train.shape[1]
        model = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        crit = nn.BCEWithLogitsLoss()
        Xt = torch.FloatTensor(X_train)
        yt = torch.FloatTensor(y_train).unsqueeze(1)
        for _ in range(50):
            opt.zero_grad(); loss = crit(model(Xt), yt); loss.backward(); opt.step()
        with torch.no_grad():
            return torch.sigmoid(model(torch.FloatTensor(X_test))).numpy().flatten()

    elif model_name == 'TabPFN':
        from tabpfn import TabPFNClassifier
        m = TabPFNClassifier(device='cpu', N_ensemble_configurations=4, seed=seed)
        n_max = min(1000, len(X_train))
        idx = np.random.RandomState(seed).choice(len(X_train), n_max, replace=False)
        m.fit(X_train.iloc[idx] if hasattr(X_train, 'iloc') else X_train[idx],
              y_train[idx])
        return m.predict_proba(X_test.values if hasattr(X_test, 'values') else X_test)[:, 1]

def run():
    print("==============================================")
    print(f" Rendering Unified Fig 5: 100-Seed Frontier PR Curves")
    print("==============================================")
    
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

    # Out-of-distribution 2023 evaluation
    train_mask = df['year'] < 2022
    test_mask = df['year'] == 2023
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    MODELS = ['Logistic Regression', 'Random Forest', 'CatBoost', 'TabNet', 'FT-Transformer', 'TabPFN']
    COLORS = {'Logistic Regression': 'coral', 'Random Forest': 'gray', 'CatBoost': 'darkred', 
              'TabNet': 'orange', 'FT-Transformer': 'green', 'TabPFN': 'purple'}
    
    common_recall = np.linspace(0, 1, 150)
    model_pr_curves = {m: [] for m in MODELS}
    model_aucs = {m: [] for m in MODELS}

    for m in MODELS:
        print(f"[*] Evaluating {m} across {N_SEEDS} seeds...")
        for s in range(N_SEEDS):
            try:
                # Scaled for linear/NN, raw for trees (though scaled is fine for trees too)
                if m in ['Logistic Regression', 'TabNet', 'FT-Transformer']:
                    # Ensure we pass values to NN-based models
                    probs = train_and_predict(m, s, X_train_scaled.values, y_train, X_test_scaled.values)
                else:
                    # Trees can handle DataFrames
                    probs = train_and_predict(m, s, X_train, y_train, X_test)
                
                auc = average_precision_score(y_test, probs)
                p, r, _ = precision_recall_curve(y_test, probs)
                p_interp = np.interp(common_recall, r[::-1], p[::-1])
                model_pr_curves[m].append(p_interp)
                model_aucs[m].append(auc)
                
                # Cleanup to prevent memory crashes
                if 'probs' in locals(): del probs
                
            except Exception as e:
                print(f"Error on {m} seed {s}: {e}")
            
            if s % 10 == 0: 
                print(f"        Seed {s}/100...")
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        # Intermediate results to console
        print(f"    {m} Result: {np.mean(model_aucs[m]):.4f} +/- {np.std(model_aucs[m]):.4f}")

    plt.figure(figsize=(10, 8))
    baseline = y_test.sum() / len(y_test)
    plt.plot([0, 1], [baseline, baseline], label=f'Baseline/Chance (PR-AUC {baseline:.2f})', linestyle=':', color='gray')
    
    for m in MODELS:
        if not model_pr_curves[m]: continue
        pr_array = np.array(model_pr_curves[m])
        mean_pr = np.mean(pr_array, axis=0)
        std_pr = np.std(pr_array, axis=0)
        
        mean_auc = np.mean(model_aucs[m])
        std_auc = np.std(model_aucs[m])
        
        plt.plot(common_recall, mean_pr, label=f"{m} (AUC={mean_auc:.3f} ± {std_auc:.3f})", color=COLORS[m], lw=2.5)
        plt.fill_between(common_recall, np.clip(mean_pr - 1.96*std_pr, 0, 1), np.clip(mean_pr + 1.96*std_pr, 0, 1), color=COLORS[m], alpha=0.1)

    plt.title(f'Unified Stage C Precision-Recall (100 Seeds, Out-of-Distribution 2023)', fontsize=14, pad=15)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(loc='lower left', fontsize=10, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.tight_layout()

    f12_path = os.path.join(OUT_DIR, "F12_Opposition_PR.png")
    plt.savefig(f12_path, dpi=300)
    print(f"[+] Successfully saved unified 100-seed curve: {f12_path}")

if __name__ == '__main__':
    run()
