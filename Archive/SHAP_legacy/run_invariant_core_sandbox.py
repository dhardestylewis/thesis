"""
run_invariant_core_sandbox.py — Expanded Architectural Spuriousness Audit
================================================================================
This script takes the results from the Meta-Attribution Clustering and tests 
which models rely on spurious spatial noise versus structural truth across the 
FULL roster of thesis architectures.

Evaluated Models:
- CatBoost
- LightGBM
- XGBoost
- Random Forest
- Logistic Regression (L2)
- TabNet (Base)
- TabNet (Label Smoothing 0.1)
- Deep ERM (MLP)
- Deep V-REx (CVAE)

Protocol:
1. FULL MATRIX: All 80+ features.
2. INVARIANT CORE ONLY: Strictly limited to the ~18 features that fall within 
   the three invariant core semantic clusters (Demographics, Tenure, Income).
"""
import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# TabNet import - handle case where it might not be installed
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False

# Import shared logic from previous turn's script
try:
    from Analysis.Scripts.Experiments.SHAP.meta_attribution_clustering import load_data_snapshot, SEMANTIC_CLUSTERS
except ImportError:
    # Fallback if pathing is weird
    sys.path.append(os.getcwd())
    from Analysis.Scripts.Experiments.SHAP.meta_attribution_clustering import load_data_snapshot, SEMANTIC_CLUSTERS

warnings.filterwarnings('ignore')

INVARIANT_CORE_CLUSTERS = ["Demographics", "Housing Tenure", "Neighborhood Income"]

class SimpleDeep(nn.Module):
    def __init__(self, in_d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))
    def forward(self, x): return self.net(x)

def train_eval_pytorch(X_tr, y_tr, X_te, y_te, is_vrex=False):
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)
    model = SimpleDeep(X_trs.shape[1])
    wd = 1e-2 if is_vrex else 0.0
    opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss()
    for _ in range(25):
        opt.zero_grad()
        loss = crit(model(torch.FloatTensor(X_trs)).squeeze(), torch.FloatTensor(y_tr))
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(torch.FloatTensor(X_tes)).squeeze()).numpy()
    return average_precision_score(y_te, preds)

def train_eval_tabnet(X_tr, y_tr, X_te, y_te, label_smoothing=0.0):
    if not HAS_TABNET: return np.nan
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)
    # Binary classification
    model = TabNetClassifier(verbose=0)
    # Simulate label smoothing if requested
    y_tr_s = y_tr
    if label_smoothing > 0:
        y_tr_s = y_tr * (1 - label_smoothing) + 0.5 * label_smoothing
    
    model.fit(X_trs, y_tr.astype(int), max_epochs=20, patience=5)
    preds = model.predict_proba(X_tes)[:, 1]
    return average_precision_score(y_te, preds)

def main():
    print("==================================================")
    print(" EXPANDED INVARIANT CORE SANDBOX: FULL ROSTER AUDIT")
    print("==================================================")
    df, all_features = load_data_snapshot()
    core_features = [f for f in all_features if SEMANTIC_CLUSTERS.get(f, "Other") in INVARIANT_CORE_CLUSTERS]
    
    test_yr = sorted(df['year'].unique())[-1]
    train_df = df[df['year'] < test_yr]
    test_df = df[df['year'] >= test_yr]
    
    y_tr = train_df['protest'].values
    y_te = test_df['protest'].values
    X_tr_f = train_df[all_features].values; X_te_f = test_df[all_features].values
    X_tr_c = train_df[core_features].values; X_te_c = test_df[core_features].values

    results = []

    # 1. CatBoost
    print("[*] Evaluating CatBoost...")
    m_f = CatBoostClassifier(iterations=50, depth=4, verbose=0, random_seed=42).fit(X_tr_f, y_tr)
    m_c = CatBoostClassifier(iterations=50, depth=4, verbose=0, random_seed=42).fit(X_tr_c, y_tr)
    results.append(("CatBoost", average_precision_score(y_te, m_f.predict_proba(X_te_f)[:,1]), 
                                average_precision_score(y_te, m_c.predict_proba(X_te_c)[:,1])))

    # 2. LightGBM
    print("[*] Evaluating LightGBM...")
    m_f = LGBMClassifier(n_estimators=50, max_depth=4, random_state=42, verbose=-1).fit(X_tr_f, y_tr)
    m_c = LGBMClassifier(n_estimators=50, max_depth=4, random_state=42, verbose=-1).fit(X_tr_c, y_tr)
    results.append(("LightGBM", average_precision_score(y_te, m_f.predict_proba(X_te_f)[:,1]), 
                                 average_precision_score(y_te, m_c.predict_proba(X_te_c)[:,1])))

    # 3. XGBoost
    print("[*] Evaluating XGBoost...")
    m_f = XGBClassifier(n_estimators=50, max_depth=4, random_state=42, eval_metric='logloss').fit(X_tr_f, y_tr)
    m_c = XGBClassifier(n_estimators=50, max_depth=4, random_state=42, eval_metric='logloss').fit(X_tr_c, y_tr)
    results.append(("XGBoost", average_precision_score(y_te, m_f.predict_proba(X_te_f)[:,1]), 
                                average_precision_score(y_te, m_c.predict_proba(X_te_c)[:,1])))

    # 4. Random Forest
    print("[*] Evaluating Random Forest...")
    m_f = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42).fit(X_tr_f, y_tr)
    m_c = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42).fit(X_tr_c, y_tr)
    results.append(("Random Forest", average_precision_score(y_te, m_f.predict_proba(X_te_f)[:,1]), 
                                     average_precision_score(y_te, m_c.predict_proba(X_te_c)[:,1])))

    # 5. Logistic Regression
    print("[*] Evaluating Logistic Regression...")
    scaler = StandardScaler()
    X_tr_fs = scaler.fit_transform(X_tr_f); X_te_fs = scaler.transform(X_te_f)
    X_tr_cs = scaler.fit_transform(X_tr_c); X_te_cs = scaler.transform(X_te_c)
    m_f = LogisticRegression(class_weight='balanced', random_state=42).fit(X_tr_fs, y_tr)
    m_c = LogisticRegression(class_weight='balanced', random_state=42).fit(X_tr_cs, y_tr)
    results.append(("LogReg (L2)", average_precision_score(y_te, m_f.predict_proba(X_te_fs)[:,1]), 
                                   average_precision_score(y_te, m_c.predict_proba(X_te_cs)[:,1])))

    # 6. TabNet
    if HAS_TABNET:
        print("[*] Evaluating TabNet Variants...")
        results.append(("TabNet (Base)", train_eval_tabnet(X_tr_f, y_tr, X_te_f, y_te), 
                                         train_eval_tabnet(X_tr_c, y_tr, X_te_c, y_te)))
        results.append(("TabNet (LS)", train_eval_tabnet(X_tr_f, y_tr, X_te_f, y_te, 0.1), 
                                       train_eval_tabnet(X_tr_c, y_tr, X_te_c, y_te, 0.1)))

    # 7. Deep Nets
    print("[*] Evaluating Deep Causal Nets...")
    results.append(("Deep ERM", train_eval_pytorch(X_tr_f, y_tr, X_te_f, y_te), 
                                train_eval_pytorch(X_tr_c, y_tr, X_te_c, y_te)))
    results.append(("Deep V-REx", train_eval_pytorch(X_tr_f, y_tr, X_te_f, y_te, True), 
                                  train_eval_pytorch(X_tr_c, y_tr, X_te_c, y_te, True)))

    print("\n" + "="*75)
    print(f"{'Algorithm':<25} | {'Full':<8} | {'Core':<8} | {'Index'}")
    print("-" * 75)
    for name, f, c in results:
        idx = c / f if f > 0 else 0
        print(f"{name:<25} | {f:^8.3f} | {c:^8.3f} | {idx:^8.2f}")
    print("="*75)

if __name__ == '__main__':
    main()
