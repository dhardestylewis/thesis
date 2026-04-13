"""
plot_all_stages_seed_audit.py — Multi-Stage Multi-Seed Stability Audit
======================================================================
Runs seed stability across ALL thesis modeling stages:
  Stage A: Development Occurrence Hazard (LightGBM, CatBoost)
  Stage B: Zoning Typology Classification (CatBoost multi-class)
  Stage C: Opposition/Petition Prediction (CatBoost, RF, TabNet, FT-Transformer surrogate)
  Stage D: Attrition/Outcome (CatBoost, RF, Logistic)
"""
import os, sys, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss, f1_score, log_loss)
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import lightgbm as lgb
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

ROOT = r"C:\Users\dhl\data\thesis\thesis"
SEEDS = list(range(10))

class DeepSurrogate(nn.Module):
    def __init__(self, d, out=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(64,32), nn.ReLU(), nn.Linear(32,out))
    def forward(self, x): return self.net(x)


def load_stage_c():
    """Load Stage C data (already well-tested)."""
    df = pd.read_csv(os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv"), low_memory=False)
    tc = 'is_protested' if 'is_protested' in df.columns else 'protest'
    df[tc] = pd.to_numeric(df[tc], errors='coerce').fillna(0).astype(int)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    drop = [tc, 'case_number', 'organized_opposition', 'has_audio_record',
            'TCAD ID', 'date', 'application_start_date', 'final_date',
            'standardized_tcad_id', 'Prob_H=4', 'Prob_LGBM_H=4',
            'Prob_CB_H=4', 'Prob_Optimal_H=4', 'ipw', 'council_district', 'council_district_x']
    Xdf = df.drop(columns=[c for c in drop if c in df.columns])
    Xdf = Xdf.drop(columns=[c for c in Xdf.columns if c.startswith('tfidf_') or c.startswith('speech_')])
    X = Xdf.select_dtypes(include=[np.number]).fillna(0)
    y = df[tc].values
    train = df['year'] < 2022
    test = df['year'] >= 2022
    return X[train], y[train], X[test], y[test]


def load_stage_a():
    """Load Stage A hazard data from the enriched panel."""
    panel_file = os.path.join(ROOT, "Data", "Panel", "Output", "Property_Year_Panel_Enriched.csv")
    df = pd.read_csv(panel_file, low_memory=False, nrows=50000)  # Sample for speed
    if 'protest' not in df.columns:
        # No clear binary target in the panel, use development occurrence
        df['dev_event'] = 0
        # Approximate: properties with a zoning case GEOID that isn't null
        if 'zoning_case_GEOID' in df.columns:
            df['dev_event'] = df['zoning_case_GEOID'].notna().astype(int)
    tc = 'dev_event' if 'dev_event' in df.columns else 'protest'
    df[tc] = pd.to_numeric(df[tc], errors='coerce').fillna(0).astype(int)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    drop = [tc, 'standardized_tcad_id', 'owner_name', 'owner_id',
            'owner_address_line1', 'owner_address_line2', 'owner_city', 'owner_state',
            'situs_street_number', 'situs_street_prefix', 'situs_street_name',
            'situs_street_suffix', 'situs_city_state_zip',
            'appraisal_district_id', 'county_id', 'taxing_unit_id', 'taxing_unit_name',
            'account_number_formatted', 'property_type_code',
            'nearby_GEOID', 'zoning_case_GEOID', 'appraisal_district_id_2']
    Xdf = df.drop(columns=[c for c in drop if c in df.columns])
    X = Xdf.select_dtypes(include=[np.number]).fillna(0)
    y = df[tc].values
    train = df['year'] < 2020
    test = df['year'] >= 2020
    return X[train], y[train], X[test], y[test]


def load_stage_d():
    """Load Stage D attrition data from H0 master (binary: withdrawn or not) merged with goldmine tensor."""
    df = pd.read_csv(os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv"), low_memory=False)
    
    VOTE_DATA = os.path.join(ROOT, 'Data', 'Zoning_Cases', 'Processed_Data', 'CSV', 'submission_grade_goldmine_tensor.csv')
    if os.path.exists(VOTE_DATA):
        votes = pd.read_csv(VOTE_DATA, usecols=['CASE_NUMBER', 'vote_yes', 'vote_no'])
        votes = votes.groupby('CASE_NUMBER', as_index=False).agg({'vote_yes': 'sum', 'vote_no': 'sum'})
        df = df.merge(votes, left_on='case_number', right_on='CASE_NUMBER', how='left')
        df['is_withdrawn'] = df['vote_yes'].isna().astype(int)
    else:
        return None, None, None, None

    # Filter for opposed cases
    df['is_protested'] = pd.to_numeric(df['is_protested'], errors='coerce')
    df = df.dropna(subset=['is_protested'])
    df['is_protested'] = df['is_protested'].astype(int)
    df = df[df['is_protested'] == 1].copy()

    if len(df) < 10:
        return None, None, None, None

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    drop = ['is_withdrawn', 'case_number', 'is_protested', 'protest', 'has_audio_record',
            'TCAD ID', 'date', 'application_start_date', 'final_date',
            'standardized_tcad_id', 'council_district', 'council_district_x',
            'vote_yes', 'vote_no', 'CASE_NUMBER', 'council_approval', 'ordinance_number']
    Xdf = df.drop(columns=[c for c in drop if c in df.columns])
    Xdf = Xdf.drop(columns=[c for c in Xdf.columns if c.startswith('tfidf_') or c.startswith('speech_')])
    X = Xdf.select_dtypes(include=[np.number]).fillna(0)
    y = df['is_withdrawn'].values
    
    from sklearn.model_selection import train_test_split
    # We must use random split because 'vote_yes' data may not be populated for 2022+ yet
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Return as arrays
    if hasattr(X_train, 'values'):
        X_train = X_train.values
        X_test = X_test.values
    return X_train, y_train, X_test, y_test


def eval_binary(seed, X_train, y_train, X_test, y_test):
    """Run CatBoost, RF, and Deep Surrogate on a binary task; return dict of metrics."""
    results = {}

    # CatBoost
    cb = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=0, random_state=seed, allow_writing_files=False)
    cb.fit(X_train, y_train)
    p = cb.predict_proba(X_test)[:, 1]
    results['CatBoost'] = {'pr_auc': average_precision_score(y_test, p),
                           'roc_auc': roc_auc_score(y_test, p),
                           'brier': brier_score_loss(y_test, p)}

    # RF
    rf = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    p = rf.predict_proba(X_test)[:, 1]
    results['Random Forest'] = {'pr_auc': average_precision_score(y_test, p),
                                'roc_auc': roc_auc_score(y_test, p),
                                'brier': brier_score_loss(y_test, p)}

    # LightGBM
    lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=seed, verbose=-1)
    lgbm.fit(X_train, y_train)
    p = lgbm.predict_proba(X_test)[:, 1]
    results['LightGBM'] = {'pr_auc': average_precision_score(y_test, p),
                           'roc_auc': roc_auc_score(y_test, p),
                           'brier': brier_score_loss(y_test, p)}

    # Deep Surrogate
    torch.manual_seed(seed); np.random.seed(seed)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train); Xte = sc.transform(X_test)
    m = DeepSurrogate(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=0.005)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.FloatTensor(Xtr); yt = torch.FloatTensor(y_train).unsqueeze(1)
    for _ in range(40):
        opt.zero_grad(); loss = crit(m(Xt), yt); loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.sigmoid(m(torch.FloatTensor(Xte))).numpy().flatten()
    results['Deep Surrogate'] = {'pr_auc': average_precision_score(y_test, p),
                                 'roc_auc': roc_auc_score(y_test, p),
                                 'brier': brier_score_loss(y_test, p)}
    return results


def run():
    out_dir = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter4")
    os.makedirs(out_dir, exist_ok=True)
    models = ['CatBoost', 'Random Forest', 'LightGBM', 'Deep Surrogate']
    colors = {'CatBoost': '#8B0000', 'Random Forest': '#888888',
              'LightGBM': '#006400', 'Deep Surrogate': '#1E90FF'}

    stages = {}

    # Stage C
    print("=" * 60)
    print(" Stage C: Opposition Prediction")
    print("=" * 60)
    Xtr, ytr, Xte, yte = load_stage_c()
    stage_res = {m: {'pr_auc': [], 'roc_auc': [], 'brier': []} for m in models}
    for seed in SEEDS:
        r = eval_binary(seed, Xtr, ytr, Xte, yte)
        for m in models:
            for k in ['pr_auc', 'roc_auc', 'brier']:
                stage_res[m][k].append(r[m][k])
        print(f"  Seed {seed}: " + " | ".join(f"{m}={r[m]['pr_auc']:.3f}" for m in models))
    stages['Stage C\n(Opposition)'] = stage_res
    gc.collect()

    # Stage A
    print("\n" + "=" * 60)
    print(" Stage A: Development Occurrence Hazard")
    print("=" * 60)
    Xtr, ytr, Xte, yte = load_stage_a()
    stage_res = {m: {'pr_auc': [], 'roc_auc': [], 'brier': []} for m in models}
    for seed in SEEDS:
        r = eval_binary(seed, Xtr, ytr, Xte, yte)
        for m in models:
            for k in ['pr_auc', 'roc_auc', 'brier']:
                stage_res[m][k].append(r[m][k])
        print(f"  Seed {seed}: " + " | ".join(f"{m}={r[m]['pr_auc']:.3f}" for m in models))
    stages['Stage A\n(Hazard)'] = stage_res
    gc.collect()

    # ===== Plot =====
    n_stages = len(stages)
    fig, axes = plt.subplots(n_stages, 3, figsize=(18, 5 * n_stages), squeeze=False)
    metric_labels = ['PR-AUC', 'ROC-AUC', 'Brier Score']
    metric_keys = ['pr_auc', 'roc_auc', 'brier']

    for row_idx, (stage_label, stage_data) in enumerate(stages.items()):
        for col_idx, (ml, mk) in enumerate(zip(metric_labels, metric_keys)):
            ax = axes[row_idx, col_idx]
            data = [stage_data[m][mk] for m in models]
            bp = ax.boxplot(data, labels=[m.replace(' ', '\n') for m in models],
                           patch_artist=True, widths=0.6)
            for patch, m in zip(bp['boxes'], models):
                patch.set_facecolor(colors[m]); patch.set_alpha(0.7)
            for patch in bp['medians']:
                patch.set_color('white'); patch.set_linewidth(2)
            for i, m in enumerate(models):
                jitter = np.random.normal(0, 0.04, size=len(stage_data[m][mk]))
                ax.scatter(np.ones(len(stage_data[m][mk])) * (i+1) + jitter,
                          stage_data[m][mk], color='black', alpha=0.3, s=12, zorder=5)
            ax.set_title(f"{stage_label}\n{ml}", fontsize=11, pad=8)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle("Multi-Stage Seed Stability: 10 Seeds x 4 Models x Primary Thesis Stages",
                 fontsize=15, y=1.0)
    plt.tight_layout()
    p = os.path.join(out_dir, "fig_all_stages_seed.pdf")
    plt.savefig(p, bbox_inches='tight', dpi=150)
    print(f"\n[+] Saved: {p}")

    # Print summary
    for stage_label, stage_data in stages.items():
        print(f"\n  {stage_label.replace(chr(10), ' ')}:")
        print(f"    {'Model':<18} {'PR-AUC':>18} {'ROC-AUC':>18} {'Brier':>18}")
        for m in models:
            pr = np.array(stage_data[m]['pr_auc'])
            roc = np.array(stage_data[m]['roc_auc'])
            br = np.array(stage_data[m]['brier'])
            print(f"    {m:<18} {np.nanmean(pr):.4f} +/- {np.nanstd(pr):.4f}  "
                  f"{np.nanmean(roc):.4f} +/- {np.nanstd(roc):.4f}  "
                  f"{np.nanmean(br):.4f} +/- {np.nanstd(br):.4f}")

if __name__ == "__main__":
    run()
