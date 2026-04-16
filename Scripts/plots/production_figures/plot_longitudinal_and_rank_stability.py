import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

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

def extract_attribution(X, y_pred, model_type):
    if model_type == 'tree':
        cb = CatBoostClassifier(iterations=30, learning_rate=0.1, verbose=0, random_state=42)
        cb.fit(X, y_pred)
        return dict(zip(X.columns, cb.feature_importances_))
    elif model_type == 'linear':
        lr = LogisticRegression(penalty='l2', max_iter=200, random_state=42)
        lr.fit(X, y_pred)
        w = np.abs(lr.coef_).flatten()
        return dict(zip(X.columns, w))
    elif model_type == 'causal':
        ridge = RidgeClassifier(alpha=1.0, random_state=42)
        ridge.fit(X, y_pred)
        w = np.abs(ridge.coef_).flatten()
        return dict(zip(X.columns, w))
    elif model_type == 'deep':
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        X_t = torch.FloatTensor(X_s)
        y_t = torch.FloatTensor(y_pred).unsqueeze(1)
        
        deep = SimpleDeepTabular(X_s.shape[1])
        optimizer = torch.optim.Adam(deep.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        
        for _ in range(40):
            optimizer.zero_grad()
            out = deep(X_t)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()
            
        w = deep.net[0].weight.data.abs().sum(dim=0).numpy()
        return dict(zip(X.columns, w))

def generate_graphics():
    ROOT = r"C:\Users\dhl\data\thesis\thesis"
    out_dir = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "ch4")
    out_dir_ex = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "exhibits")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_ex, exist_ok=True)

    data_path = os.path.join(ROOT, "Data", "Warehouse_As_Of", "canonical", "H0_Filing_Master_Enriched_v2.csv")
    if not os.path.exists(data_path):
        print(f"[!] Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path, low_memory=False)
    target_col = 'is_protested' if 'is_protested' in df.columns else 'protest'
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    SEMANTIC_CLUSTERS = {
        'acs_owner_occupied_units': 'Housing Tenure',
        'acs_renter_occupied_units': 'Housing Tenure',
        'acs_total_housing_units': 'Housing Tenure',
        'acs_race_white': 'Demographics',
        'acs_race_hispanic': 'Demographics',
        'acs_race_black': 'Demographics',
        'acs_race_asian': 'Demographics',
        'acs_median_gross_rent': 'Neighborhood Income',
        'acs_median_household_income': 'Neighborhood Income',
        'acs_poverty_count': 'Neighborhood Income',
        'acs_median_home_value': 'Neighborhood Income',
        'ldb_appraised_val': 'Property Valuation',
        'ldb_market_val': 'Property Valuation',
        'total_market_value': 'Property Valuation',
        'ldb_yr_built': 'Structure Age',
        'year_built': 'Structure Age',
        'ldb_land_acres': 'Parcel Scale',
        'gross_site_area_acres': 'Parcel Scale',
        'deed_acreage': 'Parcel Scale',
        'ldb_lotsize': 'Parcel Scale',
        'ldb_land_use': 'Land Use',
        'protest': 'Historical Activity',
        'spatial_contagion_3yr': 'Historical Activity',
        'spatial_contagion_1yr': 'Historical Activity',
        'ldb_far': 'Zoning Density',
        'ldb_units': 'Zoning Density',
        'ldb_imprv_sqft': 'Improvement Scale'
    }

    drop_cols = [target_col, 'case_number', 'council_district_x', 'TCAD ID', 'protest']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df_clean = df_clean.drop(columns=[c for c in df_clean.columns if c.startswith('tfidf_') or c.startswith('speech_') or 'date' in c.lower()])

    models = ['tree', 'deep', 'linear', 'causal']
    model_labels = {'tree': 'Tree Ensembles', 'deep': 'Deep Architectures', 'linear': 'Linear Methods', 'causal': 'Causal Algorithms'}
    colors = {'tree': 'darkred', 'deep': 'dodgerblue', 'linear': 'forestgreen', 'causal': 'purple'}
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    temporal_tracking = {m: {yr: {} for yr in years} for m in models}

    for yr in years:
        mask = (df['year'] == yr) | (df['year'] == yr-1)
        sub_df = df_clean[mask]
        sub_y = df[mask][target_col].values
        X_num = sub_df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if len(X_num) < 50 or len(np.unique(sub_y)) < 2:
            continue
            
        for m in models:
            raw_imp = extract_attribution(X_num, sub_y, m)
            sem = {}
            for col, val in raw_imp.items():
                cluster = SEMANTIC_CLUSTERS.get(col, 'Other Context')
                sem[cluster] = sem.get(cluster, 0) + val
            if 'Other Context' in sem: del sem['Other Context']
            
            tot = sum(sem.values()) + 1e-9
            for k in sem: sem[k] = (sem[k] / tot) * 100
            temporal_tracking[m][yr] = sem

    # Figure 11: Longitudinal Divergence (Line Plot)
    plt.figure(figsize=(12, 7))
    styles = {'Demographics': '-', 'Parcel Scale': '--'}
    
    for m in models:
        y_demo = [temporal_tracking[m][yr].get('Demographics', 0) for yr in years]
        y_parcel = [temporal_tracking[m][yr].get('Parcel Scale', 0) for yr in years]
        plt.plot(years, y_demo, color=colors[m], linestyle='-', linewidth=2.5, marker='o', label=f"{model_labels[m]} (Demographics)")
        plt.plot(years, y_parcel, color=colors[m], linestyle='--', linewidth=2, marker='s', alpha=0.7, label=f"{model_labels[m]} (Parcel Scale)")

    plt.title("Attribution Stability Test: Architectural Divergence Across Time", fontsize=16, pad=15)
    plt.xlabel("Terminal Observation Year", fontsize=12)
    plt.ylabel("Relative Attribution Share (%)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    f11_out = os.path.join(out_dir, "fig_ch4_30_attribution_longitudinal.pdf")
    plt.savefig(f11_out, bbox_inches='tight')
    plt.close()

    # Figure 12: Rank Stability (Bar Chart)
    rank_scores = {m: [] for m in models}
    valid_years = [y for y in years if len(temporal_tracking['tree'][y]) > 0]
    
    for m in models:
        for i in range(len(valid_years)-1):
            y1, y2 = valid_years[i], valid_years[i+1]
            c1, c2 = temporal_tracking[m][y1], temporal_tracking[m][y2]
            keys = list(set(c1.keys()).union(set(c2.keys())))
            v1 = [c1.get(k, 0) for k in keys]
            v2 = [c2.get(k, 0) for k in keys]
            rho, _ = spearmanr(v1, v2)
            if not np.isnan(rho):
                rank_scores[m].append(rho)

    mean_rhos = {m: np.mean(rank_scores[m]) for m in models}
    
    plt.figure(figsize=(8, 6))
    x_pos = np.arange(len(models))
    bars = plt.bar(x_pos, [mean_rhos[m] for m in models], color=[colors[m] for m in models], alpha=0.85, edgecolor='black', linewidth=1)
    
    plt.title("Adjacent-Window Attribution Rank Stability", fontsize=15, pad=15)
    plt.ylabel("Mean Spearman Rank Correlation (ρ)", fontsize=12)
    plt.xticks(x_pos, [model_labels[m] for m in models], fontsize=11)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() - 0.08,
                 f"{bar.get_height():.3f}", ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)

    plt.tight_layout()
    f12_out = os.path.join(out_dir_ex, "fig_attribution_rank_stability_H0.pdf")
    plt.savefig(f12_out, bbox_inches='tight')
    plt.close()

    print(f"[+] Saved Figure 11: {f11_out}")
    print(f"[+] Saved Figure 12: {f12_out}")

if __name__ == '__main__':
    generate_graphics()
