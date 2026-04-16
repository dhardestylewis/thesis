import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

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

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

class SimpleDeepTabular(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def plot_divergence_matrix():
    print("==========================================================")
    print(" Rendering Matrix: Attribution Divergence & Stability Test")
    print("==========================================================")
    
    ROOT = r"C:\Users\dhl\data\thesis\thesis"
    out_dir = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter4")
    os.makedirs(out_dir, exist_ok=True)
    
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
        'land_market_value': 'Property Valuation',
        'total_market_value': 'Property Valuation',
        'ldb_yr_built': 'Structure Age',
        'year_built': 'Structure Age',
        'year': 'Filing Timeline',
        'ldb_land_acres': 'Parcel Scale',
        'gross_site_area_acres': 'Parcel Scale',
        'deed_acreage': 'Parcel Scale',
        'ldb_lotsize': 'Parcel Scale',
        'ldb_land_use': 'Land Use',
        'lui_land_use': 'Land Use',
        'lui_general_land_use': 'Land Use',
        'protest': 'Historical Activity',
        'spatial_contagion_3yr': 'Historical Activity',
        'spatial_contagion_1yr': 'Historical Activity',
        'ldb_far': 'Zoning Density',
        'ldb_units': 'Zoning Density',
        'ldb_imprv_sqft': 'Improvement Scale',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    
    horizons = [('H0', 'H0_Filing_Master_Enriched.csv', 0), ('H3', 'H3_Filing_Master_NLP.csv', 1)]
    temporals = [('Pre-2022', lambda y: y < 2022, 0), ('Post-2022 (Regime Shift)', lambda y: y >= 2022, 1)]
    
    for hz_label, hz_file, row_idx in horizons:
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
        
        for temp_label, mask_fn, col_idx in temporals:
            ax = axes[row_idx, col_idx]
            
            mask = mask_fn(df['year'])
            sub_df = df_clean[mask]
            sub_y = df[mask][target_col].values
            X_num = sub_df.select_dtypes(include=[np.number]).fillna(0)
            
            # Model 1: Random Forest
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X_num, sub_y)
            rf_imp = dict(zip(X_num.columns, rf.feature_importances_))
            
            # Model 2: CatBoost
            cb = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=0, random_state=42)
            cb.fit(X_num, sub_y)
            cb_imp = dict(zip(X_num.columns, cb.feature_importances_))
            
            # Model 3: Deep Attentive Surrogate
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_num)
            X_t = torch.FloatTensor(X_s)
            y_t = torch.FloatTensor(sub_y).unsqueeze(1)
            
            deep_model = SimpleDeepTabular(X_s.shape[1])
            optimizer = torch.optim.Adam(deep_model.parameters(), lr=0.01)
            criterion = nn.BCEWithLogitsLoss()
            
            for _ in range(40):
                optimizer.zero_grad()
                out = deep_model(X_t)
                loss = criterion(out, y_t)
                loss.backward()
                optimizer.step()
                
            dp_w = deep_model.net[0].weight.data.abs().sum(dim=0).numpy()
            dp_imp = dict(zip(X_num.columns, dp_w))
            
            # Semantic Mapping
            rf_sem, cb_sem, dp_sem = {}, {}, {}
            for col in X_num.columns:
                cluster = SEMANTIC_CLUSTERS.get(col, 'Other Context')
                rf_sem[cluster] = rf_sem.get(cluster, 0) + rf_imp[col]
                cb_sem[cluster] = cb_sem.get(cluster, 0) + cb_imp[col]
                dp_sem[cluster] = dp_sem.get(cluster, 0) + dp_imp[col]
                
            for d in [rf_sem, cb_sem, dp_sem]:
                if 'Other Context' in d: del d['Other Context']
                tot = sum(d.values()) + 1e-9
                for k in d: d[k] = (d[k] / tot) * 100
                
            clusters = list(dp_sem.keys())
            # Sort globally by CatBoost importance so alignment is same across all 4 panels
            clusters = sorted(clusters, key=lambda k: cb_sem.get(k, 0))
            
            y_pos = np.arange(len(clusters))
            height = 0.25
            
            ax.barh(y_pos - height, [rf_sem.get(c,0) for c in clusters], height, label='Random Forest (Bagging)', color='gray', alpha=0.9)
            ax.barh(y_pos, [cb_sem.get(c,0) for c in clusters], height, label='CatBoost (Gradient Boosting)', color='darkred', alpha=0.9)
            ax.barh(y_pos + height, [dp_sem.get(c,0) for c in clusters], height, label='Deep Network (Spatial Embd.)', color='dodgerblue', alpha=0.9)
            
            ax.set_yticks(y_pos)
            if col_idx == 0:
                ax.set_yticklabels(clusters, fontsize=11)
            else:
                ax.set_yticklabels([])
                
            if row_idx == 0:
                ax.set_title(f"Evaluation Window: {temp_label}", fontsize=14, fontweight='bold', pad=15)
            if col_idx == 0:
                ax.set_ylabel(f"Horizon: {hz_label}", fontsize=14, fontweight='bold')
                
            if row_idx == 1 and col_idx == 1:
                ax.legend(loc='lower right', fontsize=11)
                
            ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.suptitle("Attribution Stability Test: Architectural Divergence Across Time and Development Horizons", fontsize=18, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    out_pdf = os.path.join(out_dir, "fig_attribution_divergence_matrix.pdf")
    plt.savefig(out_pdf)
    print(f"[+] Saved Attribution Divergence Matrix: {out_pdf}")

if __name__ == "__main__":
    plot_divergence_matrix()
