import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
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

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

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

def plot_longitudinal():
    print("===================================================================")
    print(" Rendering Longitudinal Attribution Divergence Matrix (2018-2024)")
    print("===================================================================")
    
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
        'acs_median_gross_rent': 'Nbrhood Income',
        'acs_median_household_income': 'Nbrhood Income',
        'acs_poverty_count': 'Nbrhood Income',
        'acs_median_home_value': 'Nbrhood Income',
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
        'ldb_imprv_sqft': 'Imprv Scale',
    }

    horizons = [('H0', 'H0_Filing_Master_Enriched.csv', 0), ('H3', 'H3_Filing_Master_NLP.csv', 1)]
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    fig, axes = plt.subplots(len(years), 2, figsize=(14, 18), sharex=True)
    
    for hz_label, hz_file, col_idx in horizons:
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
        
        for row_idx, yr in enumerate(years):
            ax = axes[row_idx, col_idx]
            
            # Use rolling 2-year window for stability
            # Alternatively, pinpoint specific year. (We'll pinpoint exact year to answer the prompt strictly).
            # To ensure sufficient sample size, use [yr-1, yr].
            mask = (df['year'] == yr) | (df['year'] == yr-1)
            
            sub_df = df_clean[mask]
            sub_y = df[mask][target_col].values
            X_num = sub_df.select_dtypes(include=[np.number]).fillna(0)
            
            if len(X_num) < 10:
                ax.set_title(f"{yr}: Insufficient Data", fontsize=10)
                continue
                
            cb = CatBoostClassifier(iterations=30, learning_rate=0.1, verbose=0, random_state=42)
            cb.fit(X_num, sub_y)
            cb_imp = dict(zip(X_num.columns, cb.feature_importances_))
            
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
            
            cb_sem, dp_sem = {}, {}
            for col in X_num.columns:
                cluster = SEMANTIC_CLUSTERS.get(col, 'Other Context')
                cb_sem[cluster] = cb_sem.get(cluster, 0) + cb_imp[col]
                dp_sem[cluster] = dp_sem.get(cluster, 0) + dp_imp[col]
                
            for d in [cb_sem, dp_sem]:
                if 'Other Context' in d: del d['Other Context']
                tot = sum(d.values()) + 1e-9
                for k in d: d[k] = (d[k] / tot) * 100
                
            clusters = list(dp_sem.keys())
            clusters = sorted(clusters, key=lambda k: cb_sem.get(k, 0))
            
            y_pos = np.arange(len(clusters))
            height = 0.35
            
            ax.barh(y_pos - height/2, [cb_sem.get(c,0) for c in clusters], height, label='CatBoost (Boosting)', color='darkred', alpha=0.9)
            ax.barh(y_pos + height/2, [dp_sem.get(c,0) for c in clusters], height, label='Deep Net (Proxy)', color='dodgerblue', alpha=0.9)
            
            ax.set_yticks(y_pos)
            if col_idx == 0:
                ax.set_yticklabels(clusters, fontsize=10)
            else:
                ax.set_yticklabels([])
                
            if row_idx == 0:
                ax.set_title(f"Horizon: {hz_label}", fontsize=14, fontweight='bold', pad=15)
            
            ax.set_ylabel(f"{yr}", fontsize=12, fontweight='bold')
                
            if row_idx == 4 and col_idx == 1:
                ax.legend(loc='lower right', fontsize=11)
                
            ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.suptitle("Longitudinal Stability: Architectural Divergence (2018-2024)", fontsize=18, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    out_pdf = os.path.join(out_dir, "fig_attribution_longitudinal.pdf")
    plt.savefig(out_pdf)
    print(f"[+] Saved Longitudinal Matrix: {out_pdf}")

if __name__ == "__main__":
    plot_longitudinal()
