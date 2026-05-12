import sys, os, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import warnings
warnings.filterwarnings('ignore')

USE_GPU = torch.cuda.is_available()
CB_TASK  = "GPU" if USE_GPU else "CPU"
XGB_TREE = "gpu_hist" if USE_GPU else "hist"

from sklearn.base import BaseEstimator, ClassifierMixin

class NonLinearAnchorRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=10.0, n_anchors=None):
        self.gamma = gamma
        self.n_anchors = n_anchors
        self.model = CatBoostRegressor(iterations=100, depth=5, random_seed=42, verbose=0, task_type=CB_TASK)
        self.proj_X = LinearRegression(fit_intercept=False)
        self.proj_y = LinearRegression(fit_intercept=False)
        
    def fit(self, X_transformed, y, sample_weight=None):
        A = X_transformed[:, :self.n_anchors]
        self.proj_X.fit(A, X_transformed, sample_weight=sample_weight)
        self.proj_y.fit(A, y, sample_weight=sample_weight)
        
        X_P = self.proj_X.predict(A)
        y_P = self.proj_y.predict(A)
        
        factor = np.sqrt(self.gamma) - 1.0
        X_anc = X_transformed + factor * X_P
        y_anc = y + factor * y_P
        
        self.model.fit(X_anc, y_anc, sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        
        # Map causal importances from the backbone CatBoost
        importances = self.model.get_feature_importance()
        self.feature_importances_ = importances
        
        return self

    def predict_proba(self, X_transformed):
        preds = self.model.predict(X_transformed)
        preds = np.clip(preds, 0, 1)
        return np.vstack([1 - preds, preds]).T

ROOT = Path(r'c:\Users\dhl\data\thesis\thesis')
PANEL_PATH = ROOT / "Data/Panel/biweekly_panel.csv"

def build_target(df, window):
    if window == 1: return df['petition_event'].astype(int)
    return df.groupby('case_number')['petition_event'].transform(
        lambda x: x.iloc[::-1].rolling(window=window, min_periods=1).max().iloc[::-1]
    ).fillna(0).astype(int)

def format_grid(df, metric):
    lines = []
    eval_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    anchors = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    sort_order = ['CatBoost', 'XGBoost', 'Logistic (L2)', 'Spatial-FE Logistic', 'TabNet', 'Anchor Regression (Causal)']
    ordered_models = [m for m in sort_order if m in df['Model'].unique()] + [m for m in sorted(df['Model'].unique()) if m not in sort_order]
    
    for model in ordered_models:
        for anchor in [f'Pre-{y}' for y in anchors]:
            row_sub = df[(df['Model'] == model) & (df['Anchor'] == anchor)]
            if len(row_sub) == 0: continue
            r = []
            for test_year in eval_years:
                if test_year < int(anchor[-4:]): r.append("---")
                else:
                    cv = row_sub[row_sub['Evaluate_Year'] == test_year]
                    if len(cv) == 0: r.append("---")
                    else:
                        v = cv[metric].values[0]
                        all_vals = df[(df['Anchor'] == anchor) & (df['Evaluate_Year'] == test_year)][metric]
                        if len(all_vals) > 0 and v == all_vals.max():
                            r.append(f"\\textbf{{{v:.3f}}}")
                        else:
                            r.append(f"{v:.3f}")
            lines.append(f"{model} & {anchor} & " + " & ".join(r) + r" \\")
    return lines

def run():
    print("Loading biweekly panel...", flush=True)
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    df = df.sort_values(["case_number", "period_seq"]).reset_index(drop=True)
    
    first_petition = df[df["petition_event"] == 1].groupby("case_number")["period_seq"].min()
    df["first_petition_seq"] = df["case_number"].map(first_petition)
    df = df[df["first_petition_seq"].isna() | (df["period_seq"] <= df["first_petition_seq"])].drop(columns=["first_petition_seq"]).reset_index(drop=True)
    
    df['council_district'] = df['council_district'].fillna(1).astype(str)
    df['year_str'] = df['year'].astype(str)
    
    FEATS = [
        "period_seq", "bw_sin", "bw_cos",
        "council_hearings_this_period", "cumulative_council_hearings_lag1",
        "commission_hearings_this_period", "cumulative_commission_hearings_lag1",
        "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct",
        "market_value", "building_age", "land_acres",
        "total_population", "median_household_income",
        "renter_share", "rent_burden", "affordability_proxy",
        "race_white", "median_age",
        "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", "mortgage_rate_30yr_filing_delta",
        "treasury_10yr_yield", "treasury_10yr_yield_filing_delta",
        "fed_funds_rate", "fed_funds_rate_filing_delta",
        "local_unemployment_rate", "local_unemployment_rate_filing_delta",
        "knn_petition_rate_1km", "dist_petition_rate_lag1",
        "active_cases_100m", "active_cases_250m", "active_cases_500m",
        "active_cases_1km", "active_cases_2km", "active_gravity_index_t",
        "hearing_frequency", "petition_intensity_per_ft",
        "hearing_velocity_3p", "petition_velocity_3p",
        "pdf_requested_height_ft", "pdf_requested_max_far", "pdf_proposed_height_ft",
        "pdf_story_count", "pdf_story_height_ft", "pdf_compatibility_height_ft",
    ]
    feats = [f for f in FEATS if f in df.columns]
    
    X_raw_df = df[feats].fillna(0)
    X_raw = X_raw_df.values
    years = df['year'].values
    
    spatial_prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['council_district']),
        ('num', 'passthrough', feats)
    ])
    X_sp = spatial_prep.fit_transform(df[['council_district'] + feats].fillna(0))
    n_sp_dummies = len(spatial_prep.named_transformers_['cat'].get_feature_names_out())
    
    anchor_prep = ColumnTransformer([
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['council_district', 'year_str']),
        ('num', 'passthrough', feats)
    ])
    X_anc = anchor_prep.fit_transform(df[['council_district', 'year_str'] + feats].fillna(0))
    n_anc_dummies = len(anchor_prep.named_transformers_['cat'].get_feature_names_out())
    
    SEMANTIC_CLUSTERS = {
        'renter_share': 'Housing Tenure',
        'rent_burden': 'Housing Tenure',
        'race_white': 'Demographics',
        'total_population': 'Demographics',
        'median_age': 'Demographics',
        'median_household_income': 'Neighborhood Income',
        'affordability_proxy': 'Neighborhood Income',
        'market_value': 'Property Valuation',
        'building_age': 'Structure Age',
        'land_acres': 'Parcel Scale',
        'pdf_requested_height_ft': 'Zoning Density',
        'pdf_requested_max_far': 'Zoning Density',
        'pdf_proposed_height_ft': 'Zoning Density',
        'pdf_story_count': 'Zoning Density',
        'mortgage_rate_30yr': 'Macroeconomics',
        'fed_funds_rate': 'Macroeconomics',
        'treasury_10yr_yield': 'Macroeconomics',
        'local_unemployment_rate': 'Macroeconomics',
        'knn_petition_rate_1km': 'Spatial Contagion',
        'dist_petition_rate_lag1': 'Spatial Contagion',
        'active_cases_1km': 'Spatial Contagion',
        'cumulative_petition_count': 'Institutional Friction'
    }

    # Test all 5 analytical horizons to track predictive decay smoothly
    HORIZONS = {
        "14 Days": 1,
        "3 Months": 6,
        "6 Months": 13,
        "1 Year": 26,
        "2 Years": 52
    }
    
    anchors = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    eval_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    perf_dir = ROOT / "Thesis_Draft/GSAPP_Final_Submission/Tables/appendices_drift"
    fig_dir = ROOT / "Thesis_Draft/GSAPP_Final_Submission/Figures/exhibits"
    perf_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    total_models = len(HORIZONS) * len(anchors) * 6
    model_counter = 0
    attribution_matrix = []
    all_predictive_results = []
    
    import time
    start_time = time.time()
    
    for h_name, window in HORIZONS.items():
        print(f"\n========== HORIZON: {h_name} ==========")
        h_slug = h_name.replace(' ', '').lower()
        y = build_target(df, window).values
        
        predictive_results = []
        
        for anchor in anchors:
            train_mask = years < anchor
            if train_mask.sum() < 50: 
                model_counter += 6
                continue
            if y[train_mask].sum() < 5: 
                model_counter += 6
                continue
                
            X_train_raw = X_raw[train_mask]
            y_train = y[train_mask]
            
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train_raw)
            
            X_train_sp = X_sp[train_mask]
            sp_scaler = StandardScaler()
            X_train_sp_sc = sp_scaler.fit_transform(X_train_sp)
            
            X_train_anc = X_anc[train_mask]
            anc_scaler = StandardScaler()
            X_train_anc_sc = anc_scaler.fit_transform(X_train_anc)
            
            spw = max(1.0, (len(y_train) - y_train.sum()) / max(1, y_train.sum()))
            
            from sklearn.neural_network import MLPClassifier
            
            models = {
                'CatBoost': CatBoostClassifier(iterations=100, depth=6, verbose=0, random_seed=42, task_type=CB_TASK, scale_pos_weight=spw),
                'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss', device='cuda' if USE_GPU else 'cpu', tree_method='hist', scale_pos_weight=spw),
                'Logistic (L2)': LogisticRegression(class_weight='balanced', random_state=42, max_iter=200),
                'Spatial-FE Logistic': LogisticRegression(class_weight='balanced', random_state=42, max_iter=200),
                'Anchor Regression (Causal)': NonLinearAnchorRegression(gamma=10.0, n_anchors=n_anc_dummies),
                'Deep (MLP)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, random_state=42)
            }
            
            fitted_models = {}
            for name, m in models.items():
                model_counter += 1
                elapsed = time.time() - start_time
                tpm = elapsed / max(1, model_counter - 1)
                rem = (total_models - model_counter + 1) * tpm
                print(f"[{model_counter}/{total_models}] ETA: {rem/60:.1f}m | Training {name} (Pre-{anchor}, {h_name})...", flush=True)
                
                raw_imp = np.zeros(len(feats))
                
                try:
                    if name == 'Logistic (L2)': 
                        m.fit(X_train_sc, y_train)
                        raw_imp = np.abs(m.coef_[0])
                    elif name == 'Spatial-FE Logistic': 
                        m.fit(X_train_sp_sc, y_train)
                        raw_imp = np.abs(m.coef_[0][n_sp_dummies:])
                    elif name == 'Anchor Regression (Causal)':
                        from sklearn.isotonic import IsotonicRegression
                        m.fit(X_train_anc_sc, y_train)
                        raw_train_preds = m.predict_proba(X_train_anc_sc)[:, 1]
                        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
                        iso.fit(raw_train_preds, y_train)
                        m.iso = iso
                        raw_imp = np.abs(m.feature_importances_[n_anc_dummies:])
                    elif name == 'Deep (MLP)': 
                        m.fit(X_train_sc, y_train)
                        raw_imp = np.abs(m.coefs_[0]).sum(axis=1)
                    else: 
                        m.fit(X_train_raw, y_train)
                        if name == 'CatBoost': raw_imp = m.get_feature_importance()
                        else: raw_imp = m.feature_importances_
                        
                    fitted_models[name] = m
                    
                    # Store attributions
                    total = np.sum(raw_imp)
                    if total > 0: raw_imp = (raw_imp / total) * 100
                    
                    sem_map = {}
                    for f_name, imp in zip(feats, raw_imp):
                        grp = SEMANTIC_CLUSTERS.get(f_name, "Other")
                        if grp != "Other":
                            sem_map[grp] = sem_map.get(grp, 0) + imp
                    
                    vec = pd.Series(sem_map)
                    if vec.sum() > 0: vec = vec / vec.sum() * 100
                    
                    if name == 'Anchor Regression (Causal)': fam = 'Distributionally Robust'
                    elif 'Logistic' in name: fam = 'Regularized Linear'
                    elif 'Deep (MLP)' in name: fam = 'Deep'
                    else: fam = 'Tree'
                    
                    attribution_matrix.append({
                        'Family': fam, 'Model': name, 'Horizon': h_name, 'Anchor': anchor,
                        **vec.to_dict()
                    })
                    
                except Exception as e:
                    print(f"Failed to fit {name}: {e}")
            
            for test_year in eval_years:
                if test_year < anchor: continue
                test_mask = (years == test_year)
                if test_mask.sum() < 5 or y[test_mask].sum() < 1: continue
                    
                y_test = y[test_mask]
                X_test_raw = X_raw[test_mask]
                X_test_sc = scaler.transform(X_test_raw)
                X_test_sp_sc = sp_scaler.transform(X_sp[test_mask])
                X_test_anc_sc = anc_scaler.transform(X_anc[test_mask])
                
                base_rate = y_test.sum() / len(y_test)
                
                for name, m in fitted_models.items():
                    try:
                        if name == 'Logistic (L2)' or name == 'Deep (MLP)': p = m.predict_proba(X_test_sc)[:, 1]
                        elif name == 'Spatial-FE Logistic': p = m.predict_proba(X_test_sp_sc)[:, 1]
                        elif name == 'Anchor Regression (Causal)':
                            p_raw = m.predict_proba(X_test_anc_sc)[:, 1]
                            p = m.iso.predict(p_raw)
                        else: p = m.predict_proba(X_test_raw)[:, 1]
                        
                        prauc = average_precision_score(y_test, p)
                        lift = prauc / base_rate if base_rate > 0 else 0
                        
                        predictive_results.append({
                            'Model': name, 'Anchor': f'Pre-{anchor}',
                            'Evaluate_Year': test_year, 'PRAUC': prauc, 'Lift': lift
                        })
                    except Exception as e:
                        pass
        
        all_predictive_results.extend(predictive_results)
        print(f"[+] Wrote LaTeX tables for {h_name}")
        
    print("\n[+] Generating Global Attribution and Performance Artifacts...")
    attr_df = pd.DataFrame(attribution_matrix).fillna(0)
    perf_df = pd.DataFrame(all_predictive_results)
    
    attr_df.to_csv(perf_dir / "multihorizon_attribution_matrix_raw.csv", index=False)
    perf_df.to_csv(perf_dir / "multihorizon_performance_raw.csv", index=False)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import joblib
    
    # ---------------------------------------------------------
    # GLOBAL FIGURE 10: Comparative Primary Reliance
    # ---------------------------------------------------------
    cluster_cols = [c for c in attr_df.columns if c not in ['Family', 'Model', 'Horizon', 'Anchor']]
    agg_df = attr_df.groupby('Family')[cluster_cols].mean().T
    
    # Extract True Causal DML Importances
    print("[*] Extracting Causal DML Importances...")
    try:
        m_causal = joblib.load(ROOT / "Data/Zoning_Cases/causal_models_production.pkl")
        cf_joint = m_causal['cf_joint']
        
        ex_ante = [
            'Delta_Requested_Height', 'latitude', 'longitude',
            'median_household_income', 'race_white', 'race_black', 'race_hispanic',
            'renter_share', 'rent_burden', 'total_population', 'median_age',
            'appraised_value', 'building_age', 'mortgage_rate_30yr', 'fed_funds_rate',
            'local_unemployment_rate', 'knn_petition_rate_1km', 'dist_petition_rate_lag1',
            'fire_hazard_severity', 'slope_degree', 'is_imagine_corridor', 'petition_dose'
        ]
        
        if hasattr(cf_joint, 'feature_importances_'):
            try:
                c_imp = cf_joint.feature_importances_()
            except:
                c_imp = cf_joint.feature_importances_
            
            if c_imp.ndim == 2: c_imp = c_imp.mean(axis=0)
                
            c_sem_map = {}
            for f_name, imp in zip(ex_ante, c_imp):
                grp = SEMANTIC_CLUSTERS.get(f_name, "Other")
                if grp != "Other":
                    c_sem_map[grp] = c_sem_map.get(grp, 0) + imp
            
            vec = pd.Series(c_sem_map)
            if vec.sum() > 0: vec = vec / vec.sum() * 100
            
            vec = vec.reindex(agg_df.index).fillna(0)
            agg_df['Causal Forest DML'] = vec
    except Exception as e:
        print(f"Failed to load Causal DML: {e}")
    
    agg_df['Total'] = agg_df.sum(axis=1)
    agg_df = agg_df.sort_values('Total', ascending=True).drop(columns=['Total'])
    agg_df = agg_df[agg_df.sum(axis=1) > 0]
    
    ax = agg_df.plot(kind='barh', figsize=(10, 8), width=0.8, colormap='viridis')
    plt.title("Comparative Primary Reliance: Top Feature Clusters Across Architectures", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Average Attribution Reliance (%)", fontsize=11)
    plt.ylabel("Semantic Target Cluster", fontsize=11)
    plt.legend(title="Model Family", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    out_fig = fig_dir / "fig_ch5_reliance_bar.pdf"
    plt.savefig(out_fig)
    print(f"[+] Saved Figure 10 to: {out_fig}")
    plt.close()
    
    # ---------------------------------------------------------
    # 6-MONTH HORIZON: Table 12, Table 13, and Figure 12
    # ---------------------------------------------------------
    print("\n[*] Generating 6-Month specific Archetypal Tables and Weighted Clustermap...")
    attr_6m = attr_df[attr_df['Horizon'] == '6 Months'].copy()
    perf_6m = perf_df[perf_df['Horizon'] == '6 Months'].copy()
    
    if len(attr_6m) > 0 and len(perf_6m) > 0:
        # TABLE 12: Raw Archetypal Family Attribution
        df_agg = attr_6m.groupby('Family')[cluster_cols].mean()
        c_order = df_agg.loc['Tree'].sort_values(ascending=False).index.tolist() if 'Tree' in df_agg.index else cluster_cols
        
        t12_lines = [
            r'\begin{table}[htbp]', r'\centering',
            r'\caption[Archetypal Family Attribution]{\textbf{Archetypal Family Attribution.} Average absolute model reliance allocated to each semantic feature cluster. Extracting structural predictive associations via 6-Month forecasting target across walk-forward origins.}',
            r'\label{tab:archetypal_attribution}', r'\resizebox{\textwidth}{!}{%',
            r'\begin{tabular}{lccc}', r'\toprule',
            r'\textbf{Semantic Target Cluster} & \textbf{Tree Ensembles} & \textbf{Deep Architectures} & \textbf{Linear Architectures} \\',
            r'\midrule'
        ]
        
        for c in c_order:
            t = df_agg.loc['Tree', c] if 'Tree' in df_agg.index else 0
            d = df_agg.loc['Deep', c] if 'Deep' in df_agg.index else 0
            l = df_agg.loc['Regularized Linear', c] if 'Regularized Linear' in df_agg.index else 0
            t_str, d_str, l_str = f"{t:.1f}\\%", f"{d:.1f}\\%", f"{l:.1f}\\%"
            max_val = max(t, d, l)
            if max_val > 0:
                if t == max_val: t_str = f"\\textbf{{{t_str}}}"
                elif d == max_val: d_str = f"\\textbf{{{d_str}}}"
                else: l_str = f"\\textbf{{{l_str}}}"
            t12_lines.append(f"{c} & {t_str} & {d_str} & {l_str} \\\\")
        
        t12_lines.extend([r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}'])
        with open(perf_dir / 'tbl_ch5_02_archetypal_attribution_6m.tex', 'w') as f: f.write('\n'.join(t12_lines))
        
        # TABLE 13: Performance-Weighted Archetypal
        mean_perf = perf_6m.groupby(['Model', 'Anchor'])['PRAUC'].mean().reset_index()
        mean_perf['label'] = mean_perf['Model'] + "_" + mean_perf['Anchor']
        weight_map = dict(zip(mean_perf['label'], mean_perf['PRAUC']))
        
        attr_6m['label'] = attr_6m['Model'] + "_Pre-" + attr_6m['Anchor'].astype(str)
        attr_6m['Weight'] = attr_6m['label'].map(weight_map).fillna(0)
        
        def w_avg(group_df):
            w = group_df['Weight']
            cols_to_drop = ['Weight', 'Family', 'Model', 'Horizon', 'Anchor', 'label']
            valid_cols = [x for x in group_df.columns if x not in cols_to_drop]
            if w.sum() == 0: return group_df[valid_cols].mean()
            return group_df[valid_cols].mul(w, axis=0).sum() / w.sum()
            
        df_agg_w = attr_6m.groupby('Family').apply(w_avg)
        
        t13_lines = [
            r'\begin{table}[htbp]', r'\centering',
            r'\caption[Performance-Weighted Archetypal Attribution]{\textbf{Performance-Weighted Archetypal Family Attribution.} Model attribution vectors scaled by their corresponding out-of-distribution longitudinal PR-AUC performance. Architectures that successfully generalized under domain drift dominate their archetypal class weights.}',
            r'\label{tab:archetypal_attribution_weighted}', r'\resizebox{\textwidth}{!}{%',
            r'\begin{tabular}{lccc}', r'\toprule',
            r'\textbf{Semantic Target Cluster} & \textbf{Tree Ensembles} & \textbf{Deep Architectures} & \textbf{Linear Architectures} \\',
            r'\midrule'
        ]
        
        for c in c_order:
            t = df_agg_w.loc['Tree', c] if 'Tree' in df_agg_w.index else 0
            d = df_agg_w.loc['Deep', c] if 'Deep' in df_agg_w.index else 0
            l = df_agg_w.loc['Regularized Linear', c] if 'Regularized Linear' in df_agg_w.index else 0
            t_str, d_str, l_str = f"{t:.1f}\\%", f"{d:.1f}\\%", f"{l:.1f}\\%"
            max_val = max(t, d, l)
            if max_val > 0:
                if t == max_val: t_str = f"\\textbf{{{t_str}}}"
                elif d == max_val: d_str = f"\\textbf{{{d_str}}}"
                else: l_str = f"\\textbf{{{l_str}}}"
            t13_lines.append(f"{c} & {t_str} & {d_str} & {l_str} \\\\")
            
        t13_lines.extend([r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}'])
        with open(perf_dir / 'tbl_ch5_03_archetypal_attribution_weighted_6m.tex', 'w') as f: f.write('\n'.join(t13_lines))
        
        # FIGURE 12: Performance-Weighted Meta-Attribution Clustermap
        attr_6m.set_index('label', inplace=True)
        attr_6m_vals = attr_6m[cluster_cols].copy()
        
        # Drop columns with zero variance
        attr_6m_vals = attr_6m_vals.loc[:, attr_6m_vals.var() > 0.0]
        
        # Apply performance weights
        wt_map = attr_6m['Weight'].to_dict()
        df_attr_w = attr_6m_vals.mul(attr_6m_vals.index.map(wt_map).fillna(0), axis=0)
        
        sns.set_theme(style='white')
        g_w = sns.clustermap(df_attr_w, cmap='mako', method='ward', metric='euclidean', figsize=(10, 15), linewidths=.5, annot=True, fmt=".1f", annot_kws={"size": 8})
        g_w.fig.suptitle("Performance-Weighted Meta-Attribution Clustering (6-Month Horizon)", fontsize=16, fontweight='bold', y=1.02)
        g_w.ax_heatmap.set_xlabel("Semantic Feature Clusters", fontsize=12)
        g_w.ax_heatmap.set_ylabel("Environment", fontsize=12)
        
        out_path_fig_w = fig_dir / "fig_ch5_meta_attribution_clustermap_weighted.pdf"
        g_w.savefig(out_path_fig_w, bbox_inches='tight')
        plt.close()
        
        print(f"[+] Saved 6-Month Archetypal Tables and Figure 12 Clustermap.")

if __name__ == '__main__':
    run()

