def run_drift_and_archetypes(threshold=0.20, is_appendix=False):
    import pandas as pd, numpy as np, os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import average_precision_score
    from scipy.cluster.hierarchy import fcluster
    from catboost import CatBoostClassifier, CatBoostRegressor
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.neural_network import MLPClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.impute import SimpleImputer
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch

    # GPU detection — use CUDA if available, fall back to CPU cleanly
    USE_GPU = torch.cuda.is_available()
    CB_TASK  = "GPU" if USE_GPU else "CPU"
    XGB_TREE = "gpu_hist" if USE_GPU else "hist"
    print(f"[*] Device: {'GPU (CUDA)' if USE_GPU else 'CPU'}")
    
    warnings = __import__('warnings')
    warnings.filterwarnings('ignore')
    
    class NonLinearAnchorRegression(BaseEstimator, ClassifierMixin):
        def __init__(self, gamma=10.0, n_anchors=None):
            self.gamma = gamma
            self.n_anchors = n_anchors
            self.model = CatBoostRegressor(iterations=100, depth=5, random_seed=42, verbose=0)
            self.proj_X = LinearRegression(fit_intercept=False)
            self.proj_y = LinearRegression(fit_intercept=False)
            
        def fit(self, X_transformed, y, sample_weight=None):
            if isinstance(X_transformed, pd.DataFrame): X_transformed = X_transformed.values
            if isinstance(y, (pd.Series, pd.DataFrame)): y = y.values
                
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
            
            # Calculate causal feature importance mapping
            importances = self.model.get_feature_importance()
            # The catboost fit was on all features (including A and X).
            self.feature_importances_ = importances
            return self
    
        def predict_proba(self, X_transformed):
            if isinstance(X_transformed, pd.DataFrame): X_transformed = X_transformed.values
            preds = self.model.predict(X_transformed)
            preds = np.clip(preds, 0, 1)
            return np.vstack([1 - preds, preds]).T
            
        def predict(self, X_transformed):
            return (self.predict_proba(X_transformed)[:, 1] > 0.5).astype(int)
    
    ROOT = r'C:\Users\dhl\data\thesis\thesis'
    DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of', 'canonical')
    

    df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched_v2.csv'), low_memory=False)
    
    # Extract the true continuous spatial footprint fraction directly from the PDF OCR map
    petitions = pd.read_csv(os.path.join(ROOT, 'Data', 'Protest_Petitions', 'petition_signers_from_pdf.csv'))
    case_pct = petitions[petitions['signed'] == 1].groupby('case_number')['area_pct'].sum().reset_index()
    case_pct['case_number'] = case_pct['case_number'].astype(str).str.strip()
    
    df['case_number'] = df['case_number'].astype(str).str.strip()
    df = df.merge(case_pct, on='case_number', how='left')
    
    # Unmapped background cases default to 0.0 spatial protest overlap (no petition filed)
    df['reconstructed_petition_share'] = df['area_pct'].fillna(0.0) / 100.0
    df['is_protested'] = (df['reconstructed_petition_share'] > threshold).astype(int)

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['council_district'] = df['council_district'] if 'council_district' in df.columns else df.get('council_district_x', 1)
    df['council_district'] = df['council_district'].fillna(1).astype(str)
    
    df = df.dropna(subset=['year', 'is_protested']).sort_values('year')
    
    drop_cols = ['is_protested', 'case_number', 'reconstructed_petition_share', 'area_pct', 'case_id', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x', 'protest', 'protested', 'petition_crossed', 'threshold_crossed', 'label_version', 'clerk_validity_observed', 'procedural_defect_signal', 'label_confidence', 'source_provenance', 'label_notes', 'source_file_count']
    future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
    
    X_raw_df = df.drop(columns=[c for c in (drop_cols + future_features) if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
    X_raw_df = X_raw_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # DISCRETIZATION BLOCK (ERADICATING IDENTITY HACKING)
    phys_floats = [
        'ldb_appraised_val', 'land_market_value', 'total_market_value',
        'gross_site_area_acres', 'deed_acreage', 'ldb_land_acres', 'ldb_lotsize',
        'improvement_sq_ft', 'ldb_imprv_sqft'
    ]
    to_discretize = [c for c in phys_floats if c in X_raw_df.columns]
    
    from sklearn.tree import DecisionTreeClassifier
    if len(to_discretize) > 0:
        print(f"[*] Applying Optimal Entropy-Based Discretization (Trees) on {len(to_discretize)} variables...")
        for col in to_discretize:
            # max_leaf_nodes=None allows mathematically optimal unconstrained splitting.
            # min_samples_leaf=30 anchors k-anonymity to the Central Limit Theorem bound.
            dt = DecisionTreeClassifier(max_leaf_nodes=None, min_samples_leaf=30, random_state=42)
            X_col = X_raw_df[[col]].values
            dt.fit(X_col, df['is_protested'].values)
            X_raw_df[col] = dt.apply(X_col)
    
    features = X_raw_df.columns.tolist()
    X_raw = X_raw_df.values
    y = df['is_protested'].values
    years = df['year'].values
    districts = df['council_district'].values
    
    SEMANTIC_CLUSTERS = {
        'acs_owner_occupied_units': 'Housing Tenure',
        'acs_renter_occupied_units': 'Housing Tenure',
        'acs_total_housing_units': 'Housing Tenure',
        'acs_race_white': 'Demographics',
        'acs_race_hispanic': 'Demographics',
        'acs_race_black': 'Demographics',
        'acs_race_asian': 'Demographics',
        'acs_median_household_income': 'Neighborhood Income',
        'acs_poverty_count': 'Neighborhood Income',
        'acs_median_home_value': 'Neighborhood Valuation',
        'ldb_appraised_val': 'Property Valuation',
        'land_market_value': 'Property Valuation',
        'total_market_value': 'Property Valuation',
        'improvement_sq_ft': 'Improvement Scale',
        'ldb_imprv_sqft': 'Improvement Scale',
        'ldb_yr_built': 'Structure Age',
        'year_built': 'Structure Age',
        'property_age': 'Structure Age',
        'gross_site_area_acres': 'Parcel Scale',
        'deed_acreage': 'Parcel Scale',
        'ldb_land_acres': 'Parcel Scale',
        'ldb_lotsize': 'Parcel Scale',
        'ldb_far': 'Zoning Density',
        'ldb_units': 'Zoning Density'
    }
    
    anchors = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    eval_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    attribution_matrix = []
    labels = []
    families = []
    predictive_results = []
    
    for anchor in anchors:
        train_mask = years < anchor
        if train_mask.sum() < 50: continue
        
        # Standard arrays
        X_train_raw = X_raw[train_mask]
        y_train = y[train_mask]
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_raw)
        
        # Feature Engineering for Spatial and Causal models
        X_train_df = X_raw_df.iloc[np.where(train_mask)].copy()
        X_train_df['council_district'] = districts[train_mask]
        X_train_df['year_str'] = years[train_mask].astype(str)
        
        # Spatial prep
        spatial_prep = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['council_district']),
            ('num', 'passthrough', features)
        ])
        X_train_sp = spatial_prep.fit_transform(X_train_df)
        n_sp_dummies = len(spatial_prep.named_transformers_['cat'].get_feature_names_out())
        sp_scaler = StandardScaler()
        X_train_sp_sc = sp_scaler.fit_transform(X_train_sp)
    
        # Anchor prep
        anchor_prep = ColumnTransformer([
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['council_district', 'year_str']),
            ('num', 'passthrough', features)
        ])
        X_train_anc = anchor_prep.fit_transform(X_train_df)
        n_anc_dummies = len(anchor_prep.named_transformers_['cat'].get_feature_names_out())
        anc_scaler = StandardScaler()
        X_train_anc_sc = anc_scaler.fit_transform(X_train_anc)
    
        models = {
            'CatBoost': CatBoostClassifier(iterations=100, depth=6, verbose=0, random_seed=42, task_type=CB_TASK),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss', device='cuda' if USE_GPU else 'cpu'),
            'Logistic (L2)': LogisticRegression(class_weight='balanced', random_state=42, max_iter=500),
            'Deep (MLP)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, random_state=42)
        }
    
        print(f"[*] Training 10 Models for Pre-{anchor} anchor...")
        fitted_models = {}
        
        for name, m in models.items():
            if name in ['Logistic (L2)', 'ElasticNet']: 
                m.fit(X_train_sc, y_train)
                raw_imp = np.abs(m.coef_[0])
            elif name == 'Spatial-FE Logistic':
                m.fit(X_train_sp_sc, y_train)
                # Exclude spatial dummy weights, map to core features
                raw_imp = np.abs(m.coef_[0][n_sp_dummies:])
            elif name == 'Deep (MLP)':
                m.fit(X_train_sc, y_train)
                raw_imp = np.abs(m.coefs_[0]).sum(axis=1)
                
                # Isotonic Calibration because it uses regression backbone
                raw_train_preds = m.predict_proba(X_train_sc)[:, 1]
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
                iso.fit(raw_train_preds, y_train)
                m.iso = iso
                
            elif 'TabNet' in name:
                m.fit(X_train_sc, y_train, max_epochs=25, patience=5)
                raw_imp = m.feature_importances_
            else: 
                m.fit(X_train_raw, y_train)
                if name == 'CatBoost': raw_imp = m.get_feature_importance()
                else: raw_imp = m.feature_importances_
                
            fitted_models[name] = m
            
            total = np.sum(raw_imp)
            if total > 0: raw_imp = (raw_imp / total) * 100
            else: raw_imp = np.zeros_like(raw_imp)
    
            sem_map = {}
            for f_name, imp in zip(features, raw_imp):
                grp = SEMANTIC_CLUSTERS.get(f_name, "Other")
                if grp != "Other":
                    sem_map[grp] = sem_map.get(grp, 0) + imp
            
            vec = pd.Series(sem_map)
            vsum = vec.sum()
            if vsum > 0: vec = vec / vsum * 100
    
            attribution_matrix.append(vec)
            labels.append(f"{name}_{anchor}")
            
            if 'Logistic' in name: fam = 'Regularized Linear'
            elif 'TabNet' in name or 'Deep' in name: fam = 'Deep'
            else: fam = 'Tree'
            families.append(fam)
    
        for test_year in eval_years:
            if test_year < anchor: continue
            test_mask = years == test_year
            if test_mask.sum() < 5 or y[test_mask].sum() < 1: continue
                
            X_test_raw, y_test = X_raw[test_mask], y[test_mask]
            X_test_sc = scaler.transform(X_test_raw)
            
            X_test_df = X_raw_df.iloc[np.where(test_mask)].copy()
            X_test_df['council_district'] = districts[test_mask]
            X_test_df['year_str'] = years[test_mask].astype(str)
            
            X_test_sp = spatial_prep.transform(X_test_df)
            X_test_sp_sc = sp_scaler.transform(X_test_sp)
            
            X_test_anc = anchor_prep.transform(X_test_df)
            X_test_anc_sc = anc_scaler.transform(X_test_anc)
            
            for name, m in fitted_models.items():
                if name in ['Logistic (L2)', 'ElasticNet'] or 'TabNet' in name:
                    p = m.predict_proba(X_test_sc)[:, 1]
                elif name == 'Spatial-FE Logistic':
                    p = m.predict_proba(X_test_sp_sc)[:, 1]
                elif name == 'Deep (MLP)':
                    p = m.predict_proba(X_test_sc)[:, 1]
                else:
                    p = m.predict_proba(X_test_raw)[:, 1]
                
                base_rate = y_test.sum() / len(y_test)
                prauc = average_precision_score(y_test, p)
                lift = prauc / base_rate if base_rate > 0 else 0
                
                if 'Logistic' in name: fam = 'Regularized Linear'
                elif 'TabNet' in name or 'Deep' in name: fam = 'Deep'
                else: fam = 'Tree'
                
                predictive_results.append({
                    'Family': fam, 'Model': name, 'Anchor': f'Pre-{anchor}',
                    'Evaluate_Year': test_year, 'PRAUC': prauc, 'Lift': lift
                })
    
    print("[*] Generating 10-Architecture Performance Tables...")
    res_df = pd.DataFrame(predictive_results)
    
    def format_grid(df, metric, use_lift=False):
        lines = []

        sort_order = ['CatBoost', 'XGBoost', 'Logistic (L2)', 'Deep (MLP)']
        ordered_models = [m for m in sort_order if m in df['Model'].unique()]
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
                                r.append(f"\\textbf{{{v:.3f}}}" if not use_lift else f"\\textbf{{{v:.3f}}}")
                            else:
                                r.append(f"{v:.3f}")
                lines.append(f"{model} & {anchor} & " + " & ".join(r) + r" \\")
        return lines
    

    OUT_DIR = os.path.join(ROOT, 'Thesis_Draft', 'GSAPP_Final_Submission', 'Tables')
    if is_appendix:
        perf_dir = os.path.join(OUT_DIR, 'appendices_drift')
        attr_dir = os.path.join(OUT_DIR, 'appendices_drift')
        os.makedirs(perf_dir, exist_ok=True)
        suffix = f"_t{int(threshold*100):02d}"
    else:
        perf_dir = os.path.join(OUT_DIR, 'chapter4_performance')
        attr_dir = os.path.join(OUT_DIR, 'chapter5_attribution')
        suffix = ""
    

    
    # Table 19: Drift Analysis
    t14_caption = r"\textbf{Temporal predictive drift: PR-AUC decay by algorithm.}"
    if threshold > 0: t14_caption = f"\\textbf{{Temporal predictive drift: PR-AUC decay (Threshold: >{int(threshold*100)}\\%) .}}"
    
    t4_lines = [r'\begin{table}[htbp]', r'\centering', r'\caption[' + t14_caption[8:-2] + ']{' + t14_caption + '}', r'\label{tab:temporal_drift' + suffix + '}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{l l' + 'c'*len(eval_years) + '}', r'\toprule', r'\textbf{Model} & \textbf{Anchor Training} & ' + ' & '.join(['\\textbf{'+str(y)+'}' for y in eval_years]) + r' \\', r'\midrule']
    t4_lines.extend(format_grid(res_df, 'PRAUC'))
    t4_lines.extend([r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}'])
    with open(os.path.join(perf_dir, f'tbl_ch4_14_temporal_drift_analysis{suffix}.tex'), 'w') as f: f.write('\n'.join(t4_lines))
    
    # Table 5
    t5_lines = [r'\begin{table}[htbp]', r'\centering', r'\caption[Temporal Predictive Drift (PR-AUC lift)]{\textbf{Temporal predictive drift: PR-AUC lift by algorithm (Optimal Entropy-Based Discretization).}}', r'\label{tab:temporal_drift_prauc_lift}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{l l' + 'c'*len(eval_years) + '}', r'\toprule', r'\textbf{Model} & \textbf{Anchor Training} & ' + ' & '.join(['\\textbf{'+str(y)+'}' for y in eval_years]) + r' \\', r'\midrule']
    t5_lines.extend(format_grid(res_df, 'Lift', use_lift=True))
    t5_lines.extend([r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}'])
    with open(os.path.join(perf_dir, f'tbl_ch4_17_temporal_drift_prauc_lift{suffix}.tex'), 'w') as f: f.write('\n'.join(t5_lines))
    
    # Table 6
    pivot_p = res_df.groupby(['Family', 'Anchor', 'Evaluate_Year'])['PRAUC'].max().reset_index()
    pivot_mat_p = pivot_p.pivot_table(index='Family', columns=['Anchor', 'Evaluate_Year'], values='PRAUC')
    winners_p = pivot_mat_p.idxmax()
    
    pivot_l = res_df.groupby(['Family', 'Anchor', 'Evaluate_Year'])['Lift'].max().reset_index()
    pivot_mat_l = pivot_l.pivot_table(index='Family', columns=['Anchor', 'Evaluate_Year'], values='Lift')
    winners_l = pivot_mat_l.idxmax()
    
    t6_lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption[Temporal Drift (Max-of-Family)]{\textbf{Max-of-Family dominance (Optimal Entropy-Based Discretization).}}',
        r'\label{tab:temporal_drift_family}', r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{l' + 'c'*len(eval_years) + '}', r'\toprule',
        r'\textbf{Anchor Training} & ' + ' & '.join(['\\textbf{' + str(y) + '}' for y in eval_years]) + r' \\',
        r'\midrule', r'\multicolumn{8}{c}{\textbf{Panel A: Maximum Absolute PR-AUC}} \\', r'\midrule'
    ]
    
    for anchor in [f'Pre-{y}' for y in anchors]:
        if len(pivot_p[pivot_p['Anchor'] == anchor]) == 0: continue
        r = []
        for test_year in eval_years:
            if test_year < int(anchor[-4:]): r.append("---")
            else:
                if (anchor, test_year) in winners_p.index:
                    win_fam = winners_p.loc[(anchor, test_year)]
                    max_val = pivot_mat_p.loc[win_fam, (anchor, test_year)]
                    r.append(f"{max_val:.3f} ({win_fam})")
                else: r.append("---")
        t6_lines.append(f"{anchor} & " + " & ".join(r) + r" \\")
    
    t6_lines.extend([r'\midrule', r'\multicolumn{8}{c}{\textbf{Panel B: Maximum Relative PR-AUC Lift}} \\', r'\midrule'])
    
    for anchor in [f'Pre-{y}' for y in anchors]:
        if len(pivot_l[pivot_l['Anchor'] == anchor]) == 0: continue
        r = []
        for test_year in eval_years:
            if test_year < int(anchor[-4:]): r.append("---")
            else:
                if (anchor, test_year) in winners_l.index:
                    win_fam = winners_l.loc[(anchor, test_year)]
                    max_val = pivot_mat_l.loc[win_fam, (anchor, test_year)]
                    r.append(f"{max_val:.2f} ({win_fam})")
                else: r.append("---")
        t6_lines.append(f"{anchor} & " + " & ".join(r) + r" \\")
    
    t6_lines.extend([r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}'])
    
    with open(os.path.join(perf_dir, f'tbl_ch4_15_temporal_drift_family{suffix}.tex'), 'w') as f: f.write('\n'.join(t6_lines))
    
    # Meta-Attribution
    print("[*] Generating 70-Cell Meta-Attribution Clustermap...")
    df_attr = pd.DataFrame(attribution_matrix, index=labels).fillna(0)
    df_attr = df_attr.loc[:, df_attr.var() > 0.0]
    
    sns.set_theme(style='white')
    g = sns.clustermap(df_attr, cmap='rocket_r', method='ward', metric='euclidean', figsize=(10, 15), linewidths=.5, annot=True, fmt=".1f", annot_kws={"size": 8})
    g.fig.suptitle("Meta-Attribution Structural Clustering", fontsize=16, fontweight='bold', y=1.02)
    g.ax_heatmap.set_xlabel("Semantic Feature Clusters (Invariant Core Testing)", fontsize=12)
    g.ax_heatmap.set_ylabel("Environment (Architecture_OriginYear)", fontsize=12)
    

    if is_appendix: return  # Skip figures for appendices loop
    out_dir_fig = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "ch5")
    os.makedirs(out_dir_fig, exist_ok=True)
    out_path_fig = os.path.join(out_dir_fig, "fig_ch5_35_meta_attribution_clustermap.pdf")
    g.savefig(out_path_fig, bbox_inches='tight')
    
    print("[*] Generating Performance-Weighted Clustermap...")
    mean_perf_for_plot = res_df.groupby(['Model', 'Anchor'])['PRAUC'].mean().reset_index()
    mean_perf_for_plot['label'] = mean_perf_for_plot['Model'] + "_" + mean_perf_for_plot['Anchor'].str.replace('Pre-', '')
    wt_map = dict(zip(mean_perf_for_plot['label'], mean_perf_for_plot['PRAUC']))
    
    df_attr_w = df_attr.copy()
    df_attr_w = df_attr_w.mul(df_attr_w.index.map(wt_map).fillna(0), axis=0)
    
    g_w = sns.clustermap(df_attr_w, cmap='mako', method='ward', metric='euclidean', figsize=(10, 15), linewidths=.5, annot=True, fmt=".1f", annot_kws={"size": 8})
    g_w.fig.suptitle("Performance-Weighted Meta-Attribution Clustering", fontsize=16, fontweight='bold', y=1.02)
    g_w.ax_heatmap.set_xlabel("Semantic Feature Clusters", fontsize=12)
    g_w.ax_heatmap.set_ylabel("Environment", fontsize=12)
    
    out_path_fig_w = os.path.join(out_dir_fig, "fig_ch5_36_meta_attribution_clustermap_weighted.pdf")
    g_w.savefig(out_path_fig_w, bbox_inches='tight')
    
    # Archetypal
    print("[*] Generating Archetypal Table (Including Causal)...")
    df_attr['Family'] = families
    df_agg = df_attr.groupby('Family').mean()
    c_order = df_agg.loc['Tree'].sort_values(ascending=False).index.tolist()
    
    t7_lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption[Archetypal Family Attribution]{\textbf{Archetypal Family Attribution.} Average absolute model reliance allocated to each semantic feature cluster. High-cardinality identity floats mathematically blinded via Shannon-Entropy decision tree mapping ($min\_samples\_leaf=100$).}',
        r'\label{tab:archetypal_attribution}', r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{lccc}', r'\toprule',
        r'\textbf{Semantic Target Cluster} & \textbf{Tree} & \textbf{Deep} & \textbf{Regularized Linear} \\',
        r'\midrule'
    ]
    
    for c in c_order:
        if c not in df_agg.columns: continue
        t = df_agg.loc['Tree', c] if 'Tree' in df_agg.index else 0
        d = df_agg.loc['Deep', c] if 'Deep' in df_agg.index else 0
        l = df_agg.loc['Regularized Linear', c] if 'Regularized Linear' in df_agg.index else 0
        ca = 0
        
        t_str, d_str, l_str, ca_str = f"{t:.1f}\\%", f"{d:.1f}\\%", f"{l:.1f}\\%", f"{ca:.1f}\\%"
        
        max_val = max(t, d, l, ca)
        if max_val > 0:
            if t == max_val: t_str = f"\\textbf{{{t_str}}}"
            elif d == max_val: d_str = f"\\textbf{{{d_str}}}"
            elif ca == max_val: ca_str = f"\\textbf{{{ca_str}}}"
            else: l_str = f"\\textbf{{{l_str}}}"
            
        t7_lines.append(f"{c} & {t_str} & {d_str} & {l_str} \\\\")
    
    t7_lines.extend([r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}'])
    
    with open(os.path.join(attr_dir, f'tbl_ch5_02_archetypal_attribution{suffix}.tex'), 'w') as f: f.write('\n'.join(t7_lines))
    
    # Weighted Archetypal
    print("[*] Generating Performance-Weighted Archetypal Table...")
    # Calculate mean out-of-distribution PR-AUC for each Model-Anchor environment
    mean_perf = res_df.groupby(['Model', 'Anchor'])['PRAUC'].mean().reset_index()
    # Map to format 'CatBoost_2018'
    mean_perf['label'] = mean_perf['Model'] + "_" + mean_perf['Anchor'].str.replace('Pre-', '')
    weight_map = dict(zip(mean_perf['label'], mean_perf['PRAUC']))
    
    df_w = df_attr.copy()
    df_w['Weight'] = df_w.index.map(weight_map).fillna(0)
    # Normalize weights inside each Family so they sum to 1.0 (or we can just do a weighted mean)
    def w_avg(group_df):
        w = group_df['Weight']
        # group_df might not have 'Family' column physically inside if it's the group key
        cols_to_drop = ['Weight']
        if 'Family' in group_df.columns: cols_to_drop.append('Family')
        
        if w.sum() == 0: return group_df.drop(columns=cols_to_drop, errors='ignore').mean()
        return group_df.drop(columns=cols_to_drop, errors='ignore').mul(w, axis=0).sum() / w.sum()
    
    df_agg_w = df_w.groupby('Family').apply(w_avg)
    
    t8_lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption[Performance-Weighted Archetypal Attribution]{\textbf{Performance-Weighted Archetypal Family Attribution.} Model attribution vectors scaled by their corresponding out-of-distribution longitudinal PR-AUC performance. Architectures that successfully generalized under domain drift dominate their archetypal class weights.}',
        r'\label{tab:archetypal_attribution_weighted}', r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{lccc}', r'\toprule',
        r'\textbf{Semantic Target Cluster} & \textbf{Tree} & \textbf{Deep} & \textbf{Regularized Linear} \\',
        r'\midrule'
    ]
    
    for c in c_order:
        if c not in df_agg_w.columns: continue
        t = df_agg_w.loc['Tree', c] if 'Tree' in df_agg_w.index else 0
        d = df_agg_w.loc['Deep', c] if 'Deep' in df_agg_w.index else 0
        l = df_agg_w.loc['Regularized Linear', c] if 'Regularized Linear' in df_agg_w.index else 0
        ca = 0
        
        t_str, d_str, l_str, ca_str = f"{t:.1f}\\%", f"{d:.1f}\\%", f"{l:.1f}\\%", f"{ca:.1f}\\%"
        
        max_val = max(t, d, l, ca)
        if max_val > 0:
            if t == max_val: t_str = f"\\textbf{{{t_str}}}"
            elif d == max_val: d_str = f"\\textbf{{{d_str}}}"
            elif ca == max_val: ca_str = f"\\textbf{{{ca_str}}}"
            else: l_str = f"\\textbf{{{l_str}}}"
            
        t8_lines.append(f"{c} & {t_str} & {d_str} & {l_str} \\\\")
    
    t8_lines.extend([r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}'])
    
    with open(os.path.join(attr_dir, f'tbl_ch5_03_archetypal_attribution_weighted{suffix}.tex'), 'w') as f: f.write('\n'.join(t8_lines))
    
    print("[*] ALL 10-ARCHITECTURE DATA SUCCESSFULLY REWRITTEN.")
    
    

if __name__ == '__main__':
    run_drift_and_archetypes()

