import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score
from scipy.cluster.hierarchy import fcluster
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

warnings = __import__('warnings')
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year', 'is_protested']).sort_values('year')

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date']
future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
X_raw_df = df.drop(columns=[c for c in (drop_cols + future_features) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)

# DISCRETIZATION BLOCK (ERADICATING IDENTITY HACKING)
phys_floats = [
    'ldb_appraised_val', 'land_market_value', 'total_market_value',
    'gross_site_area_acres', 'deed_acreage', 'ldb_land_acres', 'ldb_lotsize',
    'improvement_sq_ft', 'ldb_imprv_sqft'
]
# Only target those that actually exist in the matrix
to_discretize = [c for c in phys_floats if c in X_raw_df.columns]

if len(to_discretize) > 0:
    print(f"[*] Applying Anti-Memorization Discretization on {len(to_discretize)} variables...")
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile', subsample=200000)
    X_raw_df[to_discretize] = discretizer.fit_transform(X_raw_df[to_discretize])

features = X_raw_df.columns.tolist()
X_raw = X_raw_df.values
y = df['is_protested'].values
years = df['year'].values

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
    X_train_raw, y_train = X_raw[train_mask], y[train_mask]
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)

    models = {
        'CatBoost': CatBoostClassifier(iterations=100, depth=6, verbose=0, random_seed=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'Logistic (L2)': LogisticRegression(class_weight='balanced', random_state=42, max_iter=500),
        'TabNet': TabNetClassifier(verbose=0),
        'TabNet (VREx)': TabNetClassifier(optimizer_params={'weight_decay': 0.05}, verbose=0)
    }

    print(f"[*] Training Discretized models for Pre-{anchor} anchor...")
    fitted_models = {}
    
    for name, m in models.items():
        if name == 'Logistic (L2)': 
            m.fit(X_train_sc, y_train)
            raw_imp = np.abs(m.coef_[0])
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
        fam = 'Linear' if 'Logistic' in name else 'Deep' if 'TabNet' in name else 'Trees'
        families.append(fam)

    for test_year in eval_years:
        if test_year < anchor: continue
        test_mask = years == test_year
        if test_mask.sum() < 5 or y[test_mask].sum() < 1: continue
            
        X_test_raw, y_test = X_raw[test_mask], y[test_mask]
        X_test_sc = scaler.transform(X_test_raw)
        
        for name, m in fitted_models.items():
            if 'Logistic' in name or 'TabNet' in name:
                p = m.predict_proba(X_test_sc)[:, 1]
            else:
                p = m.predict_proba(X_test_raw)[:, 1]
            
            base_rate = y_test.sum() / len(y_test)
            prauc = average_precision_score(y_test, p)
            lift = prauc / base_rate if base_rate > 0 else 0
            
            fam = 'Linear' if 'Logistic' in name else 'Deep' if 'TabNet' in name else 'Trees'
            predictive_results.append({
                'Family': fam, 'Model': name, 'Anchor': f'Pre-{anchor}',
                'Evaluate_Year': test_year, 'PRAUC': prauc, 'Lift': lift
            })

print("[*] Generating Performance Tables...")
res_df = pd.DataFrame(predictive_results)

def format_grid(df, metric, use_lift=False):
    lines = []
    for model in sorted(df['Model'].unique()):
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

OUT_DIR = os.path.join(ROOT, 'Thesis_Draft', 'Draft_v1', 'Tables')

# Table 4
t4_lines = [r'\begin{table}[htbp]', r'\centering', r'\caption[Temporal Predictive Drift (PR-AUC decay)]{\textbf{Temporal predictive drift: PR-AUC decay by algorithm (Robust Discretized).}}', r'\label{tab:temporal_drift}', r'\begin{tabular}{l l' + 'c'*len(eval_years) + '}', r'\toprule', r'\textbf{Model} & \textbf{Anchor Training} & ' + ' & '.join(['\\textbf{'+str(y)+'}' for y in eval_years]) + r' \\', r'\midrule']
t4_lines.extend(format_grid(res_df, 'PRAUC'))
t4_lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
with open(os.path.join(OUT_DIR, 'temporal_drift_analysis.tex'), 'w') as f: f.write('\n'.join(t4_lines))

# Table 5
t5_lines = [r'\begin{table}[htbp]', r'\centering', r'\caption[Temporal Predictive Drift (PR-AUC lift)]{\textbf{Temporal predictive drift: PR-AUC lift by algorithm (Robust Discretized).}}', r'\label{tab:temporal_drift_prauc_lift}', r'\begin{tabular}{l l' + 'c'*len(eval_years) + '}', r'\toprule', r'\textbf{Model} & \textbf{Anchor Training} & ' + ' & '.join(['\\textbf{'+str(y)+'}' for y in eval_years]) + r' \\', r'\midrule']
t5_lines.extend(format_grid(res_df, 'Lift', use_lift=True))
t5_lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
with open(os.path.join(OUT_DIR, 'temporal_drift_prauc_lift.tex'), 'w') as f: f.write('\n'.join(t5_lines))

# Table 6
pivot_p = res_df.groupby(['Family', 'Anchor', 'Evaluate_Year'])['PRAUC'].max().reset_index()
pivot_mat_p = pivot_p.pivot_table(index='Family', columns=['Anchor', 'Evaluate_Year'], values='PRAUC')
winners_p = pivot_mat_p.idxmax()

pivot_l = res_df.groupby(['Family', 'Anchor', 'Evaluate_Year'])['Lift'].max().reset_index()
pivot_mat_l = pivot_l.pivot_table(index='Family', columns=['Anchor', 'Evaluate_Year'], values='Lift')
winners_l = pivot_mat_l.idxmax()

t6_lines = [
    r'\begin{table}[htbp]', r'\centering',
    r'\caption[Temporal Drift (Max-of-Family)]{\textbf{Max-of-Family dominance (Robust Discretized).}}',
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

with open(os.path.join(OUT_DIR, 'temporal_drift_family.tex'), 'w') as f: f.write('\n'.join(t6_lines))

# Meta-Attribution
print("[*] Generating Meta-Attribution Clustermap...")
df_attr = pd.DataFrame(attribution_matrix, index=labels).fillna(0)
df_attr = df_attr.loc[:, df_attr.var() > 0.0]

sns.set_theme(style='white')
g = sns.clustermap(df_attr, cmap='rocket_r', method='ward', metric='euclidean', figsize=(14, 20), linewidths=.5, annot=True, fmt=".1f")
g.fig.suptitle("Meta-Attribution Structural Clustering (Discretized Constraints)", fontsize=16, fontweight='bold', y=1.02)
g.ax_heatmap.set_xlabel("Semantic Feature Clusters (Invariant Core Testing)", fontsize=12)
g.ax_heatmap.set_ylabel("Environment (Architecture_OriginYear)", fontsize=12)

out_dir_fig = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "SHAP_MetaClustering")
os.makedirs(out_dir_fig, exist_ok=True)
out_path_fig = os.path.join(out_dir_fig, "meta_attribution_clustermap.pdf")
g.savefig(out_path_fig, bbox_inches='tight')

# Archetypal
print("[*] Generating Archetypal Table...")
df_attr['Family'] = families
df_agg = df_attr.groupby('Family').mean()
c_order = df_agg.loc['Trees'].sort_values(ascending=False).index.tolist()

t7_lines = [
    r'\begin{table}[htbp]', r'\centering',
    r'\caption[Archetypal Family Attribution]{\textbf{Archetypal Family Attribution (Discretized Constraint).} Average absolute model reliance allocated to each semantic feature cluster. High-cardinality identity floats mathematically blinded via Decile quantile mapping to explicitly eradicate identity hacking.}',
    r'\label{tab:archetypal_attribution}',
    r'\begin{tabular}{lccc}', r'\toprule',
    r'\textbf{Semantic Target Cluster} & \textbf{Tree Ensembles} & \textbf{Deep Architectures} & \textbf{Linear Architectures} \\',
    r'\midrule'
]

for c in c_order:
    t, d, l = df_agg.loc['Trees', c], df_agg.loc['Deep', c], df_agg.loc['Linear', c]
    t_str, d_str, l_str = f"{t:.1f}\\%", f"{d:.1f}\\%", f"{l:.1f}\\%"
    
    max_val = max(t, d, l)
    if t == max_val: t_str = f"\\textbf{{{t_str}}}"
    elif d == max_val: d_str = f"\\textbf{{{d_str}}}"
    else: l_str = f"\\textbf{{{l_str}}}"
        
    t7_lines.append(f"{c} & {t_str} & {d_str} & {l_str} \\\\")

t7_lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])

with open(os.path.join(OUT_DIR, "archetypal_attribution.tex"), 'w') as f: f.write('\n'.join(t7_lines))
print("[*] ALL DISCRETIZED DATA SUCCESSFULLY REWRITTEN.")

