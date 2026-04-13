"""
attribution_stability.py — Expanding-Window Attribution Stability Test
=======================================================================
Refactored to include:
1. Multi-model auditing (CatBoost & LightGBM)
2. Clustered vs Unclustered feature importance
3. Dynamic taxonomic drift tracking (Adjusted Rand Index for clusters over expanding windows)
4. SHAP Interaction Values extraction

Author: Daniel Hardesty Lewis
Created: 2026-04-13
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

try:
    from catboost import CatBoostClassifier
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier
except ImportError:
    pass

try:
    import shap
except ImportError:
    pass

# ---- Path Setup ----
ROOT = r"C:\Users\dhl\data\thesis\thesis"
_scripts_dir = os.path.join(ROOT, 'Analysis', 'Scripts')
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from artifact_registry import ROOT_DIR, DATA_WAREHOUSE_DIR, TraceabilityRegistry as AR

try:
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

ROOT = str(ROOT_DIR)
DATA = str(DATA_WAREHOUSE_DIR)
FIG_DIR = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Track1_Exhibits")
METRICS_DIR = str(AR.TRACK1_METRICS)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

ANCHORS = [2019, 2020, 2021, 2022, 2023]

SEMANTIC_CLUSTERS = {
    'acs_owner_occupied_units': 'Housing Tenure',
    'acs_renter_occupied_units': 'Housing Tenure',
    'acs_total_housing_units': 'Housing Tenure',
    'acs_race_white': 'Demographic Composition',
    'acs_race_hispanic': 'Demographic Composition',
    'acs_race_black': 'Demographic Composition',
    'acs_race_asian': 'Demographic Composition',
    'acs_median_gross_rent': 'Neighborhood Income & Rent',
    'acs_median_household_income': 'Neighborhood Income & Rent',
    'acs_poverty_count': 'Neighborhood Income & Rent',
    'acs_median_home_value': 'Neighborhood Income & Rent',
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
    'ldb_land_use': 'Land Use Classification',
    'lui_land_use': 'Land Use Classification',
    'lui_general_land_use': 'Land Use Classification',
    'protest': 'Historical Protest Activity',
    'spatial_contagion_3yr': 'Historical Protest Activity',
    'spatial_contagion_1yr': 'Historical Protest Activity',
    'ldb_far': 'Zoning Density',
    'ldb_units': 'Zoning Density',
    'ldb_imprv_sqft': 'Improvement Scale',
}

def _rename_feature(name):
    LABELS = {
        'gross_site_area_acres': 'Site Area', 'ldb_land_acres': 'Land Area',
        'deed_acreage': 'Deed Acreage', 'ldb_yr_built': 'Year Built',
        'ldb_ilr': 'Improvement Ratio', 'ldb_far': 'Floor Area Ratio',
        'ldb_units': 'Unit Count', 'ldb_appraised_val': 'Appraised Value',
        'ldb_market_val': 'Market Value', 'acs_median_household_income': 'Median Income',
        'acs_owner_occupied_units': 'Owner-Occupied Units', 'acs_race_white': 'White Population',
        'protest': 'Historical Protest', 'spatial_contagion_3yr': 'Nearby Protests (3yr)',
    }
    if name in LABELS: return LABELS[name]
    cleaned = name
    for prefix in ('acs_', 'ldb_', 'lui_', 'delta_'):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    return cleaned.replace('_', ' ').title()

def _cluster_features(X, features, threshold=0.30):
    corr = X[features].corr(method='spearman').abs().clip(0, 1).fillna(0)
    corr_vals = corr.values.copy()
    np.fill_diagonal(corr_vals, 1.0)
    dist = np.clip((1.0 - corr_vals + (1.0 - corr_vals).T) / 2, 0, None)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    return fcluster(Z, t=threshold, criterion='distance')

def _get_cluster_names(features, labels, shap_matrix):
    clusters = {}
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        feats = [features[i] for i in idx]
        cluster_shap = np.abs(shap_matrix[:, idx]).mean(axis=0)
        top_feat = feats[np.argmax(cluster_shap)]
        name = SEMANTIC_CLUSTERS.get(top_feat, _rename_feature(top_feat) if len(feats) == 1 else f"{_rename_feature(top_feat)} Cluster")
        if name in clusters: clusters[name].extend(idx.tolist())
        else: clusters[name] = idx.tolist()
    return clusters, labels

def run_stability_for_horizon(hz):
    data_file = os.path.join(DATA, "H0_Filing_Master_Enriched.csv" if hz == "H0" else "H3_Filing_Master_NLP.csv")
    if not os.path.exists(data_file):
        print(f"[!] Data file not found: {data_file}")
        return

    print(f"\n{'='*70}")
    print(f" ATTRIBUTION STABILITY TEST: {hz}")
    print(f"{'='*70}")

    df = pd.read_csv(data_file, low_memory=False)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year', 'is_protested']).sort_values('year').copy()
    df['is_protested'] = df['is_protested'].astype(int)

    dist_col = 'council_district' if 'council_district' in df.columns else 'council_district_x'
    df['council_district'] = df[dist_col].fillna(1) if dist_col in df.columns else 1

    STAGE_A_PROBS = str(AR.stage_a_hazard(hz) if hasattr(AR, 'stage_a_hazard') else AR.TRACK0_METRICS / 'stage_a_hazard_results.csv')
    if os.path.exists(STAGE_A_PROBS):
        df_hazard = pd.read_csv(STAGE_A_PROBS, usecols=['standardized_tcad_id', 'year', 'Prob_Optimal_H=4'])
        if 'standardized_tcad_id' in df.columns:
            df['standardized_tcad_id'] = df['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
            df_hazard['standardized_tcad_id'] = df_hazard['standardized_tcad_id'].astype(str).str.zfill(10)
            df = df.merge(df_hazard, on=['standardized_tcad_id', 'year'], how='left')
            df['ipw'] = 1.0 / np.clip(df['Prob_Optimal_H=4'].fillna(0.01), 0.0001, 1.0)
        else: df['ipw'] = 1.0
    else: df['ipw'] = 1.0

    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'has_audio_record', 'TCAD ID', 'date', 'application_start_date', 'final_date', 'standardized_tcad_id', 'Prob_H=4', 'Prob_LGBM_H=4', 'Prob_CB_H=4', 'Prob_Optimal_H=4', 'ipw', dist_col, 'council_district']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    if hz == 'H0':
        leak_cols = [c for c in df_clean.columns if c.startswith('tfidf_') or c.startswith('speech_')]
        if leak_cols: df_clean = df_clean.drop(columns=leak_cols)

    X_all = df_clean.select_dtypes(include=[np.number])
    y_all = df['is_protested']
    weights_all = df['ipw']
    features = list(X_all.columns)

    # 1. LEAKAGE-FREE REFERENCE STRUCTURE
    pre_2019_mask = df['year'] < 2019
    X_ref = X_all[pre_2019_mask]
    y_ref = y_all[pre_2019_mask]
    ref_labels = _cluster_features(X_ref, features)
    ref_cb = CatBoostClassifier(iterations=100, depth=4, verbose=0, random_seed=42).fit(X_ref, y_ref)
    ref_sv = shap.TreeExplainer(ref_cb).shap_values(X_ref)
    if isinstance(ref_sv, list): ref_sv = ref_sv[1] if len(ref_sv)>1 else ref_sv[0]
    global_cluster_map, _ = _get_cluster_names(features, ref_labels, ref_sv)

    MODELS = {
        'CatBoost': CalibratedClassifierCV(
            estimator=CatBoostClassifier(iterations=150, depth=6, l2_leaf_reg=3, learning_rate=0.03, verbose=0, random_seed=42, auto_class_weights='Balanced'),
            method='sigmoid', cv=5
        ),
        'LightGBM': CalibratedClassifierCV(
            estimator=LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1),
            method='sigmoid', cv=5
        )
    }

    all_results_clustered = []
    all_results_unclustered = []
    all_results_interactions = []

    for model_name, base_model in MODELS.items():
        print(f"\n[{model_name}] EVALUATING ACROSS EXPANDING WINDOWS...")
        
        anchor_group_shares = {}
        unclustered_shares = {}
        
        for anchor in ANCHORS:
            tr_mask = df['year'] < anchor
            te_mask = df['year'] == anchor
            n_train = tr_mask.sum()
            n_test = te_mask.sum()
            if n_train < 20 or n_test < 5 or y_all[te_mask].sum() < 1: continue

            # Dynamic Taxonomic Drift Tracking
            current_labels = _cluster_features(X_all[tr_mask], features)
            ari_score = adjusted_rand_score(ref_labels, current_labels)
            
            print(f"\n  Anchor {anchor} | N={n_train} | Taxonomy ARI vs Pre-2019: {ari_score:.3f}")

            model = clone(base_model)
            model.fit(X_all[tr_mask], y_all[tr_mask])

            # Extract base estimator for SHAP
            clf = model.calibrated_classifiers_[0].estimator
            X_shap = X_all[te_mask].sample(n=min(300, n_test), random_state=42)
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_shap)
            
            # Try parsing interaction values
            try:
                interact_sv = explainer.shap_interaction_values(X_shap)
            except Exception as e:
                interact_sv = None

            if isinstance(sv, list): sv = sv[1] if len(sv) > 1 else sv[0]
            if hasattr(sv, 'values'): sv = sv.values
            if interact_sv is not None and isinstance(interact_sv, list): 
                interact_sv = interact_sv[1] if len(interact_sv) > 1 else interact_sv[0]

            total_abs_shap = np.abs(sv).sum()
            
            # --- 1. Clustered Shares (Fixed Mapping) ---
            g_shares = {}
            for gname, gidx in global_cluster_map.items():
                g_shares[gname] = 100.0 * np.abs(sv[:, gidx]).sum() / total_abs_shap
            anchor_group_shares[anchor] = g_shares

            # --- 2. Unclustered Features ---
            f_shares = {}
            for i, fname in enumerate(features):
                f_shares[_rename_feature(fname)] = 100.0 * np.abs(sv[:, i]).sum() / total_abs_shap
            unclustered_shares[anchor] = f_shares

            # --- 3. Interactions ---
            if interact_sv is not None:
                total_interact_shap = np.abs(interact_sv).sum()
                # Aggregate symmetric off-diagonals
                ix_shares = []
                for i in range(len(features)):
                    for j in range(i+1, len(features)):
                        val = np.abs(interact_sv[:, i, j]).sum() + np.abs(interact_sv[:, j, i]).sum()
                        if val > 0:
                            p_name = f"{_rename_feature(features[i])} x {_rename_feature(features[j])}"
                            ix_shares.append((p_name, 100.0 * val / total_interact_shap))
                
                ix_shares.sort(key=lambda x: -x[1])
                for p_name, share in ix_shares[:10]:
                    all_results_interactions.append({'Model': model_name, 'Horizon': hz, 'Anchor': anchor, 'Interaction': p_name, 'Share_Pct': round(share, 2)})

            # Output logs
            print(f"    -- Top Unclustered: " + ", ".join([f"{k} {v:.1f}%" for k,v in sorted(f_shares.items(), key=lambda x: -x[1])[:3]]))
            print(f"    -- Top Clustered:   " + ", ".join([f"{k} {v:.1f}%" for k,v in sorted(g_shares.items(), key=lambda x: -x[1])[:3]]))

            for gname, share in g_shares.items():
                all_results_clustered.append({'Model': model_name, 'Horizon': hz, 'Anchor': anchor, 'Group': gname, 'Share_Pct': round(share, 2)})
            for fname, share in f_shares.items():
                all_results_unclustered.append({'Model': model_name, 'Horizon': hz, 'Anchor': anchor, 'Feature': fname, 'Share_Pct': round(share, 2)})

        # Save model-specific heatmap for clustered
        all_groups = set()
        for s in anchor_group_shares.values(): all_groups |= set(s.keys())
        g_mean = {g: np.mean([anchor_group_shares[a].get(g, 0) for a in anchor_group_shares]) for g in all_groups}
        top_groups = sorted(g_mean.keys(), key=lambda g: -g_mean[g])[:12]

        anchors_used = sorted(anchor_group_shares.keys())
        heatmap_data = np.zeros((len(top_groups), len(anchors_used)))
        for j, a in enumerate(anchors_used):
            for i, g in enumerate(top_groups):
                heatmap_data[i, j] = anchor_group_shares[a].get(g, 0)

        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(anchors_used)))
        ax.set_xticklabels([f'Train <{a}\nTest ={a}' for a in anchors_used], fontsize=9)
        ax.set_yticks(range(len(top_groups)))
        ax.set_yticklabels(top_groups, fontsize=9)
        for i in range(len(top_groups)):
            for j in range(len(anchors_used)):
                val = heatmap_data[i, j]
                c = 'white' if val > heatmap_data.max() * 0.6 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=8, color=c, fontweight='bold')
        ax.set_title(f'[{model_name}] Clustered Attribution Stability ({hz})', fontsize=13, pad=12)
        fig.colorbar(im, ax=ax, label='Share (%)', shrink=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"fig_attribution_stability_{hz}_{model_name}.pdf"))
        plt.close()

    pd.DataFrame(all_results_clustered).to_csv(os.path.join(METRICS_DIR, f"clustered_stability_{hz}.csv"), index=False)
    pd.DataFrame(all_results_unclustered).to_csv(os.path.join(METRICS_DIR, f"unclustered_stability_{hz}.csv"), index=False)
    if all_results_interactions:
        pd.DataFrame(all_results_interactions).to_csv(os.path.join(METRICS_DIR, f"interaction_stability_{hz}.csv"), index=False)
    
def main():
    for hz in ['H0', 'H3']:
        run_stability_for_horizon(hz)
    print("\n" + "="*70)
    print(" ATTRIBUTION STABILITY TEST COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
