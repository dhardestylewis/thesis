import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

import sys
try:
    # Attempt to locate the root Scripts directory
    _curr = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(_curr) != 'Scripts' and os.path.dirname(_curr) != _curr:
        _curr = os.path.dirname(_curr)
    if _curr not in sys.path:
        sys.path.insert(0, _curr)
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

from sklearn.calibration import calibration_curve
import os

import sys
_scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from artifact_registry import ROOT_DIR, TRACK1_DIR, TraceabilityRegistry as AR

ROOT = str(ROOT_DIR)
STAGE_C_OUT = str(TRACK1_DIR)
FIG_DIR = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Track1_Exhibits")
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Plain-English Feature Name Dictionary ──────────────────────────
FEATURE_LABELS = {
    # Site geometry
    'gross_site_area_acres':      'Site Area (acres)',
    'ldb_land_acres':             'Parcel Land Area (acres)',
    'ldb_lotsize':                'Lot Size',
    'lui_shape_area':             'Land Use Polygon Area',
    'land_acres':                 'Total Land Acreage',
    'deed_acreage':               'Deed Acreage',
    'ldb_constrained_area':       'Constrained Area',
    # Property characteristics
    'ldb_yr_built':               'Year Structure Built',
    'year_built':                 'Construction Year',
    'ldb_ilr':                    'Improvement-to-Land Ratio',
    'ldb_far':                    'Floor Area Ratio',
    'ldb_imprv_sqft':             'Improvement Square Footage',
    'ldb_units':                  'Existing Unit Count',
    # Valuation
    'ldb_appraised_val':          'Appraised Value',
    'ldb_market_val':             'Market Value',
    'land_market_value':          'Land Market Value',
    'improvement_market_value':   'Improvement Market Value',
    'appraised_value':            'Total Appraised Value',
    'assessed_value':             'Assessed Value',
    'net_taxable_value':          'Net Taxable Value',
    'prior_year_taxable_value':   'Prior Year Taxable Value',
    'new_construction_value':     'New Construction Value',
    'productivity_value':         'Agricultural Productivity Value',
    'exemption_amount_ex366':     'Tax Exemption Amount',
    # Land use classification
    'ldb_land_use':               'Land Use Code',
    'lui_land_use':               'Land Use Classification',
    'lui_matched':                'Land Use Match Flag',
    'lui_general_land_use':       'General Land Use Category',
    'lui_general_land_use_tv':    'Land Use Taxable Value',
    'lui_land_use_tv':            'Land Use Classification Value',
    'ldb_gen_land_use':           'General Land Use',
    # Demographics
    'acs_median_gross_rent':      'Median Gross Rent',
    'acs_race_black':             'Black Population Share',
    'acs_race_white':             'White Population Share',
    'acs_race_hispanic':          'Hispanic Population Share',
    'acs_race_asian':             'Asian Population Share',
    'acs_owner_occupied_units':   'Owner-Occupied Housing Units',
    'acs_renter_occupied_units':  'Renter-Occupied Housing Units',
    'acs_median_age':             'Median Age',
    'acs_median_home_value':      'Median Home Value',
    'acs_median_household_income':'Median Household Income',
    'acs_total_population':       'Total Population',
    'acs_total_housing_units':    'Total Housing Units',
    'acs_poverty_count':          'Population in Poverty',
    'acs_vintage':                'Census Data Vintage',
    # Spatial and political
    'latitude':                   'Latitude',
    'longitude':                  'Longitude',
    'ldb_council_district':       'Council District',
    'council_district_y':         'Council District',
    'nearby_GEOID':               'Nearby Census Tract',
    'zoning_case_GEOID':          'Zoning Case Census Tract',
    # Opposition history
    'protest':                    'Historical Protest Indicator',
    'spatial_contagion_3yr':      'Nearby Opposition, Trailing 3 Years',
    'spatial_contagion_1yr':      'Nearby Opposition, Trailing 1 Year',
    # Zoning change deltas
    'delta_max_height_ft':        'Proposed Height Increase (ft)',
    'delta_min_lot_sqft':         'Minimum Lot Size Change (sqft)',
    'delta_max_bldg_cov_pct':     'Building Coverage Change (%)',
    'delta_max_far':              'Floor Area Ratio Change',
    # Temporal
    'year':                       'Filing Year',
    'tax_year':                   'Tax Year',
    'ldb_source_year':            'Data Source Year',
    'lui_source_year':            'Land Use Data Year',
    # Administrative
    'ears_matched':               'Appraisal Record Matched',
    'situs_street_suffix':        'Street Suffix',
    'situs_city_state_zip':       'City/State/ZIP',
    'zoning_code':                'Zoning Code',
    'county_id':                  'County',
    'owner_state':                'Owner State',
    'school_district_flag':       'School District Flag',
    'account_number_formatted':   'Tax Account Number',
    'most_recent_sale_date':      'Most Recent Sale Date',
    'second_most_recent_sale_date':'Second Most Recent Sale Date',
    'record_sequence_number':     'Record Sequence',
    'supplemental_record_count':  'Supplemental Records',
    'appraisal_district_id':      'Appraisal District',
    'appraisal_district_id_2':    'Appraisal District (alt)',
    'owner_address_line1':        'Owner Address',
    'owner_address_line2':        'Owner Address Line 2',
    'owner_name':                 'Owner Name',
    'owner_city':                 'Owner City',
    'taxing_unit_id':             'Taxing Unit',
    'partial_exemption_flag':     'Partial Exemption Flag',
    'freeze_flag':                'Freeze Flag',
    'total_market_value':         'Total Market Value',
    'total_exemption_amount':     'Total Exemption Amount',
}

def _rename_feature(name):
    """Translate an internal column name to plain English."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    # Fallback: strip prefixes and clean up
    cleaned = name
    for prefix in ('acs_', 'ldb_', 'lui_', 'delta_', 'exemption_amount_', 'exemption_flag_'):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    return cleaned.replace('_', ' ').title()


def plot_all_track1_exhibits():
    with open(os.path.join(ROOT, "Analysis", "Scripts", "exhibit_titles.json"), "r") as f:
        titles = json.load(f)
        
    for hz in ['H0', 'H3']:
        print("==============================================")
        print(f" Rendering Authentic Track 1 PDF Exhibits: {hz}")
        print("==============================================")

        hz_name = "Filing Date Baseline" if hz == "H0" else "Pre-Council Horizon"
        
        # 1. Reliability Diagram (Calibration & ECE)
        preds_file = str(AR.stage_c_oof(hz))
        if os.path.exists(preds_file):
            df_oof = pd.read_csv(preds_file)
            
            plt.figure(figsize=(7, 6))
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            
            # Extract ECE metrics for labels if possible, but for simplicity we'll just plot curves
            prob_true_lr, prob_pred_lr = calibration_curve(df_oof['y_true'], df_oof['y_prob_lr'], n_bins=10)
            prob_true_rf, prob_pred_rf = calibration_curve(df_oof['y_true'], df_oof['y_prob_rf'], n_bins=10)
            prob_true_sp, prob_pred_sp = calibration_curve(df_oof['y_true'], df_oof['y_prob_spatial_lr'], n_bins=10)
            prob_true_anc, prob_pred_anc = calibration_curve(df_oof['y_true'], df_oof['y_prob_anchor'], n_bins=10)
            prob_true_cb, prob_pred_cb = calibration_curve(df_oof['y_true'], df_oof['y_prob'], n_bins=10)
            
            plt.plot(prob_pred_lr, prob_true_lr, 'v:', color='coral', label='Standard Logistic (ERM)')
            plt.plot(prob_pred_rf, prob_true_rf, '^:', color='gray', label='RandomForest (ERM)')
            plt.plot(prob_pred_sp, prob_true_sp, 'D--', color='purple', label='Spatial-FE Logistic (Domain)')
            plt.plot(prob_pred_anc, prob_true_anc, 'x-.', color='teal', label='Anchor Regression (Causal)')
            plt.plot(prob_pred_cb, prob_true_cb, 's-', color='darkred', label=f'CatBoost Primary (V-REx)', linewidth=2.5)
            
            plt.title(titles["stage_c_reliability"].format(hz=hz_name), fontsize=14)
            plt.xlabel('Mean Predicted Probability', fontsize=12)
            plt.ylabel('Fraction of Positives', fontsize=12)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"fig_calibration_ece_{hz}.pdf"))
            print(f"  [+] Saved fig_calibration_ece_{hz}.pdf")
            
        # 2. Temporal Drift
        drift_file = str(AR.stage_c_drift(hz))
        if os.path.exists(drift_file):
            plt.figure(figsize=(7, 5))
            try:
                df_drift = pd.read_csv(drift_file)
                if not df_drift.empty:
                    for anchor in df_drift['Anchor'].unique():
                        sub = df_drift[df_drift['Anchor'] == anchor]
                        plt.plot(sub['Offset'], sub['PR-AUC'], marker='o', label=f'Anchor < {anchor}')
                    plt.title(titles["stage_c_drift"].format(hz=hz_name), fontsize=14)
                    plt.xlabel('Years Out-of-Distribution (T + offset)', fontsize=12)
                    plt.ylabel('PR-AUC', fontsize=12)
                    plt.xticks([0, 1, 2, 3])
                    plt.legend()
                    plt.grid(True, alpha=0.3)
            except:
                pass
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"fig_temporal_drift_{hz}.pdf"))
            print(f"  [+] Saved fig_temporal_drift_{hz}.pdf")
            
        # 3. Policy Regimes
        regimes_file = str(AR.stage_c_regimes(hz))
        if os.path.exists(regimes_file):
            plt.figure(figsize=(8, 5))
            try:
                df_reg = pd.read_csv(regimes_file)
                if not df_reg.empty:
                    plt.bar(df_reg['Regime'], df_reg['PR-AUC'], color=['navy', 'orange', 'darkred'])
                    plt.title(titles["stage_c_policy_regimes"].format(hz=hz_name), fontsize=14)
                    plt.ylabel('PR-AUC', fontsize=12)
                    plt.ylim(0, max(0.5, df_reg['PR-AUC'].max() * 1.2))
                    plt.grid(axis='y', alpha=0.3)
            except:
                pass
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"fig_policy_regimes_{hz}.pdf"))
            print(f"  [+] Saved fig_policy_regimes_{hz}.pdf")

        # 4b. Clustered Feature Importance (Collinearity-Corrected)
        _plot_clustered_importance(hz, hz_name, titles)

        # 4c. SHAP Beeswarm
        _plot_shap_beeswarm(hz, hz_name, titles)


# ─── Helper: Clustered Feature Importance ───────────────────────────
def _plot_clustered_importance(hz, hz_name, titles):
    """
    Group features by pairwise Spearman correlation using hierarchical
    clustering (|r| > 0.7). Sum native importances within each cluster
    and label by the cluster's top-ranked member in plain English.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    fi_file = str(AR.stage_c_feature_importance(hz))
    data_file = os.path.join(
        str(ROOT), "Data", "Warehouse_As_Of",
        "H0_Filing_Master_Enriched.csv" if hz == "H0" else "H3_Filing_Master_NLP.csv"
    )
    if not os.path.exists(fi_file) or not os.path.exists(data_file):
        print(f"  [!] Skipping clustered importance ({hz}): missing files")
        return

    try:
        df_fi = pd.read_csv(fi_file)
        df_data = pd.read_csv(data_file, low_memory=False)
        X_num = df_data.select_dtypes(include=[np.number])

        common = [f for f in df_fi['Feature'] if f in X_num.columns]
        if len(common) < 5:
            print(f"  [!] Too few common features for clustering ({len(common)})")
            return

        corr = X_num[common].corr(method='spearman').abs()
        corr = corr.clip(0, 1).fillna(0)
        corr_vals = corr.values.copy()
        np.fill_diagonal(corr_vals, 1.0)
        dist = 1.0 - corr_vals
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0.0)
        dist = np.clip(dist, 0, None)

        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method='average')
        labels = fcluster(Z, t=0.30, criterion='distance')

        fi_map = dict(zip(df_fi['Feature'], df_fi['Importance']))
        cluster_imp = {}
        cluster_members = {}
        for feat, cid in zip(common, labels):
            imp = fi_map.get(feat, 0)
            cluster_imp[cid] = cluster_imp.get(cid, 0) + imp
            cluster_members.setdefault(cid, []).append((feat, imp))

        SEMANTIC_CLUSTERS = {
            'acs_owner_occupied_units': 'Housing Tenure',
            'acs_race_white': 'Demographic Composition',
            'acs_race_hispanic': 'Demographic Composition',
            'acs_race_black': 'Demographic Composition',
            'acs_median_gross_rent': 'Neighborhood Income & Rent',
            'acs_median_household_income': 'Neighborhood Income & Rent',
            'ldb_appraised_val': 'Property Valuation Metrics',
            'ldb_market_val': 'Property Valuation Metrics',
            'ldb_yr_built': 'Structure Age / Vintage',
            'year': 'Filing Timeline',
            'ldb_land_acres': 'Parcel Land Area',
            'gross_site_area_acres': 'Parcel Land Area',
            'ldb_land_use': 'Land Use Classification',
            'lui_land_use': 'Land Use Classification',
            'protest': 'Historical Protest Activity',
        }

        rows = []
        for cid, total_imp in cluster_imp.items():
            members = sorted(cluster_members[cid], key=lambda x: x[1], reverse=True)
            top_feature = members[0][0]
            n = len(members)
            
            if n == 1:
                label = _rename_feature(top_feature)
            elif top_feature in SEMANTIC_CLUSTERS:
                label = f"{SEMANTIC_CLUSTERS[top_feature]} ({n} features)"
            else:
                top_name = _rename_feature(top_feature)
                label = f"{top_name} Cluster ({n} features)"
                
            rows.append({'Cluster': label, 'Importance': total_imp, 'N': n, 'Members': members})

        df_cl = pd.DataFrame(rows).sort_values('Importance', ascending=False).head(10)
        df_cl = df_cl.sort_values('Importance', ascending=True)

        plt.figure(figsize=(11, 8))
        stacked_colors = ['#e63946', '#f4a261', '#e9c46a', '#e76f51', '#2a9d8f', '#264653']
        
        for idx, row in df_cl.reset_index(drop=True).iterrows():
            cluster_name = row['Cluster']
            members = row['Members']
            
            if row['N'] == 1:
                # Stand-alone predictor (Blue)
                plt.barh(cluster_name, row['Importance'], color='#1b4965', alpha=0.85)
            else:
                # Multi-feature cluster: stacked bar
                left_val = 0
                for c_idx, (feat_name, feat_imp) in enumerate(members):
                    color = stacked_colors[c_idx % len(stacked_colors)]
                    plt.barh(cluster_name, feat_imp, left=left_val, color=color, alpha=0.9, edgecolor='white', linewidth=0.5)
                    
                    # Optional internal annotation for very large sub-features
                    if feat_imp > 2.0:
                        short_name = _rename_feature(feat_name).replace("Share", "").replace("Metrics", "").strip()
                        # Shorten for tight clusters
                        if len(short_name) > 18: short_name = short_name[:15] + "..."
                        plt.text(left_val + (feat_imp/2), idx, short_name, ha='center', va='center', color='white', fontsize=7, fontweight='bold', clip_on=True)
                        
                    left_val += feat_imp
                    
        plt.title(titles.get("stage_c_feature_importance_clustered",
                  "Top 10 Predictor Groups After Correlation Clustering ({hz})").format(hz=hz_name),
                  fontsize=14)
        plt.xlabel('Summed Relative Importance (%)', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        out = os.path.join(FIG_DIR, f"fig_feature_importance_clustered_{hz}.pdf")
        plt.savefig(out)
        plt.close()
        print(f"  [+] Saved fig_feature_importance_clustered_{hz}.pdf")

    except Exception as e:
        print(f"  [!] Clustered importance failed: {e}")
        import traceback; traceback.print_exc()


# ─── Helper: SHAP Beeswarm ─────────────────────────────────────────
def _plot_shap_beeswarm(hz, hz_name, titles):
    """
    Load the saved CalibratedClassifierCV model, extract the base
    CatBoost estimator, and compute TreeSHAP values on a subsample.
    Feature columns are renamed to plain English before plotting.
    """
    import joblib

    model_file = str(AR.stage_c_model(hz))
    data_file = os.path.join(
        str(ROOT), "Data", "Warehouse_As_Of",
        "H0_Filing_Master_Enriched.csv" if hz == "H0" else "H3_Filing_Master_NLP.csv"
    )
    if not os.path.exists(model_file) or not os.path.exists(data_file):
        print(f"  [!] Skipping SHAP ({hz}): missing model or data file")
        return

    try:
        import shap

        cal_model = joblib.load(model_file)

        if hasattr(cal_model, 'calibrated_classifiers_'):
            base_cb = cal_model.calibrated_classifiers_[0].estimator
        elif hasattr(cal_model, 'estimator'):
            base_cb = cal_model.estimator
        else:
            print(f"  [!] Cannot extract base CatBoost from {type(cal_model)}")
            return

        df_data = pd.read_csv(data_file, low_memory=False)
        drop_cols = ['is_protested', 'case_number', 'organized_opposition',
                     'has_audio_record', 'TCAD ID', 'date',
                     'application_start_date', 'final_date',
                     'standardized_tcad_id', 'Prob_H=4', 'Prob_LGBM_H=4',
                     'Prob_CB_H=4', 'Prob_Optimal_H=4', 'ipw',
                     'council_district', 'council_district_x']
        df_clean = df_data.drop(columns=[c for c in drop_cols if c in df_data.columns])
        leak_cols = [c for c in df_clean.columns if c.startswith('tfidf_') or c.startswith('speech_')]
        if hz == 'H0' and leak_cols:
            df_clean = df_clean.drop(columns=leak_cols)
        X = df_clean.select_dtypes(include=[np.number])

        # Subsample for speed
        n_sample = min(1000, len(X))
        X_sample = X.sample(n=n_sample, random_state=42)

        explainer = shap.TreeExplainer(base_cb)
        shap_values = explainer.shap_values(X_sample)

        # Repaired collinearity: hierarchical cluster SHAP
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        corr = X.corr(method='spearman').abs().clip(0, 1).fillna(0)
        corr_vals = corr.values.copy()
        np.fill_diagonal(corr_vals, 1.0)
        dist = np.clip((1.0 - corr_vals + (1.0 - corr_vals).T)/2, 0, None)
        np.fill_diagonal(dist, 0.0)
        Z = linkage(squareform(dist, checks=False), method='average')
        labels = fcluster(Z, t=0.30, criterion='distance')
        
        SEMANTIC_CLUSTERS = {
            'acs_owner_occupied_units': 'Housing Tenure',
            'acs_race_white': 'Demographic Composition',
            'acs_race_hispanic': 'Demographic Composition',
            'acs_race_black': 'Demographic Composition',
            'acs_median_gross_rent': 'Neighborhood Income & Rent',
            'acs_median_household_income': 'Neighborhood Income & Rent',
            'ldb_appraised_val': 'Property Valuation Metrics',
            'ldb_market_val': 'Property Valuation Metrics',
            'ldb_yr_built': 'Structure Age / Vintage',
            'year': 'Filing Timeline',
            'ldb_land_acres': 'Parcel Land Area',
            'gross_site_area_acres': 'Parcel Land Area',
            'ldb_land_use': 'Land Use Classification',
            'lui_land_use': 'Land Use Classification',
            'protest': 'Historical Protest Activity',
        }
        features = list(X.columns)
        new_cols = {}
        new_shap = []
        for cid in np.unique(labels):
            idx = np.where(labels == cid)[0]
            cluster_feats = [features[i] for i in idx]
            # Use the feature with highest max absolute SHAP as top_feature for naming
            cluster_shap_sums = np.abs(shap_values[:, idx]).max(axis=0)
            top_feature = cluster_feats[np.argmax(cluster_shap_sums)]
            n = len(cluster_feats)
            
            if n == 1:
                label = _rename_feature(top_feature)
            elif top_feature in SEMANTIC_CLUSTERS:
                label = f"{SEMANTIC_CLUSTERS[top_feature]} ({n} features)"
            else:
                label = f"{_rename_feature(top_feature)} Cluster ({n} features)"
                
            new_cols[label] = X_sample[cluster_feats].mean(axis=1) # average feature magnitude for coloring
            new_shap.append(shap_values[:, idx].sum(axis=1))       # mathematically aggregate collinear SHAP attributions

        X_display = pd.DataFrame(new_cols, index=X_sample.index)
        shap_matrix = np.column_stack(new_shap)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_matrix, X_display, max_display=15, show=False, plot_size=None)
        plt.title(titles.get("stage_c_shap_beeswarm",
                  "SHAP Feature Attribution ({hz})").format(hz=hz_name),
                  fontsize=14)
        plt.tight_layout()
        out = os.path.join(FIG_DIR, f"fig_shap_beeswarm_{hz}.pdf")
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        print(f"  [+] Saved fig_shap_beeswarm_{hz}.pdf")

    except Exception as e:
        print(f"  [!] SHAP beeswarm failed: {e}")
        import traceback; traceback.print_exc()


if __name__ == '__main__':
    plot_all_track1_exhibits()

