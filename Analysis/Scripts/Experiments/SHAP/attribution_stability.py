"""
attribution_stability.py — Expanding-Window Attribution Stability Test
=======================================================================
Matches the exact rolling-origin anchors [2019, 2020, 2021, 2022, 2023]
and horizons [H0, H3] from StageC_opposition_risk.py (PART A).

For each anchor:
  1. Retrain CatBoost on year < anchor (clone of optimal_model spec)
  2. Extract TreeSHAP on year == anchor (held-out evaluation year)
  3. Compute clustered coarse-group attribution shares (Spearman |r|>0.7)
  4. Record per-group mean |SHAP| and share %

Outputs:
  - CSV: attribution_stability_{hz}.csv
  - Figure: fig_attribution_stability_{hz}.pdf (heatmap of group shares)
  - Figure: fig_attribution_rank_stability_{hz}.pdf (rank correlation)
  - Console: Spearman rank-order correlation across adjacent anchors

Author: Daniel Hardesty Lewis
Created: 2026-04-12
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings
import json
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

try:
    from catboost import CatBoostClassifier
except ImportError:
    raise ImportError("CatBoost required for this script")

try:
    import shap
except ImportError:
    raise ImportError("SHAP required for this script")

# ---- Path Setup ----
ROOT = r"C:\Users\dhl\data\thesis\thesis"
_scripts_dir = os.path.join(ROOT, 'Analysis', 'Scripts')
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from artifact_registry import ROOT_DIR, DATA_WAREHOUSE_DIR, TRACK1_DIR, TraceabilityRegistry as AR

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

# ---- Match StageC anchors exactly ----
ANCHORS = [2019, 2020, 2021, 2022, 2023]

# ---- Semantic cluster names (from plot_Track1_exhibits_real.py) ----
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

# ---- Feature name cleaner (from plot_Track1_exhibits_real.py) ----
def _rename_feature(name):
    LABELS = {
        'gross_site_area_acres': 'Site Area',
        'ldb_land_acres': 'Land Area',
        'deed_acreage': 'Deed Acreage',
        'ldb_yr_built': 'Year Built',
        'ldb_ilr': 'Improvement Ratio',
        'ldb_far': 'Floor Area Ratio',
        'ldb_units': 'Unit Count',
        'ldb_appraised_val': 'Appraised Value',
        'ldb_market_val': 'Market Value',
        'acs_median_household_income': 'Median Income',
        'acs_owner_occupied_units': 'Owner-Occupied Units',
        'acs_race_white': 'White Population',
        'protest': 'Historical Protest',
        'spatial_contagion_3yr': 'Nearby Protests (3yr)',
    }
    if name in LABELS:
        return LABELS[name]
    cleaned = name
    for prefix in ('acs_', 'ldb_', 'lui_', 'delta_'):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    return cleaned.replace('_', ' ').title()


def _cluster_features(X, features, threshold=0.30):
    """Hierarchical clustering by Spearman |r| > 0.7 (distance < 0.30)."""
    corr = X[features].corr(method='spearman').abs().clip(0, 1).fillna(0)
    corr_vals = corr.values.copy()
    np.fill_diagonal(corr_vals, 1.0)
    dist = np.clip((1.0 - corr_vals + (1.0 - corr_vals).T) / 2, 0, None)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, t=threshold, criterion='distance')
    return labels


def _get_cluster_names(features, labels, shap_matrix):
    """Assign semantic names to clusters, return {cluster_name: [feature_indices]}."""
    clusters = {}
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        feats = [features[i] for i in idx]
        cluster_shap = np.abs(shap_matrix[:, idx]).mean(axis=0)
        top_feat = feats[np.argmax(cluster_shap)]
        n = len(feats)

        if top_feat in SEMANTIC_CLUSTERS:
            name = SEMANTIC_CLUSTERS[top_feat]
        elif n == 1:
            name = _rename_feature(top_feat)
        else:
            name = f"{_rename_feature(top_feat)} Cluster"

        # If name already exists (because two distinct clusters share a semantic ID), 
        # MERGE them to give a single conceptual share to the user.
        if name in clusters:
            clusters[name].extend(idx.tolist())
        else:
            clusters[name] = idx.tolist()
    return clusters, labels


def run_stability_for_horizon(hz):
    """Run the expanding-window attribution stability test for one horizon."""
    data_file = os.path.join(DATA,
        "H0_Filing_Master_Enriched.csv" if hz == "H0" else "H3_Filing_Master_NLP.csv")

    if not os.path.exists(data_file):
        print(f"[!] Data file not found: {data_file}")
        return

    print(f"\n{'='*70}")
    print(f" ATTRIBUTION STABILITY TEST: {hz}")
    print(f"{'='*70}")

    # ---- Load data ----
    df = pd.read_csv(data_file, low_memory=False)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year']).sort_values('year').copy()
    df['is_protested'] = df['is_protested'].fillna(0).astype(int)

    dist_col = 'council_district' if 'council_district' in df.columns else 'council_district_x'
    if dist_col not in df.columns:
        df['council_district'] = 1
    else:
        df['council_district'] = df[dist_col].fillna(1)

    # IPW
    STAGE_A_PROBS = str(AR.stage_a_hazard(hz) if hasattr(AR, 'stage_a_hazard') else AR.TRACK0_METRICS / 'stage_a_hazard_results.csv')
    if os.path.exists(STAGE_A_PROBS):
        df_hazard = pd.read_csv(STAGE_A_PROBS, usecols=['standardized_tcad_id', 'year', 'Prob_Optimal_H=4'])
        if 'standardized_tcad_id' in df.columns:
            df['standardized_tcad_id'] = df['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
            df_hazard['standardized_tcad_id'] = df_hazard['standardized_tcad_id'].astype(str).str.zfill(10)
            df = df.merge(df_hazard, on=['standardized_tcad_id', 'year'], how='left')
            df['ipw'] = 1.0 / np.clip(df['Prob_Optimal_H=4'].fillna(0.01), 0.0001, 1.0)
        else:
            df['ipw'] = 1.0
    else:
        df['ipw'] = 1.0

    # Feature selection
    drop_cols = ['is_protested', 'case_number', 'organized_opposition',
                 'has_audio_record', 'TCAD ID', 'date', 'application_start_date',
                 'final_date', 'standardized_tcad_id', 'Prob_H=4', 'Prob_LGBM_H=4',
                 'Prob_CB_H=4', 'Prob_Optimal_H=4', 'ipw', dist_col, 'council_district']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    if hz == 'H0':
        leak_cols = [c for c in df_clean.columns if c.startswith('tfidf_') or c.startswith('speech_')]
        if leak_cols:
            df_clean = df_clean.drop(columns=leak_cols)

    X_all = df_clean.select_dtypes(include=[np.number])
    y_all = df['is_protested']
    weights_all = df['ipw']
    features = list(X_all.columns)

    # ---- 1. LEAKAGE-FREE REFERENCE STRUCTURE (Pre-2019) ----
    print(f"  [+] Computing Reference Structure (Strictly Pre-2019 Baseline)...")
    pre_2019_mask = df['year'] < 2019
    X_ref = X_all[pre_2019_mask]
    y_ref = y_all[pre_2019_mask]
    
    # Define reference conceptually from the first stable window
    ref_labels = _cluster_features(X_ref, features)
    
    # Use a baseline model on the reference set to get group names
    ref_cb = CatBoostClassifier(iterations=100, depth=4, verbose=0, random_seed=42).fit(X_ref, y_ref)
    ref_sv = shap.TreeExplainer(ref_cb).shap_values(X_ref)
    if isinstance(ref_sv, list): ref_sv = ref_sv[1] if len(ref_sv)>1 else ref_sv[0]
    
    global_cluster_map, _ = _get_cluster_names(features, ref_labels, ref_sv)
    print(f"  [+] Reference structure fixed with {len(global_cluster_map)} groups (Leakage-Free).")

    # ---- 2. REGIME STRUCTURE AUDIT (Post-2022) ----
    print(f"  [+] Auditing for Modern Regime Structure Shift (2022+)...")
    post_2022_mask = df['year'] >= 2022
    modern_labels = _cluster_features(X_all[post_2022_mask], features)
    # Check for rank correlation between ref_labels and modern_labels (as proxy for structural drift)
    struct_stability, _ = spearmanr(ref_labels, modern_labels)
    print(f"  [!] Global Correlation Structure Stability (Pre vs Post regime): {struct_stability:.3f}")

    # ---- Model spec ----
    base_model = CalibratedClassifierCV(
        estimator=CatBoostClassifier(
            iterations=150, depth=6, l2_leaf_reg=3,
            learning_rate=0.03, verbose=0, random_seed=42,
            auto_class_weights='Balanced'
        ),
        method='sigmoid', cv=5
    )

    # ---- Rolling-origin loop ----
    all_results = []
    anchor_group_shares = {}  # {anchor: {group_name: share%}}

    for anchor in ANCHORS:
        tr_mask = df['year'] < anchor
        te_mask = df['year'] == anchor

        n_train = tr_mask.sum()
        n_test = te_mask.sum()
        n_pos_test = y_all[te_mask].sum()

        if n_train < 20 or n_test < 5 or n_pos_test < 1:
            print(f"\n  Anchor {anchor}: SKIP (train={n_train}, test={n_test}, pos={n_pos_test})")
            continue

        print(f"\n  Anchor {anchor}: train year<{anchor} ({n_train}), test year=={anchor} ({n_test}, {n_pos_test} pos)")

        # Retrain
        cb = clone(base_model)
        cb.fit(X_all[tr_mask], y_all[tr_mask], sample_weight=weights_all[tr_mask])

        # Extract base CatBoost from CalibratedClassifierCV
        base_cb = cb.calibrated_classifiers_[0].estimator

        # TreeSHAP on test set
        X_test = X_all[te_mask]
        n_shap = min(1000, len(X_test))
        X_shap = X_test.sample(n=n_shap, random_state=42) if len(X_test) > n_shap else X_test

        explainer = shap.TreeExplainer(base_cb)
        sv = explainer.shap_values(X_shap)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        if hasattr(sv, 'values'):
            sv = sv.values

        # Compute group-level attribution shares using the FROZEN global_cluster_map
        total_abs_shap = np.abs(sv).sum()
        group_shares = {}
        for gname, gidx in global_cluster_map.items():
            group_abs = np.abs(sv[:, gidx]).sum()
            share = 100.0 * group_abs / total_abs_shap
            group_shares[gname] = share

        anchor_group_shares[anchor] = group_shares

        # Console output
        print(f"    {'Group':<35} {'Share':>7}")
        print(f"    {'-'*44}")
        for g, s in sorted(group_shares.items(), key=lambda x: -x[1]):
            if s > 1.0:
                print(f"    {g:<35} {s:>6.1f}%")

        for gname, share in group_shares.items():
            all_results.append({
                'Horizon': hz,
                'Anchor': anchor,
                'Group': gname,
                'Share_Pct': round(share, 2),
            })

    # ---- Save CSV ----
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(METRICS_DIR, f"attribution_stability_{hz}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n  [+] Saved {csv_path}")

    # ---- Build unified group set (top groups by max share across anchors) ----
    all_groups = set()
    for shares in anchor_group_shares.values():
        all_groups |= set(shares.keys())

    # Rank groups by mean share across anchors
    group_mean = {}
    for g in all_groups:
        vals = [anchor_group_shares[a].get(g, 0) for a in anchor_group_shares]
        group_mean[g] = np.mean(vals)

    top_groups = sorted(group_mean.keys(), key=lambda g: -group_mean[g])[:12]

    # ---- Heatmap Figure ----
    anchors_used = sorted(anchor_group_shares.keys())
    heatmap_data = np.zeros((len(top_groups), len(anchors_used)))
    for j, anchor in enumerate(anchors_used):
        for i, g in enumerate(top_groups):
            heatmap_data[i, j] = anchor_group_shares[anchor].get(g, 0)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(anchors_used)))
    ax.set_xticklabels([f'Train <{a}\nTest ={a}' for a in anchors_used], fontsize=9)
    ax.set_yticks(range(len(top_groups)))
    ax.set_yticklabels(top_groups, fontsize=9)

    # Annotate cells
    for i in range(len(top_groups)):
        for j in range(len(anchors_used)):
            val = heatmap_data[i, j]
            color = 'white' if val > heatmap_data.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    ax.set_title(f'Attribution Stability Across Expanding Windows ({hz})', fontsize=13, pad=12)
    ax.set_xlabel('Rolling-Origin Anchor', fontsize=11)
    fig.colorbar(im, ax=ax, label='Attribution Share (%)', shrink=0.8)
    plt.tight_layout()

    heatmap_path = os.path.join(FIG_DIR, f"fig_attribution_stability_{hz}.pdf")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"  [+] Saved {heatmap_path}")

    # ---- Regime Shift Summary ----
    print(f"\n{'='*70}")
    print(f" REGIME SHIFT SUMMARY ({hz})")
    print(f"{'='*70}")
    
    pre_anchors = [a for a in anchors_used if a < 2022]
    post_anchors = [a for a in anchors_used if a >= 2022]
    
    if pre_anchors and post_anchors:
        def get_avg_shares(anchors):
            totals = {}
            for a in anchors:
                for g, s in anchor_group_shares[a].items():
                    totals[g] = totals.get(g, 0) + s
            return {g: v / len(anchors) for g, v in totals.items()}
        
        pre_avg = get_avg_shares(pre_anchors)
        post_avg = get_avg_shares(post_anchors)
        
        print(f"    {'Group':<30} {'Pre-22 Avg%':>12} {'Post-22 Avg%':>12} {'Shift':>8}")
        print(f"    {'-'*65}")
        all_groups = sorted(set(pre_avg.keys()) | set(post_avg.keys()))
        for g in all_groups:
            s1 = pre_avg.get(g, 0)
            s2 = post_avg.get(g, 0)
            if s1 > 2.0 or s2 > 2.0:
                print(f"    {g:<30} {s1:>12.1f}% {s2:>12.1f}% {s2-s1:>7.1f}%")

    # ---- Rank Stability (Spearman rho between adjacent anchors) ----
    if len(anchors_used) >= 2:
        print(f"\n  RANK STABILITY (Spearman rho between adjacent anchors):")
        print(f"    {'Pair':<25} {'rho':>8} {'p-value':>10}")
        print(f"    {'-'*45}")

        rho_values = []
        for k in range(len(anchors_used) - 1):
            a1, a2 = anchors_used[k], anchors_used[k + 1]
            v1 = [anchor_group_shares[a1].get(g, 0) for g in top_groups]
            v2 = [anchor_group_shares[a2].get(g, 0) for g in top_groups]
            rho, pval = spearmanr(v1, v2)
            rho_values.append(rho)
            print(f"    {a1} -> {a2}{'':>15} {rho:>7.3f} {pval:>10.4f}")

        mean_rho = np.mean(rho_values)
        print(f"\n    Mean rho across adjacent anchors: {mean_rho:.3f}")

        # Also: full-window correlation (first vs last anchor)
        v_first = [anchor_group_shares[anchors_used[0]].get(g, 0) for g in top_groups]
        v_last = [anchor_group_shares[anchors_used[-1]].get(g, 0) for g in top_groups]
        rho_full, pval_full = spearmanr(v_first, v_last)
        print(f"    Full-span rho ({anchors_used[0]} vs {anchors_used[-1]}): {rho_full:.3f} (p={pval_full:.4f})")

        # ---- Rank stability line plot ----
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        pairs = [f'{anchors_used[k]}->{anchors_used[k+1]}' for k in range(len(anchors_used)-1)]
        ax2.plot(pairs, rho_values, 'o-', color='#1b4965', linewidth=2, markersize=8)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect stability')
        ax2.axhline(y=mean_rho, color='#e63946', linestyle=':', alpha=0.8,
                    label=f'Mean rho = {mean_rho:.3f}')
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel('Spearman rho (rank correlation)', fontsize=11)
        ax2.set_xlabel('Adjacent anchor pairs', fontsize=11)
        ax2.set_title(f'Attribution Rank Stability ({hz})', fontsize=13)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        rank_path = os.path.join(FIG_DIR, f"fig_attribution_rank_stability_{hz}.pdf")
        plt.savefig(rank_path)
        plt.close()
        print(f"  [+] Saved {rank_path}")


def main():
    for hz in ['H0', 'H3']:
        run_stability_for_horizon(hz)

    print("\n" + "="*70)
    print(" ATTRIBUTION STABILITY TEST COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
