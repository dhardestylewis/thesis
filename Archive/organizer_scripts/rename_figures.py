import os, shutil

FIG = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Figures'

# Rename map: (subdir, old_name) -> new_name
# Scheme: fig_ch<N>_<seq>_<slug>.png
# Ch4 = model performance / benchmarks / drift
# Ch5 = attribution, clustering, ablation (interpretability)

renames = [
    # ── performance ──────────────────────────────────────────────────────
    ('performance', 'plot_final_degradation_lineplot.png',         'fig_ch4_01_ood_degradation_lineplot.png'),
    ('performance', 'plot_final_offset_degradation.png',           'fig_ch4_02_temporal_offset_degradation.png'),
    ('performance', 'plot_final_ood_boxplot.png',                  'fig_ch4_03_ood_prauc_boxplot.png'),
    ('performance', 'plot_ltr_performance_drift.png',              'fig_ch4_04_ltr_performance_drift.png'),
    ('performance', 'plot_multiseed_performance_drift.png',        'fig_ch4_05_multiseed_performance_drift.png'),
    ('performance', 'plot_multiseed_unclustered_performance.png',  'fig_ch4_06_multiseed_unclustered_performance.png'),
    ('performance', 'plot_offset_decay.png',                       'fig_ch4_07_offset_prauc_decay.png'),
    ('performance', 'plot_anchor_stability.png',                   'fig_ch4_08_anchor_stability.png'),
    ('performance', 'plot_architectural_parity.png',               'fig_ch4_09_architectural_parity.png'),
    ('performance', 'plot_extreme_ltr_decay.png',                  'fig_ch4_10_extreme_ltr_decay.png'),
    ('performance', 'plot_extreme_ltr_poison.png',                 'fig_ch4_11_extreme_ltr_poison.png'),
    ('performance', 'plot_nontree_classifier_poison.png',          'fig_ch4_12_nontree_classifier_poison.png'),
    ('performance', 'plot_nontree_regression_rescue.png',          'fig_ch4_13_nontree_regression_rescue.png'),
    ('performance', 'plot_omnibus_classifier_poison.png',          'fig_ch4_14_omnibus_classifier_poison.png'),
    ('performance', 'plot_omnibus_ltr_poison.png',                 'fig_ch4_15_omnibus_ltr_poison.png'),
    ('performance', 'plot_omnibus_ltr_target_parity.png',          'fig_ch4_16_omnibus_ltr_target_parity.png'),
    ('performance', 'plot_omnibus_regression_rescue.png',          'fig_ch4_17_omnibus_regression_rescue.png'),

    # ── attribution ───────────────────────────────────────────────────────
    ('attribution', 'plot_ltr_attribution.png',                    'fig_ch5_01_ltr_feature_attribution.png'),
    ('attribution', 'plot_ltr_meta_attribution.png',               'fig_ch5_02_ltr_meta_attribution.png'),
    ('attribution', 'plot_ltr_weighted_meta_attribution.png',      'fig_ch5_03_ltr_weighted_meta_attribution.png'),
    ('attribution', 'plot_ltr_clustermap_meta_attribution.png',    'fig_ch5_04_ltr_clustermap_meta_attribution.png'),
    ('attribution', 'plot_semantic_cluster_attribution.png',       'fig_ch5_05_semantic_cluster_attribution.png'),
    ('attribution', 'plot_preclustered_semantic_omnimap.png',      'fig_ch5_06_preclustered_semantic_omnimap.png'),
    ('attribution', 'plot_preclustered_semantic_lineplots.png',    'fig_ch5_07_preclustered_semantic_lineplots.png'),
    ('attribution', 'plot_final_preclustered_shap.png',            'fig_ch5_08_preclustered_shap_bar.png'),
    ('attribution', 'plot_final_native_clustered_shap.png',        'fig_ch5_09_native_clustered_shap_bar.png'),
    ('attribution', 'plot_final_plain_english_shap.png',           'fig_ch5_10_plain_english_shap_bar.png'),
    ('attribution', 'plot_final_top15_unsupervised_shap.png',      'fig_ch5_11_top15_unsupervised_shap.png'),
    ('attribution', 'plot_final_unsupervised_shap.png',            'fig_ch5_12_unsupervised_shap_full.png'),
    ('attribution', 'plot_final_meta_attribution.png',             'fig_ch5_13_meta_attribution_heatmap.png'),
    ('attribution', 'plot_final_ood_domain_heatmap.png',           'fig_ch5_14_ood_domain_attribution_heatmap.png'),
    ('attribution', 'plot_final_unsupervised_temporal_heatmap.png','fig_ch5_15_unsupervised_temporal_heatmap.png'),

    # ── clustering ────────────────────────────────────────────────────────
    ('clustering',  'plot_final_preclustered_v2.png',              'fig_ch5_16_preclustered_env_matrix.png'),
    ('clustering',  'plot_final_postclustered_v2.png',             'fig_ch5_17_postclustered_env_matrix.png'),
    ('clustering',  'plot_recursive_clustermap.png',               'fig_ch5_18_recursive_clustermap.png'),
    ('clustering',  'plot_recursive_omni_clustermap.png',          'fig_ch5_19_recursive_omni_clustermap.png'),
    ('clustering',  'plot_recursive_semantic_omni_clustermap.png', 'fig_ch5_20_recursive_semantic_omni_clustermap.png'),
    ('clustering',  'plot_final_recursive_omnibus_clustermap.png', 'fig_ch5_21_omnibus_metaattrib_clustermap.png'),
    ('clustering',  'plot_thermodynamic_megastack.png',            'fig_ch5_22_thermodynamic_megastack.png'),

    # ── ablation ──────────────────────────────────────────────────────────
    ('ablation',    'plot_final_temporal_ablation.png',            'fig_ch5_23_temporal_ablation_curve.png'),
    ('ablation',    'plot_final_recursive_ablation_degradation.png','fig_ch5_24_recursive_domain_ablation.png'),
    ('ablation',    'plot_final_native_ablation_degradation.png',  'fig_ch5_25_native_domain_ablation.png'),
]

for subdir, old, new in renames:
    src = os.path.join(FIG, subdir, old)
    dst = os.path.join(FIG, subdir, new)
    if not os.path.exists(src):
        print('SKIP: ' + old)
        continue
    os.rename(src, dst)
    print(old + '  ->  ' + new)

print('\nDone.')
