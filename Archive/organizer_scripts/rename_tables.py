import os, re, shutil

DRAFT = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Draft_v1'
TBL   = os.path.join(DRAFT, 'Tables')

# Scheme: tbl_ch<N>_<seq>_<slug>.tex
renames = [
    # chapter1_descriptive
    ('chapter1_descriptive', 'Table_1_Summary_Stats.tex',    'tbl_ch1_01_summary_stats.tex'),
    ('chapter1_descriptive', 'Table_2_OLS_Results.tex',      'tbl_ch1_02_ols_results.tex'),
    ('chapter1_descriptive', 'lib_ast.tex',                  'tbl_ch1_03_lib_ast.tex'),
    ('chapter1_descriptive', 'summary_stats_table.tex',      'tbl_ch1_04_summary_stats_scratch.tex'),

    # chapter4_performance
    ('chapter4_performance', 'Table7_StageC_PR.tex',          'tbl_ch4_01_stagec_pr.tex'),
    ('chapter4_performance', 'alternative_architectures.tex', 'tbl_ch4_02_alternative_architectures.tex'),
    ('chapter4_performance', 'calibration_benchmark.tex',     'tbl_ch4_03_calibration_benchmark.tex'),
    ('chapter4_performance', 'calibration_primary_ood.tex',   'tbl_ch4_04_calibration_primary_ood.tex'),
    ('chapter4_performance', 'comprehensive_benchmark.tex',   'tbl_ch4_05_comprehensive_benchmark.tex'),
    ('chapter4_performance', 'disqualification_matrix.tex',   'tbl_ch4_06_disqualification_matrix.tex'),
    ('chapter4_performance', 'foundation_frontier.tex',       'tbl_ch4_07_foundation_frontier.tex'),
    ('chapter4_performance', 'metrics_config.tex',            'tbl_ch4_08_metrics_config.tex'),
    ('chapter4_performance', 'multi_horizon_results.tex',     'tbl_ch4_09_multi_horizon_results.tex'),
    ('chapter4_performance', 'performance_integrity_audit.tex','tbl_ch4_10_performance_integrity_audit.tex'),
    ('chapter4_performance', 'seed_summary.tex',              'tbl_ch4_11_seed_summary.tex'),
    ('chapter4_performance', 'stage_b.tex',                   'tbl_ch4_12_stage_b.tex'),
    ('chapter4_performance', 'stagea_limits.tex',             'tbl_ch4_13_stagea_limits.tex'),
    ('chapter4_performance', 'temporal_drift_analysis.tex',   'tbl_ch4_14_temporal_drift_analysis.tex'),
    ('chapter4_performance', 'temporal_drift_family.tex',     'tbl_ch4_15_temporal_drift_family.tex'),
    ('chapter4_performance', 'temporal_drift_lift.tex',       'tbl_ch4_16_temporal_drift_lift.tex'),
    ('chapter4_performance', 'temporal_drift_prauc_lift.tex', 'tbl_ch4_17_temporal_drift_prauc_lift.tex'),
    ('chapter4_performance', 'unclustered_stability.tex',     'tbl_ch4_18_unclustered_stability.tex'),

    # chapter5_attribution
    ('chapter5_attribution', 'Table12_Attrition_Timeline.tex',      'tbl_ch5_01_attrition_timeline.tex'),
    ('chapter5_attribution', 'archetypal_attribution.tex',           'tbl_ch5_02_archetypal_attribution.tex'),
    ('chapter5_attribution', 'archetypal_attribution_weighted.tex',  'tbl_ch5_03_archetypal_attribution_weighted.tex'),
    ('chapter5_attribution', 'semantic_feature_mapping.tex',         'tbl_ch5_04_semantic_feature_mapping.tex'),
    ('chapter5_attribution', 'spuriousness_index.tex',               'tbl_ch5_05_spuriousness_index.tex'),

    # chapter6_causal
    ('chapter6_causal', 'geographic_causal.tex',      'tbl_ch6_01_geographic_causal.tex'),
    ('chapter6_causal', 'ipw_balance_diagnostics.tex','tbl_ch6_02_ipw_balance_diagnostics.tex'),
    ('chapter6_causal', 'ipw_diagnostics_summary.tex','tbl_ch6_03_ipw_diagnostics_summary.tex'),

    # misc
    ('misc', 'fig_ipw_overlap_balance.tex', 'tbl_misc_01_ipw_overlap_balance.tex'),
    ('misc', 'test_table.tex',              'tbl_misc_02_test_scratch.tex'),
]

# Build old->new mapping for tex patching (old input path -> new input path)
path_map = {}
for subdir, old, new in renames:
    src = os.path.join(TBL, subdir, old)
    dst = os.path.join(TBL, subdir, new)
    if not os.path.exists(src):
        print('SKIP: ' + old)
        continue
    os.rename(src, dst)
    print(old + '  ->  ' + new)
    # Build both the old pre-patch path and post-patch (chapter subdir) path
    path_map[f'Tables/{old}'] = f'Tables/{subdir}/{new}'
    path_map[f'Tables/{subdir}/{old}'] = f'Tables/{subdir}/{new}'

print('\nDone renaming.\n')

# Patch tex files
tex_files = [os.path.join(DRAFT, 'Austin_NIMBY_Thesis_Draft.tex')]
sections_dir = os.path.join(DRAFT, 'Sections')
for f in sorted(os.listdir(sections_dir)):
    if f.endswith('.tex'):
        tex_files.append(os.path.join(sections_dir, f))

for tf in tex_files:
    with open(tf, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    new_text = text
    for old_path, new_path in path_map.items():
        new_text = new_text.replace(old_path, new_path)
    if new_text != text:
        with open(tf, 'w', encoding='utf-8') as f:
            f.write(new_text)
        print(f'Patched tex: {os.path.basename(tf)}')

print('\nAll done.')
