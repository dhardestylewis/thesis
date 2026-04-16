import os, shutil

FIG_ROOT = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Draft_v1\Figures'
TBL_ROOT = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Draft_v1\Tables'

def mv(base, src, dst):
    s = os.path.join(base, src)
    d = os.path.join(base, dst)
    if not os.path.exists(s):
        print('SKIP: ' + src)
        return
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.move(s, d)
    print(src + ' -> ' + dst)

# ── Figures ──────────────────────────────────────────────────────────────────
# Two loose PNGs look like Chapter 5/6 exhibits → Chapter5 subdir
mv(FIG_ROOT, 'Fig9_Model_Comparison_PR_AUC.png',  'Chapter5/Fig9_Model_Comparison_PR_AUC.png')
mv(FIG_ROOT, 'Fig10_Hyperparameter_Sweeps.png',   'Chapter5/Fig10_Hyperparameter_Sweeps.png')

# Four loose PDFs are generic topic figures
os.makedirs(os.path.join(FIG_ROOT, 'Core'), exist_ok=True)
for f in ['fig_calibration_ece.pdf', 'fig_feature_importance.pdf',
          'fig_policy_regimes.pdf',  'fig_temporal_drift.pdf']:
    mv(FIG_ROOT, f, 'Core/' + f)

# Audit files → admin subfolder
os.makedirs(os.path.join(FIG_ROOT, 'audit'), exist_ok=True)
mv(FIG_ROOT, 'figures_audit.txt',          'audit/figures_audit.txt')
mv(FIG_ROOT, 'figures_audit_checklist.md', 'audit/figures_audit_checklist.md')

# ── Tables ───────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(TBL_ROOT, 'chapter1_descriptive'),  exist_ok=True)
os.makedirs(os.path.join(TBL_ROOT, 'chapter4_performance'),  exist_ok=True)
os.makedirs(os.path.join(TBL_ROOT, 'chapter5_attribution'),  exist_ok=True)
os.makedirs(os.path.join(TBL_ROOT, 'chapter6_causal'),       exist_ok=True)
os.makedirs(os.path.join(TBL_ROOT, 'misc'),                  exist_ok=True)

ch1 = ['Table_1_Summary_Stats.tex', 'Table_2_OLS_Results.tex', 'lib_ast.tex',
        'summary_stats_table.tex']

ch4 = ['alternative_architectures.tex', 'calibration_benchmark.tex',
       'calibration_primary_ood.tex', 'comprehensive_benchmark.tex',
       'disqualification_matrix.tex', 'foundation_frontier.tex',
       'multi_horizon_results.tex', 'performance_integrity_audit.tex',
       'seed_summary.tex', 'stage_b.tex', 'stagea_limits.tex',
       'Table7_StageC_PR.tex', 'temporal_drift_analysis.tex',
       'temporal_drift_family.tex', 'temporal_drift_lift.tex',
       'temporal_drift_prauc_lift.tex', 'unclustered_stability.tex',
       'metrics_config.tex']

ch5 = ['archetypal_attribution.tex', 'archetypal_attribution_weighted.tex',
       'semantic_feature_mapping.tex', 'spuriousness_index.tex',
       'Table12_Attrition_Timeline.tex']

ch6 = ['geographic_causal.tex', 'ipw_balance_diagnostics.tex',
       'ipw_diagnostics_summary.tex']

misc = ['test_table.tex']

for f in ch1:  mv(TBL_ROOT, f, 'chapter1_descriptive/' + f)
for f in ch4:  mv(TBL_ROOT, f, 'chapter4_performance/' + f)
for f in ch5:  mv(TBL_ROOT, f, 'chapter5_attribution/' + f)
for f in ch6:  mv(TBL_ROOT, f, 'chapter6_causal/' + f)
for f in misc: mv(TBL_ROOT, f, 'misc/' + f)

# Catch any remaining
for fname in sorted(os.listdir(TBL_ROOT)):
    if fname.endswith('.tex') and os.path.isfile(os.path.join(TBL_ROOT, fname)):
        mv(TBL_ROOT, fname, 'misc/' + fname)

print('\nFigures/ root:')
for f in sorted(os.listdir(FIG_ROOT)):
    tag = '[DIR]' if os.path.isdir(os.path.join(FIG_ROOT, f)) else '[FILE]'
    print('  ' + tag + ' ' + f)

print('\nTables/ root:')
for f in sorted(os.listdir(TBL_ROOT)):
    tag = '[DIR]' if os.path.isdir(os.path.join(TBL_ROOT, f)) else '[FILE]'
    print('  ' + tag + ' ' + f)
