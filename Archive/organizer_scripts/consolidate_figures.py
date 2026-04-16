"""
Consolidates:
  - Thesis_Draft/Figures/  (42 already-renamed fig_ch4/5_* files)
  - Thesis_Draft/Draft_v1/Figures/  (130 files, mixed naming)

Into a single canonical:
  - Thesis_Draft/Draft_v1/Figures/
    ch2/ ch4/ ch5/ ch6/ appendix/ core/ archive/ audit/

All files renamed to fig_ch<N>_<seq>_<slug>.<ext>
"""

import os, shutil

DRAFT = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Draft_v1'
FIG   = os.path.join(DRAFT, 'Figures')
EXT_FIG = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Figures'  # the other one

for d in ['ch2', 'ch4', 'ch5', 'ch6', 'appendix', 'core', 'archive', 'audit', 'exhibits']:
    os.makedirs(os.path.join(FIG, d), exist_ok=True)

def mv(src, dst_subdir, new_name=None):
    if not os.path.exists(src):
        print('SKIP: ' + os.path.basename(src))
        return
    fname = new_name if new_name else os.path.basename(src)
    dst = os.path.join(FIG, dst_subdir, fname)
    if os.path.exists(dst):
        print('DUP (skip): ' + fname)
        return
    shutil.move(src, dst)
    print(os.path.basename(src) + '  ->  ' + dst_subdir + '/' + fname)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Migrate Thesis_Draft/Figures/ (already renamed fig_ch4/5_* files)
# ═══════════════════════════════════════════════════════════════════════════
for subdir, chapter_dst in [('performance','ch4'), ('attribution','ch5'),
                              ('clustering','ch5'),  ('ablation','ch5')]:
    src_dir = os.path.join(EXT_FIG, subdir)
    if not os.path.isdir(src_dir): continue
    for f in sorted(os.listdir(src_dir)):
        mv(os.path.join(src_dir, f), chapter_dst)

# Remove now-empty external Figures dir
try:
    shutil.rmtree(EXT_FIG)
    print('\nRemoved: Thesis_Draft/Figures/ (consolidated)')
except: pass

# ═══════════════════════════════════════════════════════════════════════════
# 2. Chapter4 — standardize mixed F#/Fig/fig_ prefix
# ═══════════════════════════════════════════════════════════════════════════
ch4 = os.path.join(FIG, 'Chapter4')
ch4_renames = [
    ('F12_Opposition_PR.png',                     'fig_ch4_20_opposition_pr_curve.png'),
    ('F22_Joint_Policy_Map.png',                  'fig_ch4_21_joint_policy_map.png'),
    ('F23_Spatial_Error.png',                     'fig_ch4_22_spatial_error.png'),
    ('F8_Calibration.png',                        'fig_ch4_23_calibration.png'),
    ('Fig_3D_Temporal_Drift.png',                 'fig_ch4_24_3d_temporal_drift.png'),
    ('StageA_Figure3_PR_Curves.png',              'fig_ch4_25_stagea_pr_curves.png'),
    ('fig1_spatial_distribution.png',             'fig_ch4_26_spatial_distribution_hires.png'),
    ('fig_all_stages_seed.pdf',                   'fig_ch4_27_all_stages_seed.pdf'),
    ('fig_attribution_divergence.pdf',            'fig_ch4_28_attribution_divergence.pdf'),
    ('fig_attribution_divergence_matrix.pdf',     'fig_ch4_29_attribution_divergence_matrix.pdf'),
    ('fig_attribution_longitudinal.pdf',          'fig_ch4_30_attribution_longitudinal.pdf'),
    ('fig_combined_calibration_reliability.pdf',  'fig_ch4_31_calibration_reliability.pdf'),
    ('fig_combined_calibration_reliability.png',  'fig_ch4_31_calibration_reliability.png'),
    ('fig_fpr_fnr_longitudinal.pdf',              'fig_ch4_32_fpr_fnr_longitudinal.pdf'),
    ('fig_full_frontier_seed.pdf',                'fig_ch4_33_full_frontier_seed.pdf'),
    ('fig_ood_offset_decay.pdf',                  'fig_ch4_34_ood_offset_decay.pdf'),
    ('fig_ood_seed_variance.pdf',                 'fig_ch4_35_ood_seed_variance.pdf'),
    ('fig_seed_stability.pdf',                    'fig_ch4_36_seed_stability.pdf'),
    ('fig_stage_b_continuous_seed_mae.pdf',       'fig_ch4_37_stage_b_continuous_mae.pdf'),
    ('fig_stage_b_perclass.pdf',                  'fig_ch4_38_stage_b_perclass.pdf'),
    ('fig_stage_b_perclass_boxplot.pdf',          'fig_ch4_39_stage_b_perclass_boxplot.pdf'),
    ('fig_stage_b_seed.pdf',                      'fig_ch4_40_stage_b_seed.pdf'),
]
for old, new in ch4_renames:
    mv(os.path.join(ch4, old), 'ch4', new)
# tex snippet is not a figure - move to Tables
tex_snip = os.path.join(ch4, 'fig_ipw_overlap_balance.tex')
if os.path.exists(tex_snip):
    shutil.move(tex_snip, os.path.join(DRAFT, 'Tables', 'misc', 'fig_ipw_overlap_balance.tex'))
    print('fig_ipw_overlap_balance.tex -> Tables/misc/')
shutil.rmtree(ch4, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
# 3. Chapter5 — causal/DiD figures
# ═══════════════════════════════════════════════════════════════════════════
ch5 = os.path.join(FIG, 'Chapter5')
ch5_renames = [
    ('Electoral_Placebo_DiD.png',   'fig_ch5_30_electoral_placebo_did.png'),
    ('F16_Petition_RD.png',         'fig_ch5_31_petition_rd.png'),
    ('F17_DiD_EventStudy.png',      'fig_ch5_32_did_event_study.png'),
    ('F17_HOME_EventStudy.png',     'fig_ch5_33_home_event_study.png'),
    ('fig_causal_context_did.png',  'fig_ch5_34_causal_context_did.png'),
]
for old, new in ch5_renames:
    mv(os.path.join(ch5, old), 'ch5', new)
# Duplicates from Archive → archive
for f in ['F17_DiD_EventStudy_Old.png', 'Fig10_Hyperparameter_Sweeps.png',
          'Fig9_Model_Comparison_PR_AUC.png']:
    mv(os.path.join(ch5, f), 'archive', f)
shutil.rmtree(ch5, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
# 4. Chapter6
# ═══════════════════════════════════════════════════════════════════════════
ch6 = os.path.join(FIG, 'Chapter6')
ch6_renames = [
    ('F19_TextFrame_Composition.png',          'fig_ch6_01_textframe_composition.png'),
    ('F20_Stakeholder_Heatmap.png',            'fig_ch6_02_stakeholder_heatmap.png'),
    ('F20_Stakeholder_Spatial_Multiples.png',  'fig_ch6_03_stakeholder_spatial_multiples.png'),
]
for old, new in ch6_renames:
    mv(os.path.join(ch6, old), 'ch6', new)
shutil.rmtree(ch6, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
# 5. SHAP_MetaClustering
# ═══════════════════════════════════════════════════════════════════════════
shap = os.path.join(FIG, 'SHAP_MetaClustering')
mv(os.path.join(shap, 'meta_attribution_clustermap.pdf'),          'ch5', 'fig_ch5_35_meta_attribution_clustermap.pdf')
mv(os.path.join(shap, 'meta_attribution_clustermap_weighted.pdf'), 'ch5', 'fig_ch5_36_meta_attribution_clustermap_weighted.pdf')
shutil.rmtree(shap, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
# 6. Track1_Predictive — flatten nested Figures/ subdir, deduplicate
# ═══════════════════════════════════════════════════════════════════════════
t1p = os.path.join(FIG, 'Track1_Predictive')
t1p_renames = [
    ('fig_clustered_dynamic.pdf',   'fig_ch4_50_clustered_dynamic.pdf'),
    ('fig_clustered_dynamic.png',   'fig_ch4_50_clustered_dynamic.png'),
    ('fig_unclustered_dynamic.pdf', 'fig_ch4_51_unclustered_dynamic.pdf'),
    ('fig_unclustered_dynamic.png', 'fig_ch4_51_unclustered_dynamic.png'),
]
for old, new in t1p_renames:
    mv(os.path.join(t1p, old), 'ch4', new)
# Nested Figures subdir
nested = os.path.join(t1p, 'Figures')
if os.path.isdir(nested):
    mv(os.path.join(nested, 'Typology_Temporal_Incidence.png'), 'ch4', 'fig_ch4_52_typology_temporal_incidence.png')
    for f in os.listdir(nested):  # duplicates
        fp = os.path.join(nested, f)
        if os.path.isfile(fp): os.remove(fp)
    shutil.rmtree(nested, ignore_errors=True)
shutil.rmtree(t1p, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
# 7. Track1_Exhibits — already well-named, move as-is to exhibits/
# ═══════════════════════════════════════════════════════════════════════════
t1e = os.path.join(FIG, 'Track1_Exhibits')
if os.path.isdir(t1e):
    for f in sorted(os.listdir(t1e)):
        mv(os.path.join(t1e, f), 'exhibits')
    shutil.rmtree(t1e, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
# 8. Core PDFs
# ═══════════════════════════════════════════════════════════════════════════
core_src = os.path.join(FIG, 'Core')
core_renames = [
    ('fig_calibration_ece.pdf',   'fig_core_01_calibration_ece.pdf'),
    ('fig_feature_importance.pdf','fig_core_02_feature_importance.pdf'),
    ('fig_policy_regimes.pdf',    'fig_core_03_policy_regimes.pdf'),
    ('fig_temporal_drift.pdf',    'fig_core_04_temporal_drift.pdf'),
]
for old, new in core_renames:
    mv(os.path.join(core_src, old), 'core', new)
shutil.rmtree(core_src, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
# 9. Exploratory_EDA
# ═══════════════════════════════════════════════════════════════════════════
eda = os.path.join(FIG, 'Exploratory_EDA')
mv(os.path.join(eda, 'waller_buffer_map.png'), 'ch2', 'fig_ch2_01_waller_buffer_map.png')
shutil.rmtree(eda, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
# 10. Qualitative_Appendix
# ═══════════════════════════════════════════════════════════════════════════
qa = os.path.join(FIG, 'Qualitative_Appendix')
app_renames = [
    ('interview_guide.pdf',              'fig_app_01_interview_guide.pdf'),
    ('project_overview.pdf',             'fig_app_02_project_overview.pdf'),
    ('protocol_summary.pdf',             'fig_app_03_protocol_summary.pdf'),
    ('sample_protest_petition.pdf',      'fig_app_04_sample_protest_petition.pdf'),
    ('sample_protest_petition_v15.pdf',  'fig_app_04_sample_protest_petition_v15.pdf'),
]
for old, new in app_renames:
    mv(os.path.join(qa, old), 'appendix', new)
shutil.rmtree(qa, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
# 11. Archive_Pipelines → archive/
# ═══════════════════════════════════════════════════════════════════════════
arc = os.path.join(FIG, 'Archive_Pipelines')
if os.path.isdir(arc):
    for f in sorted(os.listdir(arc)):
        mv(os.path.join(arc, f), 'archive', f)
    shutil.rmtree(arc, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
# 12. audit/ — already in right place
# ═══════════════════════════════════════════════════════════════════════════
# already at Figures/audit/ — leave it

print('\n\nFinal Figures/ structure:')
for d in sorted(os.listdir(FIG)):
    dp = os.path.join(FIG, d)
    if os.path.isdir(dp):
        count = len([f for f in os.listdir(dp) if os.path.isfile(os.path.join(dp, f))])
        print(f'  {d}/  ({count} files)')
    else:
        print(f'  {d}')
