"""
Final Thesis_Draft org pass:
1. Apply fig_ch5_* naming to Sequential_Clustermaps heatmaps and ltr files
2. Apply dat_* naming to Thesis_Draft/Data CSVs
3. Collapse Figures_backup_hires into Figures/archive (or delete if duplicate)
4. Flatten Updates/ loose files into dated subdirs
5. Clean Project_One_Pager build artifacts
6. Rename Thesis_Draft/Scripts/ scripts
7. Move Draft_v1/Archive organizer scripts to root Archive/
"""
import os, shutil

TD   = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft'
DV1  = os.path.join(TD, 'Draft_v1')
ARC  = r'C:\Users\dhl\data\Thesis\thesis\Archive'

def mv(src, dst):
    if not os.path.exists(src):
        print('SKIP: ' + os.path.basename(src))
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        print('DUP: ' + os.path.basename(src))
        return
    shutil.move(src, dst)
    print(os.path.basename(src) + ' -> ' + os.path.relpath(os.path.dirname(dst), TD))

def ren(src, new_name):
    dst = os.path.join(os.path.dirname(src), new_name)
    if not os.path.exists(src):
        print('SKIP: ' + os.path.basename(src))
        return
    if os.path.exists(dst):
        print('DUP: ' + new_name)
        return
    os.rename(src, dst)
    print(os.path.basename(src) + '  ->  ' + new_name)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Sequential_Clustermaps — rename to fig_ch5_seq_slug scheme
#    Heatmaps = fixed-axis OOD domain heatmaps => ch5 attribution family
#    LTR clustermaps => ch5 clustering family
# ═══════════════════════════════════════════════════════════════════════════
HM  = os.path.join(TD, 'Sequential_Clustermaps', 'heatmaps')
LTR = os.path.join(TD, 'Sequential_Clustermaps', 'ltr')

for yr, seq in [('2018','01'),('2019','02'),('2020','03'),('2021','04'),('2022','05'),('2023','06')]:
    ren(os.path.join(HM,  f'plot_fixed_heatmap_{yr}.png'),     f'fig_ch5_hm_{seq}_{yr}_ood_domain_heatmap.png')
    ren(os.path.join(LTR, f'plot_ltr_clustermap_{yr}.png'),    f'fig_ch5_cl_{seq}_{yr}_ltr_clustermap.png')

ren(os.path.join(LTR, 'plot_ltr_clustermap_timelapse.gif'),    'fig_ch5_cl_07_ltr_clustermap_timelapse.gif')

# ═══════════════════════════════════════════════════════════════════════════
# 2. Thesis_Draft/Data — rename CSVs to dat_<domain>_<seq>_<slug> scheme
# ═══════════════════════════════════════════════════════════════════════════
ATT = os.path.join(TD, 'Data', 'attribution')
PRF = os.path.join(TD, 'Data', 'performance')

att_renames = [
    ('LTR_Feature_Attribution.csv',          'dat_attr_01_ltr_feature_attribution.csv'),
    ('LTR_Meta_Feature_Attribution.csv',      'dat_attr_02_ltr_meta_attribution.csv'),
    ('LTR_Weighted_Meta_Attribution.csv',     'dat_attr_03_ltr_weighted_meta_attribution.csv'),
    ('LTR_Clustermap_Meta_Attribution.csv',   'dat_attr_04_ltr_clustermap_meta_attribution.csv'),
    ('Preclustered_LTR_Omni_Clustermap.csv',  'dat_attr_05_preclustered_ltr_omni_clustermap.csv'),
    ('Recursive_LTR_Clustermap.csv',          'dat_attr_06_recursive_ltr_clustermap.csv'),
    ('Recursive_LTR_Omni_Clustermap.csv',     'dat_attr_07_recursive_ltr_omni_clustermap.csv'),
    ('Semantic_Cluster_Attribution.csv',      'dat_attr_08_semantic_cluster_attribution.csv'),
    ('Omni_Feature_Clusters_Explicit.csv',    'dat_attr_09_omni_feature_clusters_explicit.csv'),
    ('unsupervised_domain_dictionary.md',     'dat_attr_10_unsupervised_domain_dictionary.md'),
]
for old, new in att_renames:
    ren(os.path.join(ATT, old), new)

prf_renames = [
    ('Omnibus_LTR_Matrix.csv',                  'dat_perf_01_omnibus_ltr_matrix.csv'),
    ('Omnibus_LTR_Matrix_Extreme.csv',           'dat_perf_02_omnibus_ltr_matrix_extreme.csv'),
    ('Omnibus_Nontree_Matrix.csv',               'dat_perf_03_omnibus_nontree_matrix.csv'),
    ('Omnibus_Stacking_Matrix.csv',              'dat_perf_04_omnibus_stacking_matrix.csv'),
    ('Multiseed_Performance_Matrix.csv',         'dat_perf_05_multiseed_performance_matrix.csv'),
    ('Multiseed_Unclustered_Performance_Matrix.csv','dat_perf_06_multiseed_unclustered_performance.csv'),
    ('Multiseed_PRAUC_Table5_Validation.csv',    'dat_perf_07_multiseed_prauc_table5_validation.csv'),
    ('Mega_Matrix_Full_Results.csv',             'dat_perf_08_mega_matrix_full_results.csv'),
]
for old, new in prf_renames:
    ren(os.path.join(PRF, old), new)

# ═══════════════════════════════════════════════════════════════════════════
# 3. Figures_backup_hires — collapse into Figures/archive, then remove
# ═══════════════════════════════════════════════════════════════════════════
FIG_ARC = os.path.join(DV1, 'Figures', 'archive')
BACKUP  = os.path.join(DV1, 'Figures_backup_hires')

# Move the 2 loose files at the backup root
for f in ['Fig10_Hyperparameter_Sweeps.png', 'Fig9_Model_Comparison_PR_AUC.png']:
    mv(os.path.join(BACKUP, f), os.path.join(FIG_ARC, 'hires_' + f))

# Walk all sub-dirs and move files (they are backup copies of already-moved files)
for dirpath, dirs, files in os.walk(BACKUP):
    for f in files:
        fp = os.path.join(dirpath, f)
        mv(fp, os.path.join(FIG_ARC, 'hires_' + f))

# Remove the now-empty backup tree
shutil.rmtree(BACKUP, ignore_errors=True)
print('Removed Figures_backup_hires/')

# ═══════════════════════════════════════════════════════════════════════════
# 4. Updates/ — move loose files into the 2026-date subdirs they belong to
# ═══════════════════════════════════════════════════════════════════════════
UPD = os.path.join(TD, 'Updates')

# Create date subdirs
for d in ['2025-12', '2026-01', '2026-02', '2026-03', 'templates']:
    os.makedirs(os.path.join(UPD, d), exist_ok=True)

upd_map = {
    '2025-12_Advisor_Status_Update_Template.pdf':                      '2025-12',
    '2026-01_Status_Update_GDOC_READY.txt':                            '2026-01',
    '2026-02_01_Status_Update.docx':                                   '2026-02',
    '2026-02_01_Status_Update_GDOC_READY.txt':                         '2026-02',
    '2026-03_02_Status_Update.docx':                                   '2026-03',
    'Columbia_Thesis-Status_Update-Daniel_Hardesty_Lewis-20260201-Predicting_NIMBYism.docx': '2026-02',
    'Columbia_Thesis_Status_Update_Daniel_Hardesty_Lewis_2026-02-01_Predicting_NIMBYism.docx': '2026-02',
    'Weekly_Status_Update_Template.pdf':                               'templates',
    'email_to_dory.md':                                                '2026-03',
}
# Rename to consistent scheme first, then move
raw_map = {
    '2025-12-22_Advisor_Status_Update_Template.pdf':                        ('2025-12', '2025-12_advisor_status_update_template.pdf'),
    '2026-01-21_Status_Update_GDOC_READY.txt':                              ('2026-01', '2026-01-21_status_update_gdoc.txt'),
    '2026-02-01_Status_Update.docx':                                        ('2026-02', '2026-02-01_status_update.docx'),
    '2026-02-01_Status_Update_GDOC_READY.txt':                              ('2026-02', '2026-02-01_status_update_gdoc.txt'),
    '2026-03-02_Status_Update.docx':                                        ('2026-03', '2026-03-02_status_update.docx'),
    'Columbia_Thesis-Status_Update-Daniel_Hardesty_Lewis-20260201-Predicting_NIMBYism.docx': ('2026-02', '2026-02-01_columbia_status_update.docx'),
    'Columbia_Thesis_Status_Update_Daniel_Hardesty_Lewis_2026-02-01_Predicting_NIMBYism.docx': ('2026-02', '2026-02-01_columbia_status_update_v2.docx'),
    'Weekly_Status_Update_Template.pdf':                                    ('templates', 'weekly_status_update_template.pdf'),
    'email_to_dory.md':                                                     ('2026-03', '2026-03_email_to_dory.md'),
}
for old_name, (subdir, new_name) in raw_map.items():
    src = os.path.join(UPD, old_name)
    dst = os.path.join(UPD, subdir, new_name)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        print(old_name + ' -> Updates/' + subdir + '/' + new_name)

# Move the 2026-01-21 subdir content into 2026-01/
src_dir = os.path.join(UPD, '2026-01-21_Status_Update')
if os.path.isdir(src_dir):
    for f in os.listdir(src_dir):
        shutil.move(os.path.join(src_dir, f), os.path.join(UPD, '2026-01', f))
    shutil.rmtree(src_dir)
    print('Merged 2026-01-21_Status_Update/ -> Updates/2026-01/')

# Move Templates/ content into templates/
src_t = os.path.join(UPD, 'Templates')
if os.path.isdir(src_t):
    for f in os.listdir(src_t):
        shutil.move(os.path.join(src_t, f), os.path.join(UPD, 'templates', f))
    shutil.rmtree(src_t)
    print('Merged Templates/ -> Updates/templates/')

# ═══════════════════════════════════════════════════════════════════════════
# 5. Project_One_Pager — remove LaTeX build artifacts, keep source + PDF
# ═══════════════════════════════════════════════════════════════════════════
POP = os.path.join(TD, 'Project_One_Pager')
for f in os.listdir(POP):
    if f.endswith('.out'):
        os.remove(os.path.join(POP, f))
        print('Deleted build artifact: ' + f)

# ═══════════════════════════════════════════════════════════════════════════
# 6. Thesis_Draft/Scripts/ — rename to functional scheme
# ═══════════════════════════════════════════════════════════════════════════
SCR = os.path.join(TD, 'Scripts')
scr_renames = [
    ('fix_text.py',         'util_fix_tex_text.py'),
    ('make_docx.py',        'util_export_docx.py'),
    ('make_dory_update.py', 'util_make_advisor_update.py'),
]
for old, new in scr_renames:
    ren(os.path.join(SCR, old), new)

# ═══════════════════════════════════════════════════════════════════════════
# 7. Draft_v1/Archive — move organizer scripts to root Archive/
#    keep .bak and test.pdf since they're draft-specific
# ═══════════════════════════════════════════════════════════════════════════
DV1_ARC = os.path.join(DV1, 'Archive')
for f in ['organize_draft_v1.py', 'organize_figures_tables.py']:
    mv(os.path.join(DV1_ARC, f),
       os.path.join(ARC, 'organizer_scripts', f))

print('\nThesis_Draft top-level:')
for f in sorted(os.listdir(TD)):
    tag = '[DIR]' if os.path.isdir(os.path.join(TD, f)) else '[FILE]'
    print('  ' + tag + ' ' + f)
