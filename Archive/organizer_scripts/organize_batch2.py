import os, shutil

def mv(src, dst):
    if not os.path.exists(src):
        print('SKIP: ' + src)
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    print(os.path.basename(src) + ' -> ' + dst.replace(os.path.dirname(os.path.dirname(src)), ''))

# ── 1. Sequential_Clustermaps ─────────────────────────────────────────────────
SEQ = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Sequential_Clustermaps'
os.makedirs(os.path.join(SEQ, 'data'),     exist_ok=True)
os.makedirs(os.path.join(SEQ, 'heatmaps'), exist_ok=True)
os.makedirs(os.path.join(SEQ, 'ltr'),      exist_ok=True)

for yr in ['2018', '2019', '2020', '2021', '2022', '2023']:
    mv(os.path.join(SEQ, f'LTR_Clustermap_Matrix_{yr}.csv'),  os.path.join(SEQ, 'data',     f'LTR_Clustermap_Matrix_{yr}.csv'))
    mv(os.path.join(SEQ, f'plot_fixed_heatmap_{yr}.png'),     os.path.join(SEQ, 'heatmaps', f'plot_fixed_heatmap_{yr}.png'))
    mv(os.path.join(SEQ, f'plot_ltr_clustermap_{yr}.png'),    os.path.join(SEQ, 'ltr',      f'plot_ltr_clustermap_{yr}.png'))

mv(os.path.join(SEQ, 'plot_ltr_clustermap_timelapse.gif'), os.path.join(SEQ, 'ltr', 'plot_ltr_clustermap_timelapse.gif'))

print('Sequential_Clustermaps done:', sorted(os.listdir(SEQ)))

# ── 2. Draft_v1/scripts ───────────────────────────────────────────────────────
SCR = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Draft_v1\scripts'
os.makedirs(os.path.join(SCR, 'patching'), exist_ok=True)
os.makedirs(os.path.join(SCR, 'tex'),      exist_ok=True)

# Patching/rewrite tools
for f in ['fix_thesis.py', 'replace_thesis.py', 'patch.py', 'fix_captions.py', 'update_subsections.py']:
    mv(os.path.join(SCR, f), os.path.join(SCR, 'patching', f))

# TeX-specific helpers
mv(os.path.join(SCR, 'rewrite_lof.py'), os.path.join(SCR, 'tex', 'rewrite_lof.py'))

print('Draft_v1/scripts done:', sorted(os.listdir(SCR)))

# ── 3. Outputs ────────────────────────────────────────────────────────────────
OUT = r'C:\Users\dhl\data\Thesis\thesis\Outputs'
os.makedirs(os.path.join(OUT, 'grid_search'),  exist_ok=True)
os.makedirs(os.path.join(OUT, 'tex_exports'),  exist_ok=True)
os.makedirs(os.path.join(OUT, 'audits'),       exist_ok=True)
os.makedirs(os.path.join(OUT, 'catboost'),     exist_ok=True)

# Grid search results
for f in ['grid_tuning_results.csv', 'grid_tuning_results_expanded.csv',
          'grid_tuning_results_full.csv', 'grid_results_markdown.txt',
          'grid_results_universal_markdown.txt']:
    mv(os.path.join(OUT, f), os.path.join(OUT, 'grid_search', f))

# TeX/manuscript text exports
for f in ['thesis_full_text.txt', 'thesis_tex.txt', 'tex_files.txt', 'top_tex.txt',
          'jmlr_pages.txt', 'jmlr_pages_utf8.txt']:
    mv(os.path.join(OUT, f), os.path.join(OUT, 'tex_exports', f))

# Audit reports
for f in ['RECENCY_AUDIT_2026.md', 'figures_audit.txt']:
    mv(os.path.join(OUT, f), os.path.join(OUT, 'audits', f))

# Consolidate both catboost dirs into one
for cb_dir in ['catboost_info', 'catboost_info_draft_v1']:
    src_dir = os.path.join(OUT, cb_dir)
    dst_dir = os.path.join(OUT, 'catboost', cb_dir)
    if os.path.exists(src_dir):
        shutil.move(src_dir, dst_dir)
        print(cb_dir + ' -> catboost/' + cb_dir)

print('Outputs done:', sorted(os.listdir(OUT)))
