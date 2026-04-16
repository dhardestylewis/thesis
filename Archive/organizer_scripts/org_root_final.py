"""
Top-level root cleanup:
1. Move compile_thesis.ps1 -> scripts/pipeline/
2. Move loose run_*.py entry points -> scripts/pipeline/
3. Move HardestyLewis_Daniel_Thesis_Draft_April2026.pdf -> Submitted/
4. Move Tables/metrics_config.tex -> Thesis_Draft/Draft_v1/Tables/chapter4_performance/
   (it is a stale duplicate at root)
5. Delete dev/null (empty dir artifact)
6. Delete __pycache__/
7. Standardize dir casing: lowercase dirs -> TitleCase to match Archive/, Data/, etc.
   BUT only the thesis-authored dirs, not python/git conventioned ones (.git, .venv, src, etc.)
"""
import os, shutil

ROOT = r'C:\Users\dhl\data\Thesis\thesis'

def mv(src, dst):
    if not os.path.exists(src):
        print('SKIP: ' + os.path.basename(src))
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    print(os.path.basename(src) + '  ->  ' + os.path.relpath(os.path.dirname(dst), ROOT))

# 1. compile_thesis.ps1 -> scripts/pipeline/
mv(os.path.join(ROOT, 'compile_thesis.ps1'),
   os.path.join(ROOT, 'scripts', 'pipeline', 'compile_thesis.ps1'))

# 2. Loose entry-point scripts -> scripts/pipeline/
for f in ['execute_thesis_pipeline.py', 'run_thesis.py',
          'run_warehouse_builder.py', 'run_rolling_origin_lift.py']:
    mv(os.path.join(ROOT, f), os.path.join(ROOT, 'scripts', 'pipeline', f))

# 3. Submitted artifact PDF
mv(os.path.join(ROOT, 'HardestyLewis_Daniel_Thesis_Draft_April2026.pdf'),
   os.path.join(ROOT, 'Submitted', 'HardestyLewis_Daniel_Thesis_Draft_April2026.pdf'))

# 4. Stale root Tables/ (single orphan file)
stale_tbl = os.path.join(ROOT, 'Tables', 'metrics_config.tex')
canon_tbl  = os.path.join(ROOT, 'Thesis_Draft', 'Draft_v1', 'Tables',
                           'chapter4_performance', 'tbl_ch4_08_metrics_config.tex')
if os.path.exists(stale_tbl):
    if os.path.exists(canon_tbl):
        os.remove(stale_tbl)
        print('Deleted stale duplicate: Tables/metrics_config.tex')
    else:
        mv(stale_tbl, canon_tbl)
# Remove now-empty root Tables/
root_tbl = os.path.join(ROOT, 'Tables')
if os.path.isdir(root_tbl) and not os.listdir(root_tbl):
    os.rmdir(root_tbl)
    print('Removed empty: Tables/')

# 5. dev/null artifact
dev_null = os.path.join(ROOT, 'dev', 'null')
if os.path.exists(dev_null):
    os.remove(dev_null)
    print('Deleted: dev/null')
dev_dir = os.path.join(ROOT, 'dev')
if os.path.isdir(dev_dir) and not os.listdir(dev_dir):
    os.rmdir(dev_dir)
    print('Removed empty: dev/')

# 6. __pycache__
pycache = os.path.join(ROOT, '__pycache__')
if os.path.isdir(pycache):
    shutil.rmtree(pycache)
    print('Removed: __pycache__/')

# 7. Casing: author-controlled lowercase dirs that should match TitleCase convention
# Leave: .git, .venv, .claude, .meta, .github, src (Python convention)
# Standardize: scripts, scratch, configs, docs, registries, reporting, thesis_pipeline
casing_fixes = [
    ('scripts',         'Scripts'),
    ('scratch',         'Scratch'),
    ('configs',         'Configs'),
    ('docs',            'Docs'),
    ('registries',      'Registries'),
    ('reporting',       'Reporting'),
    ('thesis_pipeline', 'Thesis_Pipeline'),
]
for old, new in casing_fixes:
    src = os.path.join(ROOT, old)
    dst = os.path.join(ROOT, new)
    if os.path.isdir(src) and not os.path.isdir(dst):
        # Windows rename is case-insensitive so need a temp intermediate
        tmp = os.path.join(ROOT, old + '_tmp_rename')
        os.rename(src, tmp)
        os.rename(tmp, dst)
        print(old + '/  ->  ' + new + '/')

print('\nDone. Top-level:')
for f in sorted(os.listdir(ROOT)):
    tag = '[DIR] ' if os.path.isdir(os.path.join(ROOT, f)) else '[FILE]'
    print('  ' + tag + ' ' + f)
