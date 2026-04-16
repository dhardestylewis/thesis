import os, shutil

ROOT = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Draft_v1'

def mv(src, dst):
    s = os.path.join(ROOT, src)
    d = os.path.join(ROOT, dst)
    if not os.path.exists(s):
        print('SKIP: ' + src)
        return
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.move(s, d)
    print(src + ' -> ' + dst)

os.makedirs(os.path.join(ROOT, 'build'), exist_ok=True)

# LaTeX build artifacts for both compile variants
build_exts = ['.aux', '.bbl', '.blg', '.fdb_latexmk', '.fls', '.lof', '.log', '.lot', '.out', '.toc']
for ext in build_exts:
    mv('Austin_NIMBY_Thesis_Draft' + ext,       'build/Austin_NIMBY_Thesis_Draft' + ext)
    mv('Austin_NIMBY_Thesis_Draft_build' + ext, 'build/Austin_NIMBY_Thesis_Draft_build' + ext)

mv('Austin_NIMBY_Thesis_Draft_build.pdf',       'build/Austin_NIMBY_Thesis_Draft_build.pdf')
mv('Austin_NIMBY_Thesis_Draft.git-blame.txt',   'build/Austin_NIMBY_Thesis_Draft.git-blame.txt')

# Dollar-sign $job.* latexmk temp files
for ext in ['.bbl', '.blg', '.fdb_latexmk', '.fls', '.pdf']:
    mv('$job' + ext, 'build/$job' + ext)

# Compilation logs and aux text dumps
for f in ['compilation_output.txt', 'compilation_output_pass2.txt', 'full_log.txt', 'aux_content.txt']:
    mv(f, 'build/' + f)

# CatBoost artifacts that shouldn't be inside the draft dir
if os.path.exists(os.path.join(ROOT, 'catboost_info')):
    shutil.move(os.path.join(ROOT, 'catboost_info'), os.path.join(ROOT, 'build', 'catboost_info'))
    print('catboost_info/ -> build/catboost_info/')

# Stray .tex scratch files -> Tables/
mv('summary_stats_table.tex', 'Tables/summary_stats_table.tex')
mv('test_table.tex',          'Tables/test_table.tex')

print('\nDone.')
print('\nRemaining at root:')
for f in sorted(os.listdir(ROOT)):
    tag = '[DIR]' if os.path.isdir(os.path.join(ROOT, f)) else '[FILE]'
    print('  ' + tag + ' ' + f)
