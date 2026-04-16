import os, shutil

BUILD   = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Draft_v1\build'
OUTPUTS = r'C:\Users\dhl\data\Thesis\thesis\Outputs'

def mv(src, dst):
    s = os.path.join(BUILD, src)
    d = os.path.join(BUILD, dst)
    if not os.path.exists(s):
        print('SKIP: ' + src)
        return
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.move(s, d)
    print(src + ' -> ' + dst)

os.makedirs(os.path.join(BUILD, 'canonical'), exist_ok=True)
os.makedirs(os.path.join(BUILD, 'secondary'), exist_ok=True)
os.makedirs(os.path.join(BUILD, 'logs'),      exist_ok=True)

# Canonical build intermediates
for ext in ['.aux', '.bbl', '.blg', '.fdb_latexmk', '.fls', '.lof', '.log', '.lot', '.out', '.toc', '.git-blame.txt']:
    mv('Austin_NIMBY_Thesis_Draft' + ext, 'canonical/Austin_NIMBY_Thesis_Draft' + ext)

# Secondary _build variant
for ext in ['.aux', '.bbl', '.blg', '.fdb_latexmk', '.fls', '.lof', '.log', '.lot', '.out', '.toc', '.pdf']:
    mv('Austin_NIMBY_Thesis_Draft_build' + ext, 'secondary/Austin_NIMBY_Thesis_Draft_build' + ext)

# $job.* temp files
job_prefix = '$job'
for ext in ['.bbl', '.blg', '.fdb_latexmk', '.fls', '.pdf']:
    mv(job_prefix + ext, 'secondary/' + job_prefix + ext)

# Compilation logs
for f in ['compilation_output.txt', 'compilation_output_pass2.txt', 'full_log.txt', 'aux_content.txt']:
    mv(f, 'logs/' + f)

# catboost_info does not belong in a LaTeX build dir
ci_src = os.path.join(BUILD, 'catboost_info')
if os.path.exists(ci_src):
    ci_dst = os.path.join(OUTPUTS, 'catboost_info_draft_v1')
    shutil.move(ci_src, ci_dst)
    print('catboost_info/ -> Outputs/catboost_info_draft_v1/')

print('\nDone. build/ root:')
for f in sorted(os.listdir(BUILD)):
    tag = '[DIR]' if os.path.isdir(os.path.join(BUILD, f)) else '[FILE]'
    print('  ' + tag + ' ' + f)
