"""
Rename Austin_NIMBY_Thesis_Draft -> Lewis_2026_NIMBYism_Austin_Thesis
across all files: .tex, .pdf, build_thesis.ps1, .gitignore, build/ artifacts.
"""
import os, shutil

OLD = 'Austin_NIMBY_Thesis_Draft'
NEW = 'Lewis_2026_NIMBYism_Austin_Thesis'
DV1 = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Draft_v1'

def patch(fp, old, new):
    try:
        raw = open(fp, 'rb').read()
        # Try UTF-16 (the .gitignore is UTF-16)
        for enc in ['utf-16', 'utf-8', 'cp1252']:
            try:
                text = raw.decode(enc)
                new_text = text.replace(old, new)
                if new_text != text:
                    open(fp, 'w', encoding=enc).write(new_text)
                    print(f'Patched ({enc}): ' + os.path.relpath(fp, DV1))
                return
            except: continue
    except Exception as e:
        print(f'ERR {fp}: {e}')

def ren(src, dst):
    if os.path.exists(src):
        if os.path.exists(dst):
            print('DUP: ' + os.path.basename(dst))
        else:
            os.rename(src, dst)
            print(os.path.basename(src) + '  ->  ' + os.path.basename(dst))
    else:
        print('SKIP: ' + os.path.basename(src))

# 1. Rename the .tex and .pdf source files
ren(os.path.join(DV1, OLD + '.tex'), os.path.join(DV1, NEW + '.tex'))
ren(os.path.join(DV1, OLD + '.pdf'), os.path.join(DV1, NEW + '.pdf'))

# 2. Patch build_thesis.ps1
ps1 = r'C:\Users\dhl\data\Thesis\thesis\scripts\pipeline\build_thesis.ps1'
patch(ps1, OLD, NEW)
# Also patch the fallback _build suffix
patch(ps1, OLD + '_build', NEW + '_build')

# 3. Patch .gitignore (UTF-16)
patch(os.path.join(DV1, '.gitignore'), OLD, NEW)

# 4. Rename build/canonical artifacts
BUILD_CAN = os.path.join(DV1, 'build', 'canonical')
BUILD_SEC = os.path.join(DV1, 'build', 'secondary')
for build_dir in [BUILD_CAN, BUILD_SEC]:
    if not os.path.isdir(build_dir): continue
    for f in sorted(os.listdir(build_dir)):
        if OLD in f:
            ren(os.path.join(build_dir, f),
                os.path.join(build_dir, f.replace(OLD, NEW)))

# 5. Patch the .tex itself for any \jobname or self-references
tex = os.path.join(DV1, NEW + '.tex')
patch(tex, OLD, NEW)

# 6. Rename the .bak in Archive if present
bak = os.path.join(DV1, 'Archive', OLD + '.tex.bak')
ren(bak, os.path.join(DV1, 'Archive', NEW + '.tex.bak'))

print('\nDone. Draft_v1 root files:')
for f in sorted(os.listdir(DV1)):
    if os.path.isfile(os.path.join(DV1, f)):
        print('  ' + f)
