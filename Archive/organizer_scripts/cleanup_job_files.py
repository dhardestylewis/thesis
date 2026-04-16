import os
import urllib.request

DV1 = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Draft_v1'

# Restore .gitignore
gitignore_path = os.path.join(DV1, '.gitignore')
with open(gitignore_path, 'w', encoding='utf-16') as f:
    f.write('Lewis_2026_NIMBYism_Austin_Thesis.pdf\n')
print('Restored: .gitignore')

# Delete actual $job files
removed = []
for f in os.listdir(DV1):
    if f.startswith('$job.'):
        fp = os.path.join(DV1, f)
        os.remove(fp)
        removed.append(f)
print('Deleted:', removed)
