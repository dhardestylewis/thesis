import os
import re

DRAFT = r'C:\Users\dhl\data\Thesis\thesis\Thesis_Draft\Draft_v1'

tex_files = [os.path.join(DRAFT, 'Austin_NIMBY_Thesis_Draft.tex')]
sections_dir = os.path.join(DRAFT, 'Sections')
for f in os.listdir(sections_dir):
    if f.endswith('.tex'):
        tex_files.append(os.path.join(sections_dir, f))

inc_paths = set()
input_paths = set()

for tf in tex_files:
    with open(tf, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    for m in re.finditer(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}', text):
        inc_paths.add(m.group(1))
    for m in re.finditer(r'\\input\{([^}]+)\}', text):
        input_paths.add(m.group(1))

print("Includegraphics:")
for p in sorted(inc_paths):
    print("  " + p)
print("\nInput:")
for p in sorted(input_paths):
    print("  " + p)
