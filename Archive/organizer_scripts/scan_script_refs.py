import os, re

THESIS = r'C:\Users\dhl\data\Thesis\thesis'
roots = [
    os.path.join(THESIS, 'scripts'),
    os.path.join(THESIS, 'scratch'),
    os.path.join(THESIS, 'Thesis_Draft', 'Draft_v1', 'scripts'),
]

# Scripts calling other scripts
call_pat = re.compile(
    r'(subprocess|os\.system|python|exec|run)\b.{0,120}'
    r'(generate_|plot_|fix_|patch_|update_|replace_|rewrite_|12_generate|13_render|compute_ace|tag_generated|unclustered_table|buffer_map)',
    re.I
)

# Hardcoded paths to data/thesis dirs
path_pat = re.compile(
    r'["\']([A-Za-z]:[/\\][^"\']{10,}|(?:\.\.?/|\.\.?\\)[^"\']{5,}|'
    r'(?:Data|Warehouse_As_Of|Panel|Thesis_Draft|Outputs|scratch)[/\\][^"\']{5,})["\']'
)

for root in roots:
    if not os.path.isdir(root):
        continue
    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            if not (f.endswith('.py') or f.endswith('.ps1')):
                continue
            fp = os.path.join(dirpath, f)
            try:
                lines = open(fp, encoding='utf-8', errors='ignore').readlines()
            except:
                continue
            hits = []
            for i, line in enumerate(lines, 1):
                stripped = line.rstrip()
                if call_pat.search(stripped):
                    hits.append(f'  [CALL] L{i}: {stripped[:120]}')
                else:
                    for m in path_pat.finditer(stripped):
                        val = m.group(1)
                        # Only flag if it looks like a real path containing old structure
                        if any(x in val for x in ['Tables/', 'Tables\\', 'Figures/', 'Figures\\',
                                                    'Warehouse_As_Of', 'Panel/', 'Panel\\',
                                                    'generate_', 'plot_', 'patch_']):
                            hits.append(f'  [PATH] L{i}: {stripped[:120]}')
                            break
            if hits:
                rel = os.path.relpath(fp, THESIS)
                print(rel)
                for h in hits[:10]:
                    print(h)
                print()
