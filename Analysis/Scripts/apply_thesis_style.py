import os

injection = """
import sys
try:
    # Attempt to locate the root Scripts directory
    _curr = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(_curr) != 'Scripts' and os.path.dirname(_curr) != _curr:
        _curr = os.path.dirname(_curr)
    if _curr not in sys.path:
        sys.path.insert(0, _curr)
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass
"""

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith('.py') or file == 'thesis_style.py' or file == 'apply_thesis_style.py':
                continue
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                continue
            
            if 'from thesis_style import set_thesis_style' not in content:
                lines = content.split('\n')
                insert_idx = -1
                for i, line in enumerate(lines):
                    if line.startswith('import matplotlib') or line.startswith('import seaborn'):
                        insert_idx = i
                if insert_idx != -1:
                    lines.insert(insert_idx + 1, injection)
                    with open(path, 'w', encoding='utf-8', newline='') as f:
                        f.write('\n'.join(lines))
                    print(f"Injected into {path}")

process_directory(r"c:\Users\dhl\data\thesis\thesis\Analysis\Scripts")
print("Injection complete.")
