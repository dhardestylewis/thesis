import os

def check_missing_style(directory):
    missing_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith('.py') or file == 'thesis_style.py' or file == 'apply_thesis_style.py' or file == 'clean_styles.py' or file == 'check_missing.py':
                continue
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            has_mpl = 'import matplotlib' in content or 'import seaborn' in content or 'from matplotlib' in content
            has_style = 'set_thesis_style' in content
            
            if has_mpl and not has_style:
                missing_files.append(path)
                
    for f in missing_files:
        print(f"Missing style: {f}")

check_missing_style(r"c:\Users\dhl\data\thesis\thesis\Analysis\Scripts")
