import os
import re

def clean_directory(directory):
    pattern = re.compile(r'^\s*(sns\.set_theme\(.*?\)|sns\.set_style\(.*?\)|plt\.style\.use\(.*?\))', re.MULTILINE | re.DOTALL)
    
    # Just a simple line-by-line check might be safer to avoid breaking multi-line strings, 
    # but let's just do line by line:
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith('.py') or file == 'thesis_style.py':
                continue
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            new_lines = []
            modified = False
            for line in lines:
                if 'sns.set_theme' in line or 'sns.set_style' in line or 'plt.style.use' in line:
                    target = line.strip()
                    if target.startswith('sns.set_theme') or target.startswith('sns.set_style') or target.startswith('plt.style.use'):
                        # Comment it out
                        new_lines.append('# Removed local style: ' + line)
                        modified = True
                        continue
                new_lines.append(line)
                
            if modified:
                with open(path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                print(f"Cleaned styles in {path}")

clean_directory(r"c:\Users\dhl\data\thesis\thesis\Analysis\Scripts")
