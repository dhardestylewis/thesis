import re

pth = r'Thesis_Draft\Draft_v1\Lewis_2026_NIMBYism_Austin_Thesis.tex'
with open(pth, 'r', encoding='utf-8') as f:
    text = f.read()

def safe_replace(full_text, target, repl):
    target_pattern = re.escape(target).replace(r'\ ', r'\s+').replace(r'\n', r'\s+')
    return re.sub(target_pattern, repl, full_text, count=1)

# TabNet Fix
t1 = r"\textbf{Deep Architectures (MLP/TabNet)}"
r1 = r"\textbf{Deep Architectures (MLP)}"
text = safe_replace(text, t1, r1)

# 10 Architecture Gauntlet Fix
t2 = r"the expanded 10-architecture gauntlet explicitly enforces"
r2 = r"the unified 4-architecture benchmark explicitly enforces"
text = safe_replace(text, t2, r2)

t3 = r"ensuring that all 10 architectures are evaluated strictly on"
r3 = r"ensuring that all 4 architectures are evaluated strictly on"
text = safe_replace(text, t3, r3)

with open(pth, 'w', encoding='utf-8') as f:
    f.write(text)

print("SUCCESS")
