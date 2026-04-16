import sys
import os

pth = r'Thesis_Draft\Draft_v1\Lewis_2026_NIMBYism_Austin_Thesis.tex'
with open(pth, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if l.startswith(r'\subsubsection{Threshold: >'):
        lines[i] = r'\clearpage' + '\n' + l
    elif l.startswith(r'\subsection{SHAP Beeswarm Exhibit}'):
        lines[i] = r'\clearpage' + '\n' + l

# Now to isolate Institutional Geographic Overlay Analyses into a new Section C, and shift Policy Event Studies to D
blocks = []
curr = []
for l in lines:
    if l.startswith(r'\subsection{Institutional Geographic Overlay Analyses}'):
        # Close out the current block
        blocks.append(curr)
        curr = [r'\newpage' + '\n', r'\section{Institutional Context and Causal Overlays}' + '\n', r'\label{app:institutional_overlays}' + '\n']
        curr.append(l)
    elif l.startswith(r'\section{Policy Event Studies}'):
        blocks.append(curr)
        curr = [l]
    else:
        curr.append(l)
blocks.append(curr)

out = []
for b in blocks:
    out.extend(b)

with open(pth, 'w', encoding='utf-8') as f:
    f.writelines(out)

print("SUCCESS")
