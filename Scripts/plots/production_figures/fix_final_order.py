import sys
import os

pth = r'Thesis_Draft\Draft_v1\Lewis_2026_NIMBYism_Austin_Thesis.tex'
with open(pth, 'r', encoding='utf-8') as f:
    lines = f.readlines()

blocks = {}
curr = 'MAIN'
blocks[curr] = []

for i, l in enumerate(lines):
    if l.startswith(r'\subsection{Expanded Methods Appendix:'):
        curr = 'TECH_DETAILS'
        blocks[curr] = [l]
    elif l.startswith(r'\subsection{Temporal Drift Threshold Sensitivity}'):
        curr = 'DRIFT'
        blocks[curr] = [l]
    elif l.startswith(r'\subsection{SHAP Beeswarm Exhibit}'):
        curr = 'SHAP'
        blocks[curr] = [l]
    elif l.startswith(r'\section{Institutional Context and Causal Overlays}'):
        # Ignore this generated header
        continue
    elif l.startswith(r'\label{app:institutional_overlays}'):
        # Ignore this generated header
        continue
    elif l.startswith(r'\subsection{Institutional Geographic Overlay Analyses}'):
        curr = 'OVERLAYS'
        blocks[curr] = [l]
    elif l.startswith(r'\section{Policy Event Studies}'):
        curr = 'EVENTS_HEADER'
        blocks[curr] = [l]
    elif l.startswith(r'\subsection{HOME Initiative Event Study}'):
        curr = 'HOME'
        blocks[curr] = [l]
    elif l.startswith(r'\end{document}'):
        curr = 'END'
        blocks[curr] = [l]
    else:
        blocks[curr].append(l)

# Let's cleanly construct the output
out = []
out.extend(blocks['MAIN'])
out.extend(blocks['TECH_DETAILS'])
out.extend(blocks['DRIFT'])
out.extend(blocks['SHAP'])

# Build clean Appendix C
out.append(r'\newpage' + '\n')
out.append(r'\section{Institutional Context and Causal Overlays}' + '\n')
out.append(r'\label{app:institutional_overlays}' + '\n')
out.extend(blocks['OVERLAYS'])

# Build clean Appendix D
out.extend(blocks['EVENTS_HEADER'])
out.extend(blocks['HOME'])
out.extend(blocks['END'])

with open(pth, 'w', encoding='utf-8') as f:
    f.writelines(out)

print("SUCCESS")
