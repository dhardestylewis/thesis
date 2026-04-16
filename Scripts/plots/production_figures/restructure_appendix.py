import os

pth = r'Thesis_Draft\Draft_v1\Lewis_2026_NIMBYism_Austin_Thesis.tex'
with open(pth, 'r', encoding='utf-8') as f:
    lines = f.readlines()

blocks = {}
curr = 'MAIN'
blocks[curr] = []

for i, l in enumerate(lines):
    if l.startswith(r'\appendix'):
        curr = 'APP_GUIDE'
        blocks[curr] = [l]
    elif l.startswith(r'\section{Quantitative Supplement}'):
        curr = 'QUANT_SUPP_HEADER'
        blocks[curr] = [l]
    elif l.startswith(r'\subsection{SHAP Beeswarm Exhibit}'):
        curr = 'SHAP_BEESWARM'
        blocks[curr] = [l]
    elif l.startswith(r'\section{Qualitative Documentation}'):
        curr = 'QUAL_DOCS'
        blocks[curr] = [l]
    elif l.startswith(r'\section{Expanded Methods and Robustness}'):
        curr = 'EXPANDED_METHODS_HEADER'
        blocks[curr] = [l]
    elif l.startswith(r'\subsection{Expanded Methods Appendix:'):
        curr = 'TECH_DETAILS'
        blocks[curr] = [l]
    elif l.startswith(r'\subsection{Institutional Geographic Overlay'):
        curr = 'OVERLAYS'
        blocks[curr] = [l]
    elif l.startswith(r'\subsection{HOME Initiative Event Study}'):
        curr = 'HOME_EVENT'
        blocks[curr] = [l]
    elif l.startswith(r'\subsection{Temporal Drift Threshold Sensitivity}'):
        curr = 'DRIFT_SCREENS'
        blocks[curr] = [l]
    elif l.startswith(r'\end{document}'):
        curr = 'END'
        blocks[curr] = [l]
    else:
        blocks[curr].append(l)

# Rewrite the APP_GUIDE
app_guide_text = r"""
\newpage
\noindent \textbf{Appendix Guide.} The appendix is organized into three blocks. \textit{Qualitative Documentation} (Appendix A) contains recruitment, IRB, interview, petition, and panel artifacts mapping to Chapter 3. \textit{Quantitative Methods and Robustness} (Appendix B) contains detailed modeling configurations, variable mapping, calibration transparency, and exploratory prediction diagnostics mapping to Chapters 4 and 5. \textit{Policy Event Studies} (Appendix C) contains independent tracking of the HOME Initiative shock. Detailed navigation is provided by the table of contents.
"""
# Keep '\appendix\n'
blocks['APP_GUIDE'] = [r'\appendix' + '\n', app_guide_text]

# Rewrite Quantitative Header
blocks['QUANT_SUPP_HEADER'] = [r'\newpage' + '\n', r'\section{Quantitative Methods and Robustness}' + '\n', r'\label{app:quantitative_supp}' + '\n\n']

# Create Event Studies Header
blocks['EVENTS_HEADER'] = [r'\newpage' + '\n', r'\section{Policy Event Studies}' + '\n', r'\label{app:policy_events}' + '\n\n']

# Let's glue them!
out_lines = []
out_lines.extend(blocks['MAIN'])
out_lines.extend(blocks['APP_GUIDE'])
out_lines.extend(blocks['QUAL_DOCS'])
out_lines.extend(blocks['QUANT_SUPP_HEADER'])
out_lines.extend(blocks['TECH_DETAILS'])
out_lines.extend(blocks['OVERLAYS'])
out_lines.extend(blocks['DRIFT_SCREENS'])
out_lines.extend(blocks['SHAP_BEESWARM'])
out_lines.extend(blocks['EVENTS_HEADER'])
out_lines.extend(blocks['HOME_EVENT'])
out_lines.extend(blocks['END'])

with open(pth, 'w', encoding='utf-8') as f:
    f.writelines(out_lines)

print("SUCCESS")
