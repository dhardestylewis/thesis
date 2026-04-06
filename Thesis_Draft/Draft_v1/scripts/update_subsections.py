import re

filepath = 'Austin_NIMBY_Thesis_Draft.tex'

with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update the Subsection tags
replacements = {
    r'\\subsection\{Development Hazard Classification\}': r'\\subsection{Stage A: Development Hazard Classification}',
    r'\\subsection\{Valid Petition Risk Pipeline\}': r'\\subsection{Stage C: Valid Petition Risk Pipeline}',
    r'\\subsection\{Expected Petition Probability\}': r'\\subsection{Stage F: Expected Petition Probability}',
    r'\\subsection\{Causal Mechanisms and Identification\}': r'\\subsection{Causal Identification: Spatial Interventions and Institutional Reform}',
    r'\\subsection\{Natural Language Argument Framing\}': r'\\subsection{NLP Framing: Natural Language Argument Extraction}',
    r'between pure structural randomness and deterministic physical observation\.': r'between stochastic behavioral ceilings and deterministic remote sensing classification bounds.'
}

for old, new in replacements.items():
    text = re.sub(old, new, text)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print("Subsections and physical terminology updated.")
