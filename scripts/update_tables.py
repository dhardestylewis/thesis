import re

filepath = r'Thesis_Draft\Draft_v1\Austin_NIMBY_Thesis_Draft.tex'

with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

table_replacements = {
    r'\\caption\{Valid Petition Incidence by Proposed Land Use in Austin \(Mercatus Center\)\}': r'\\caption{\\textbf{Context:} Valid Petition Incidence by Proposed Land Use in Austin (Mercatus Center)}',
    r'\\caption\{Annual Distribution of Discretionary Zoning Cases \(2007 - 2024\)\}': r'\\caption{\\textbf{Context:} Annual Distribution of Discretionary Zoning Cases (2007 - 2024)}',
    r'\\caption\{Historical Panel Descriptive Statistics\}': r'\\caption{\\textbf{Context:} Historical Panel Descriptive Statistics}',
    r'\\caption\{Sample Filtration from Raw Records to Analytic Sample\}': r'\\caption{\\textbf{Pipeline Engineering:} Sample Filtration from Raw Records to Analytic Sample}',
    r'\\caption\{Development Occurrence: Nested Targets\}': r'\\caption{\\textbf{Stage A:} Development Occurrence Nested Targets}',
    r'\\caption\{Development Occurrence: Hazard Model Performance\}': r'\\caption{\\textbf{Stage A:} Development Occurrence Hazard Model Performance}'
}

for old, new in table_replacements.items():
    text = re.sub(old, new, text)

# Inject ACE reporting into text
text = text.replace(r'standard Expected Calibration Error (ECE), this pipeline reports the absolute Brier Score.', 
                    r'standard Expected Calibration Error (ECE), this pipeline reports the absolute Brier Score and Adaptive Calibration Error (ACE = \metricACE{}).')
                    
text = text.replace(r'Brier Scores operate as a critical robust alternative because standard ECE structurally degrades',
                    r'Brier Scores \cite{hegre2019views} and Adaptive Calibration (which evaluates equal-mass rather than equal-width bins) operate as critical robust alternatives because standard ECE structurally degrades')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print("List of Tables structurally tagged and ACE terminology injected.")
