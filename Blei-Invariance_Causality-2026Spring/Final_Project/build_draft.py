import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
source_path = os.path.join(ROOT, "Thesis_Draft", "GSAPP_Final_Submission", "Lewis_Daniel_GSAPPUP2026_Thesis.tex")
target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hardesty-lewis_final_project.tex")

with open(source_path, "r", encoding="utf-8") as f:
    text = f.read()

# Extract abstract
abstract_match = re.search(r"\\begin{abstract}(.*?)\\end{abstract}", text, re.DOTALL)
abstract = abstract_match.group(1).strip() if abstract_match else ""

# Extract Literature review
lit_review_match = re.search(
    r"\\section{Literature Review}(.*?)(?=\\section{Methodology Overview})", text, re.DOTALL
)
lit_review = lit_review_match.group(1).strip() if lit_review_match else ""

# Extract Methods (Methodology Overview -> Filing-Date Primary Results)
methods_match = re.search(
    r"\\section{Methodology Overview}(.*?)(?=\\section{Filing-Date Primary Results})", text, re.DOTALL
)
methods = methods_match.group(1).strip() if methods_match else ""

# Extract Results (Filing-Date Primary Results -> Identification Strategy)
results_match = re.search(
    r"\\section{Filing-Date Primary Results}(.*?)(?=\\section{Identification Strategy})", text, re.DOTALL
)
results = results_match.group(1).strip() if results_match else ""


def fix_includes(content):
    content = re.sub(
        r"\\input{Tables/(.*?)}",
        r"\\input{../../Thesis_Draft/GSAPP_Final_Submission/Tables/\1}",
        content,
    )
    content = re.sub(
        r"\\input{(summary_stats_table.tex)}",
        r"\\input{../../Thesis_Draft/GSAPP_Final_Submission/Tables/summary_stats_table.tex}",
        content,
    )
    content = re.sub(
        r"\\input{Sections/planning_context}",
        r"\\input{../../Thesis_Draft/GSAPP_Final_Submission/Sections/planning_context}",
        content,
    )
    return content


lit_review = fix_includes(lit_review)
methods = fix_includes(methods)
results = fix_includes(results)

jmlr_template = f"""\\documentclass[twoside,11pt]{{article}}
\\usepackage{{jmlr2e}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{float}}
\\usepackage{{subcaption}}
\\usepackage{{enumitem}}
\\usepackage{{tikz}}
\\usetikzlibrary{{positioning, arrows.meta, fit, backgrounds}}
\\usepackage{{algorithm}}
\\usepackage{{algorithmic}}
\\usepackage{{longtable}}

\\graphicspath{{{{../../Thesis_Draft/GSAPP_Final_Submission/Figures/}}{{../../Analysis/Output/}}{{Figures/}}}}
\\input{{../../Thesis_Draft/GSAPP_Final_Submission/Tables/metrics_config.tex}}

\\jmlrheading{{1}}{{2026}}{{1-30}}{{4/26}}{{--/--}}{{--}}{{Daniel Hardesty Lewis}}
\\ShortHeadings{{Predicting Formal Protest Petitions Against Housing Development}}{{Hardesty Lewis}}
\\firstpageno{{1}}

\\begin{{document}}

\\title{{Predicting Formal Protest Petitions Against Housing Development: Evidence from Austin, Texas}}

\\author{{\\name Daniel Hardesty Lewis \\email daniel.lewis@columbia.edu \\\\
       \\addr Graduate School of Architecture, Planning and Preservation\\\\
       Columbia University\\\\
       New York, NY 10027, USA}}

\\editor{{David Blei}}

\\maketitle

\\begin{{abstract}}%
{abstract}
\\end{{abstract}}

\\begin{{keywords}}
  Causal Inference, Invariant Causal Prediction, Urban Planning, Housing Policy, Predictive Modeling
\\end{{keywords}}

\\section{{Related Research}}
{lit_review}

\\input{{../../Thesis_Draft/GSAPP_Final_Submission/Sections/planning_context}}

\\section{{Methods}}
{methods}

\\section{{Evaluation}}
{results}

\\section{{Causal Methodology Extensions}}
\\input{{../../Thesis_Draft/GSAPP_Final_Submission/Sections/blei_methodology.tex}}

\\acknowledgements{{
I would like to express my deepest gratitude to my thesis advisor, Dory Kornfeld, for her guidance, and Professor David Blei for insights on causal invariance and Bayesian Optimization.
}}

\\bibliography{{../../Thesis_Draft/GSAPP_Final_Submission/references}}

\\end{{document}}
"""

with open(target_path, "w", encoding="utf-8") as f:
    f.write(jmlr_template)

print(f"Generated {os.path.basename(target_path)}")
