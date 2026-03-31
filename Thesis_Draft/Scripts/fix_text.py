import re
import os

filepath = 'Austin_NIMBY_Thesis_Draft.tex'

with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Strip colloquial quotes
terms = [
    "regulatory tax", "homevoters", "politics of exclusion", "scale-dependent preferences", 
    "neighborhood defenders", "seriatim", "citywide bargains", "proposed comprehensive zoning change",
    "digital poorhouse", "regularities", "Weapons of Math Destruction", "veneer of objectivity",
    "traps", "fair", "supportive", "determinative", "dirty", "oppositional"
]
for term in terms:
    text = text.replace(f"``{term}''", term)

# Also fix the weird single-quote ones if they exist
text = text.replace("`regulatory tax'", "regulatory tax")

# 2. Add ToC and Acknowledgments block if not already there
toc_block = r'''\end{abstract}

\newpage
\section*{Acknowledgments}
[Acknowledgments go here. Typically, this is where you thank your thesis advisor, reviewers, mentors, colleagues, and any institutions that provided support or funding.]

\newpage
\tableofcontents
\listoffigures
\listoftables
\newpage
'''
if r'\tableofcontents' not in text:
    text = text.replace(r'\end{abstract}', toc_block)

# 3. Restructure dashes safely (to avoid breaking TikZ code `(--)`)
replacements = {
    "describe -- the fuzzy": "describe. The fuzzy",
    "describe---the fuzzy": "describe. The fuzzy",
    "threshold -- the key lever available to homevoters -- causally": "threshold, the key lever available to homevoters, causally",
    "threshold---the key lever available to homevoters---causally": "threshold, the key lever available to homevoters, causally",
    "preferences -- residents": "preferences: residents",
    "preferences---residents": "preferences: residents",
    "study -- a legally": "study, a legally",
    "study---a legally": "study, a legally",
    "predictions -- and Texas": "predictions, while Texas",
    "predictions---and Texas": "predictions, while Texas",
    "change -- uniform citywide changes allowing more residential development, new zoning codes/maps, or overlay districts along major roadways and transit corridors -- and exempts": "change (uniform citywide changes allowing more residential development, new zoning codes/maps, or overlay districts along major roadways and transit corridors) and exempts",
    "change---uniform citywide changes allowing more residential development, new zoning codes/maps, or overlay districts along major roadways and transit corridors---and exempts": "change (uniform citywide changes allowing more residential development, new zoning codes/maps, or overlay districts along major roadways and transit corridors) and exempts",
    "mechanism -- both": "mechanism, as both involve",
    "mechanism---both": "mechanism, as both involve",
    "problem -- prediction": "problem of prediction",
    "problem---prediction": "problem of prediction",
    "problem -- planners": "problem; planners",
    "problem---planners": "problem; planners",
    "decisions -- an application": "decisions, an application",
    "decisions---an application": "decisions, an application",
    "decisions -- redlining, racial covenants, exclusionary zoning -- meaning": "decisions such as redlining, racial covenants, and exclusionary zoning, meaning",
    "decisions---redlining, racial covenants, exclusionary zoning---meaning": "decisions such as redlining, racial covenants, and exclusionary zoning, meaning",
    "direct -- dirty": "direct: dirty",
    "direct---dirty": "direct: dirty",
    "zoning -- the exact domain": "zoning, the exact domain",
    "zoning---the exact domain": "zoning, the exact domain",
    "threshold -- important for": "threshold, which is important for",
    "threshold---important for": "threshold, which is important for",
    "controls -- a problematic": "controls, a problematic",
    "controls---a problematic": "controls, a problematic",
    "negative -- meaning regression": "negative. This means regression",
    "negative---meaning regression": "negative. This means regression",
    "predictions -- reinforcing": "predictions, reinforcing",
    "predictions---reinforcing": "predictions, reinforcing",
    "approach -- diverse evaluators probing for harmful outputs -- adapts": "approach of diverse evaluators probing for harmful outputs adapts",
    "approach---diverse evaluators probing for harmful outputs---adapts": "approach of diverse evaluators probing for harmful outputs adapts",
    "intersections---thereby": "intersections, thereby",
    "intersections -- thereby": "intersections, thereby",
    "dynamic -- decisions shaped": "dynamic of decisions shaped",
    "dynamic---decisions shaped": "dynamic of decisions shaped",
    "observables -- that is, the fact": "observables (i.e., the fact",
    "observables---that is, the fact": "observables (i.e., the fact",
    "lift -- wildly": "lift, wildly",
    "baseline---wildly": "baseline, wildly",
    "baseline -- wildly": "baseline, wildly",
    "over baseline---wildly": "over baseline, wildly",
    "over baseline -- wildly": "over baseline, wildly",
    "2007--2024": "2007 to 2024",
    "H_1--H_3": "H_1 to H_3",
    "2015--2019": "2015 to 2019",
}

for old, new in replacements.items():
    text = text.replace(old, new)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print('Success')
