import re

file_path = r'c:\Users\dhl\data\thesis\thesis\Thesis_Draft\Draft_v1\Austin_NIMBY_Thesis_Draft.tex'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit(1)

def replace(old, new, count=0):
    global content
    if old not in content and not hasattr(old, 'pattern'):
        print(f"WARNING: Not found - {repr(old)[:100]}")
    elif hasattr(old, 'pattern'):
        content, n = old.subn(new, content, count)
        if n == 0:
            print(f"WARNING: Regex Not found - {old.pattern}")
    else:
        content = content.replace(old, new)

# 1. Graduate School of Model
replace("Graduate School of Model", "Graduate School of Architecture")

# 2. Intro legal sentence
replace(
    "if signatures representing 20\% of the area of lots within 200 feet of the proposed change are collected, the petition triggers a three-fourths supermajority requirement for City Council approval.",
    "if signatures representing 20\% of the relevant adjoining land area were collected and the municipal clerk certified the petition as valid, the petition historically required a three-fourths supermajority for City Council approval. (Effective September 1, 2025, House Bill 24 repealed and replaced this mechanism with a differentiated, multi-tier system under \\S~211.0061.)"
)
replace("Under Texas Local Government Code", "Under the pre-2025 framework of Texas Local Government Code")

# 3. HB 24 Paragraph completely (Line 123)
old_hb24 = "This persistent localized gridlock was ultimately addressed by the Texas legislature in 2025 via House Bill 24, which reformed the statute by stipulating that valid petitions against \\emph{residential upzonings} are modified by HB 24; for certain noncomprehensive residential changes, adjoining-owner protests now require a higher threshold and carry a different vote rule (though petitions against nonresidential rezonings remain fully intact). Consequently, predicting whether a qualifying protest petition will be filed against an application forms the core empirical challenge of this thesis."
new_hb24 = "The Texas Legislature structurally altered this framework in 2025; under House Bill 24, the state established new \\S~211.0061, replacing the uniform 20\% trigger with differentiated standards based on project type and requiring a two-thirds rather than three-fourths council vote for certain residential rezonings (while largely preserving the prior framework for nonresidential cases). Consequently, predicting whether a qualifying formal protest petition will be filed against an application forms the core empirical challenge of this thesis."
replace(old_hb24, new_hb24)

# 4. Flowchart caption and arrow text
replace("node[right, text=red] {\\small Triggers} node[right, yshift=-0.4cm, text=red] {\\small 9-of-11 vote}", 
        "node[right, text=red] {\\small May Require} node[right, yshift=-0.4cm, text=red] {\\small Supermajority}")
replace(
    "A valid protest petition representing 20\% of the area of lots immediately adjoining the proposed change would require, under the applicable legal framework,}",
    "A valid protest petition representing 20\% of the area of lots immediately adjoining the proposed change would require, under the applicable legal framework, a supermajority vote of the City Council for approval, though recusal or absence of council members can alter the exact number of affirmative votes needed.}"
)

# 5. Remove 9-of-11 unless caveated
replace("possessing the votes (9 of 11) to override", 
        "possessing the supermajority (typically 9 votes, though fewer may be required if a council member is recused or absent) to override")

# 6. RD sharp / fuzzy / reduced-form
replace("Sharp Regression Discontinuity", "Threshold-Based Regression Discontinuity")
old_rd_explanation = "This estimate should be interpreted as an Intent-to-Treat (ITT) effect rather than a local average treatment effect. The specification treats the 20\% threshold as sharp---that is, it assumes deterministic compliance whereby crossing the statutory margin automatically triggers the supermajority requirement. In practice, the dataset does not record whether the City Clerk formally certified each threshold breach, so some degree of administrative non-compliance may attenuate the true effect. The sharp RD label is appropriate for the statistical specification but overstates the institutional certainty of compliance."
new_rd_explanation = "This estimate should be interpreted as a reduced-form effect at the threshold. Because the dataset does not record whether the City Clerk formally certified each petition as legally compliant, the analysis relies on an unverified calculated area share. Rather than assuming the exact threshold acts as a deterministic sharp trigger, the specification simply estimates the shift in average processing time associated with crossing the statutory 20\% margin. Consequently, this is a threshold-based reduced-form model where institutional compliance remains unobserved."
replace(old_rd_explanation, new_rd_explanation)

# 7. Demote or cut the HOME section
# Remove references to HOME Phase 1 and 2 from Institutional Calendar
replace("HOME Phase 1\\footnote{Ordinance No. 20231207-001, adopted December 7, 2023; applications accepted February 5, 2024.} and HOME Phase 2\\footnote{Ordinance No. 20240516-006, adopted May 16, 2024; applications accepted August 16, 2024, with Wildland-Urban Interface and Uprooted-designated areas delayed to November 16, 2024.}; and the HB 24 effective date",
        "the HB 24 effective date")
# Remove Panel B about HOME
old_panel_b = r"""\vspace{0.6cm}
\textbf{\small Panel B: HOME Initiative Event Study (Difference-in-Differences)}\par\vspace{0.3cm}
\begin{tikzpicture}[
    node distance=1.5cm and 3cm,
    var/.style={rectangle, draw, fill=white, minimum height=0.8cm, align=center, rounded corners},
    unobs/.style={rectangle, draw, dashed, fill=gray!10, minimum height=0.8cm, align=center, rounded corners},
    arrow/.style={thick,->,>=stealth},
    dashed_arrow/.style={thick,dashed,->,>=stealth}
]
\node[var] (Post) {Time Shock $t$ \\ \footnotesize (Post-HOME Phase 1)};
\node[var, below=1.5cm of Post] (Treated) {Eligibility $E$ \\ \footnotesize (Treated Parcel)};
\node[var, right=2cm of Post, yshift=-1.15cm, very thick] (D) {Treatment $D$ \\ \footnotesize ($E \times t$)};
\node[var, right=2cm of D] (Y) {$Y$ \\ \footnotesize (Council Dissent / No Votes)};
\node[unobs, left=1cm of Treated] (U) {$U$ \\ \footnotesize (Unobserved Factors)};

\draw[arrow] (Post) -- (D);
\draw[arrow] (Treated) -- (D);
\draw[arrow] (D) -- node[above] {$\tau$} (Y);
\draw[arrow] (Post) to[bend left=15] (Y.north west);
\draw[arrow] (Treated) to[bend right=15] (Y.south west);
\draw[dashed_arrow] (U) -- (Treated);
\draw[dashed_arrow] (U) to[bend right=40] (Y.south);
\end{tikzpicture}"""
replace(old_panel_b, "")

replace("Schematic of the Austin Zoning Process]{Austin municipal zoning process. The figure schematically shows where a qualifying protest petition could make a three-fourths council vote relevant under the study-period framework. A valid protest petition representing 20\\% of the area of lots immediately adjoining the proposed change would require, under the applicable legal framework, a supermajority vote of the City Council for approval, though recusal or absence of council members can alter the exact number of affirmative votes needed.}",
        "Schematic of the Austin Zoning Process]{Austin municipal zoning process. The figure schematically shows where a qualifying protest petition could make a three-fourths council vote relevant under the study-period framework. A valid protest petition representing 20\\% of the area of lots immediately adjoining the proposed change would require, under the applicable legal framework, a supermajority vote of the City Council for approval, though recusal or absence of council members can alter the exact number of affirmative votes needed.}")
# adjust line 584
replace("caption[Identification Diagrams for the Threshold and HOME Analyses]{Methodological causal inference graphs mapping the specific institutional mechanisms modeled in the thesis. \\textbf{Panel A} illustrates the Threshold-Based Regression Discontinuity (RD), isolating council delay ($Y$) via the statutory 20\\% protest petition threshold running variable ($R$). \\textbf{Panel B} illustrates the Difference-in-Differences Event Study, isolating council dissent ($Y$) arising from the intersection of the discrete HOME Phase 1 policy time shock ($t$) and parcel eligibility ($E$).}",
        "caption[Identification Diagram for the Threshold Analysis]{Methodological causal inference graph mapping the specific institutional mechanism modeled in the thesis. \\textbf{Panel A} illustrates the Threshold-Based Regression Discontinuity (RD), isolating council delay ($Y$) via the statutory 20\\% protest petition threshold running variable ($R$).}")


# replace the start of 2022 diff in diff
old_diff_start = "The HOME Initiative \\cite{pool2023} models a global legislative shift, but its adoption is citywide and therefore provides no geographic control group. The November 2022 council elections offer a complementary source of variation"
new_diff_start = "The November 2022 council elections offer a source of geographic variation"
replace(old_diff_start, new_diff_start)
replace("\\textit{(3) HOME Staggered DiD}     & protested      & 0.325                     & 0.310  & 0.296                   & 518 \\\\", "")
replace("Designs (3)--(5) are non-informative", "Designs (4)--(5) are non-informative")
replace("HOME Phase~2 yields degenerate estimates due to insufficient post-treatment observations. ", "")

# 8. Remove architecture, tool, etc.
replace("predictive architecture", "predictive methodology")
replace("predictive tool", "predictive model")
replace("predictive system", "predictive model")
replace("policy-forecasting features", "policy features")
replace("policy-forecasting", "policy evaluation")
replace("decision-support benchmarks", "evaluative benchmarks")

# 9. Standardize outcome name
replace("core opposition model", "formal protest petition model")
replace("Opposition Risk PR Curves", "Formal Protest Petition PR Curves")
replace("Opposition Risk:", "Formal Protest Petition:")
replace("opposition risk", "formal protest petition occurrence")
replace("organized opposition", "formal protest petitions")
replace("Opposition Risk", "Formal Protest Petition")
replace("formal protest petition risk", "formal protest petition occurrence")

# 10. Fixes
replace("This thesis ultimately demonstrates", "This thesis demonstrates")
replace("definitively establishes", "establishes")
replace("functions identically for land use: cases exceeding the measured petition threshold appear to face additional voting friction and longer processing times.", 
        "creates institutional friction for land use: cases exceeding the measured petition threshold appear to face additional voting requirements and longer processing times.")
replace("attributable to crossing the statutory margin", "associated with crossing the statutory margin")
replace("must be contingent on", "should consider")
replace("This is a operationalized", "This is an operationalized")
replace("by Model Model", "by Model")
replace("modeling models", "models")
replace("evaluated at filing (evaluated at filing)", "evaluated at the initial filing date")
replace("trailing 3-year the count of", "trailing 3-year count of")

replace("Predicted Hotspot Density vs. Realized Events", "Predicted Geographic Occurrence vs. Realized Events")
replace("Placebo Falsification:", "Placebo evaluation:")
replace("Placebo Falsification", "Placebo Evaluation")

# Add Sample Map Problem Context
sample_text = """\\end{table}

The analytical samples vary by analytical goal. The primary predictive evaluation (Stage C) uses the full sample of 7,074 discretionary geographic cases. In contrast, the causal quasi-experiments are restricted sub-samples designed to isolate specific mechanisms. The 2022 Difference-in-Differences and the Event Study evaluations utilize $N=518$ because they are restricted to active single-member districts immediately surrounding the 2022 election cycle. The Invariant Causal Prediction (ICP) analysis utilizes $N=1,186$ multi-parcel geometric environments to represent cohesive spatial clusters. Finally, the institutional geographic overlay regressions evaluate policy flags aggregated at the council-term level across periods, resulting in $N=20$ macro-observations.

Unlike approaches that require complete-case geometric intersections,"""
replace("\\end{table}\n\nUnlike approaches that require complete-case geometric intersections,", sample_text)

# Few edge cases
replace("The formal protest petition forecast is the core of the thesis", "The Formal Protest Petition forecast is the core of the thesis")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Modifications mapped and saved to {file_path}")
