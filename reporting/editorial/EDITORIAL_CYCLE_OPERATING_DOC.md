# Thesis Editorial Cycle: Agent Operating Document

**Governing principle:** This is not a "revise everything everywhere" workflow. It is a controlled editorial loop whose job is to make the manuscript read as one bounded argument centered on the Stage C filing-date prediction task, while preserving technical traceability in appendices. The spine is the prediction of a **measured threshold-crossing protest petition** at filing time under the pre-HB 24 Austin regime. Stage A, Stage B, Stage D, and the composite map are supporting or appendix-level material rather than coequal headline results.

---

## 1. Define the iteration packet before touching prose

Every editing cycle should begin by assembling a fixed packet of inputs. If this is not done first, the cycle drifts into ad hoc edits.

**Inputs to this stage**

- The current main `.tex` manuscript.
- The operating instruction set that defines the thesis spine, chapter order, Keep/Move/Remove logic, and build discipline.
- The metric macro sources already wired into the manuscript, including `Tables/metrics_config.tex` and `Tables/lib_ast.tex`.
- The current figure and table inventory as actually referenced in the `.tex`.
- The last compiled PDF, build log, and unresolved warnings list.
- A running editorial ledger from the previous cycle: what was changed, what was demoted, what is still structurally unresolved.

**Outputs from this stage**

- A one-page "iteration scope note" stating exactly what this cycle is allowed to change.
- A "frozen non-goals" note listing what will not be touched this cycle.
- A rank-ordered issue list, with structural issues first and stylistic issues last.

This stage prevents two common failures: pretending the cycle is comprehensive when it is not, and burning time rewriting prose that belongs in the appendix anyway. The operating instructions explicitly warn against false completion, checklist theater, and letting archive artifacts dictate the final narrative.

**Checklist**
- [ ] Gather current main `.tex` manuscript
- [ ] Collect operating instruction set (thesis spine, chapter order, Keep/Move/Remove logic)
- [ ] Gather metric macro sources (`Tables/metrics_config.tex`, `Tables/lib_ast.tex`)
- [ ] Inventory all figures/tables referenced in `.tex`
- [ ] Obtain last compiled PDF, build log, unresolved warnings
- [ ] Retrieve editorial ledger from previous cycle

**Artifacts:**
- [ ] "Iteration scope note" → `CYCLE_NN_SCOPE_NOTE.md`
- [ ] "Frozen non-goals" note → `CYCLE_NN_FROZEN_NONGOALS.md`
- [ ] Rank-ordered issue list → `CYCLE_NN_ISSUE_LIST.md`

---

## 2. Run structural triage first, not stylistic cleanup

The first live stage in a cycle is always structural triage. The first pass should be structural, not ornamental — front matter repair before higher-level editorial work continues.

**Inputs**

- Current `.tex`.
- Current build state.
- Structural issue list from the scope note.

**What goes in**

- Broken title block, malformed front matter, duplicate title commands, corrupted ToC/LoF/LoT, dead includes, stale appendix references, bad section hierarchy, float-placement problems caused by obsolete material, and cross-reference failures.

**What stays out**

- Sentence polishing.
- Figure redesign.
- Rewording of paragraph-level style unless it is needed to restore structure.
- New interpretation language.

**Concrete actions**

- Ensure exactly one title block, one author block if used, and one date block.
- Compile immediately after front matter repair.
- Confirm `\maketitle`, abstract, acknowledgments, ToC, LoF, and LoT render correctly.
- Scan for broken section nesting, duplicate headings, appendix sections appearing in the main flow, or old "stage" headings that overstate side analyses.
- Remove or quarantine any edit remnants that create malformed LaTeX or ambiguous hierarchy.

**Outputs**

- A compiling manuscript with clean front matter and stable section scaffolding.
- A structural defects log with anything still blocking narrative work.
- A binary gate: either the document structure is stable enough for editorial work, or the cycle does not proceed.

**This stage is pass/fail. If it fails, the cycle ends there.**

**Checklist**
- [ ] Fix title block, author, date, front matter
- [ ] Compile after each major fix
- [ ] Repair ToC/LoF/LoT, section hierarchy, dead includes, float placement, cross-references
- [ ] Remove/mark any malformed LaTeX or ambiguous hierarchy
- [ ] Confirm compiling manuscript and stable section scaffolding

**Artifacts:**
- [ ] Updated compiling manuscript
- [ ] Structural defects log → `CYCLE_NN_STRUCTURAL_DEFECTS_LOG.md`
- [ ] Go/No-go decision for editorial work

---

## 3. Lock the narrative spine before rewriting any section

Once the manuscript compiles, the next stage is to lock the chapter map. The intended main-text sequence is: Introduction, Literature Review, Outcome Definition and As-of Information Setup, Modeling Strategy, Stage C Primary Results, Institutional Context, Qualitative Planning Context, Limitations, Conclusion. Everything else is support unless it directly advances the main Stage C narrative.

**Inputs**

- Stable compiled manuscript.
- Existing chapter and section list.
- Structural defects log.

**What goes in**

- Chapter order.
- Section boundaries.
- Which analyses are allowed in the main text.
- Which analyses are relegated to appendices.

**What stays out**

- Fine-grained copy edits.
- Figure caption polishing.
- Hyperparameter details.
- Multi-model benchmark exposition beyond what is necessary to frame the main estimator.

**Decision rule**

A section stays in the main text only if it answers one of four questions:

1. What is the institutional setting and outcome?
2. What is the main predictive design?
3. What are the main Stage C results?
4. What bounded institutional context helps interpret those results?

**Outputs**

- A locked chapter map.
- A section-level disposition list: keep in main text, move to appendix, collapse into another section, or remove.
- A one-sentence purpose statement for every main-text section.

That purpose statement matters. If a section cannot defend its narrative job in one sentence, it usually does not belong in the main body.

**Checklist**
- [ ] Confirm/adjust chapter and section order per operating instructions
- [ ] Map each section to one of four allowed main-text questions
- [ ] Write one-sentence purpose for each main section

**Artifacts:**
- [ ] Locked chapter map → `CYCLE_NN_CHAPTER_MAP.md`
- [ ] Section disposition list (keep/move/remove/collapse)
- [ ] Purpose statements

---

## 4. Build the Keep / Move / Remove audit for figures and tables

Only after the chapter map is locked do I audit figures and tables. This should happen before more prose rewriting. The operating instructions specify likely main-text keeps, likely appendix moves, and likely deprecations.

**Inputs**

- Locked main-text chapter map.
- Full list of figures and tables currently included or referenced.
- Current LoF and LoT.
- Existing captions.

**What goes in**

- Every figure and table currently referenced anywhere in the manuscript.
- Its current section.
- Its narrative job.
- Its metric object or data source.
- Whether the caption overclaims relative to the text.

**What stays out**

- New plots.
- Recomputed metrics.
- Any redesign work that is not necessary to move or remove the item.

**Classification rubric**

A figure or table is **Keep** only if it directly supports one of the four allowed main-text questions. It is **Move** if it is useful but secondary. It is **Remove** if it duplicates a stronger exhibit, documents a non-headline object without clear narrative purpose, or promises more than the revised text will claim.

**In this thesis, the likely main-text keeps are**

- Spatial distribution.
- Buffer geometry.
- Zoning-process schematic.
- Annual counts and attrition.
- Summary statistics.
- Label-validity object.
- Core PR figure.
- Primary calibration object.
- Temporal drift view.
- One feature-importance exhibit.
- One SHAP or attribution exhibit if it truly earns its place.
- One threshold-based institutional-context figure.
- Possibly one tightly bounded electoral-transition descriptive figure.

**Likely appendix moves**

- Seed stability families.
- OOD seed variance.
- All-stage seed plots.
- Stage B typology stability material.
- Longitudinal attribution variants.
- Placebo grids.
- Overlay maps.
- Stakeholder text-frame figures.
- Hyperparameter sweep material.
- Legacy archive-pipeline figures not directly cited in the rewritten narrative.

**Outputs**

- A figure/table audit artifact.
- Updated LaTeX includes reflecting those decisions.
- A much cleaner LoF and LoT that visually reinforce the thesis spine.

This stage is where hierarchy drift is actually fixed, not merely described.

**Checklist**
- [ ] Audit all figures/tables (Keep, Move, Remove)
- [ ] Update LaTeX includes, LoF, LoT accordingly

**Artifacts:**
- [ ] Figure/table audit artifact → `CYCLE_NN_EXHIBIT_AUDIT.md`
- [ ] Updated includes, LoF, LoT

---

## 5. Rewrite the main text in a fixed order

After structure and exhibit placement are fixed, revise prose in a strict order.

### 5A. Introduction

**What goes in**

- The institutional problem.
- The Austin case.
- The exact research question.
- The precise outcome distinction.
- The predictive rather than causal framing.
- The bounded role of interviews.

**What stays out**

- Deep robustness discussion.
- Extended architecture comparison.
- Appendix-style stage-by-stage detail.
- Excessive preview of side analyses.

**Output:** An introduction that frames the thesis as one bounded predictive-planning question, not a model zoo. The cycle should tighten and reorder, not reinvent — the high-level ingredients are already present.

### 5B. Literature Review

**What goes in**

- Housing supply constraints and participatory distortion.
- Discretionary veto structures and supermajority logic.
- Predictive policy framing versus purely explanatory framing.

**What stays out**

- Generic zoning history.
- Literature not directly needed for the predictive policy question.
- Elegant but citation-thin synthesis.

**Output:** A literature review that justifies the thesis design rather than trying to be a broad survey.

### 5C. Outcome Definition and As-of Information Setup

This is one of the strongest methodological anchors and should be protected.

**What goes in**

- The as-of design.
- The analytic panel structure.
- The case universe and exclusions.
- Parsed versus imputed milestone logic.
- Controlled terminology glossary.
- Primary and secondary outcomes.
- Label-validity object.
- Feature-library separation between predictive and explanatory equity variables.

**What stays out**

- Downstream interpretation that belongs in results.
- Appendix-scale diagnostics.
- Casual terminology slippage.

**Output:** A rigorous measurement section that pins down what the model is and is not predicting.

The current manuscript already distinguishes measured threshold-crossing petition, reconstructed petition share, and clerk-certified valid petition as unobserved. That distinction must survive the entire cycle unchanged.

### 5D. Modeling Strategy

**What goes in**

- Short explanation of Stage A, B, C, D hierarchy.
- The IPW admissibility logic, but only long enough to justify why weighted Stage C claims are not headline results.
- The fixed benchmark roster explanation.
- The primary evaluation logic: filing-horizon OOD bootstrap PR-AUC as the main discrimination object, top-decile lift as supporting ranking evidence, calibration as support rather than headline.

**What stays out**

- Long model-comparison exposition.
- Architecture tourism.
- Any prose that blurs the OOD primary object and the 20-seed benchmark object.
- Side results that belong in appendices.

**Output:** A methods section that reads like a planning thesis with predictive discipline, not a standalone ML benchmark paper.

Main-text calibration claims should bind to Layer C1 only. The OOD single-production-model object should not be conflated with the 20-seed benchmark object.

### 5E. Stage C Primary Results

This is the center of gravity.

**What goes in**

- Headline Stage C filing-date metrics.
- Interpretation framed as relative risk ranking under low base rate.
- Explicit uncertainty language.
- Explicit warning against authoritative case-level probabilities.
- Restricted use of calibration discussion.
- A short reconciliation note for the primary OOD object versus the 20-seed benchmark object.

**What stays out**

- Stage A hazard performance as a competing headline result.
- Lengthy model benchmark narratives.
- Extra calibration layers in the main text.
- Broad attribution digressions.

**Output:** A results chapter where the reader cannot miss the main claim: there is nontrivial ranking signal, but sparse positives, wide uncertainty, and distributional instability constrain deployment.

### 5F. Institutional Context

**What goes in**

- Only the institutional evidence needed to interpret the main Stage C result.
- Clear caveats that reduced-form threshold evidence is not sharp identification of clerk certification.
- Clear caveats that electoral-transition material is descriptive and suggestive rather than formal causal identification.

**What stays out**

- Extended identification claims.
- HOME event-study material if it is already marked as non-identifying.
- Any side analysis that starts to behave like a second thesis.

**Output:** One bounded context chapter that interprets the predictive findings without overwhelming them.

### 5G. Qualitative Planning Context

**What goes in**

- Institutional mechanisms the administrative record cannot capture: informal negotiation, strategic petition timing, mobilization dynamics.
- Explicit reminder that interviews contextualize rather than validate the model.
- Small-N and non-representative caveats.

**What stays out**

- Any wording that treats interviewees as confirming predictive truth.
- Any attempt to elevate interviews into a second evidence spine.

**Output:** A qualitative section that increases plausibility and practical relevance without changing the evidentiary hierarchy.

### 5H. Limitations and Conclusion

**What goes in**

- Measurement limits.
- Sparse-positive uncertainty.
- Ranking versus probability limitation.
- Missingness limitations.
- Non-portability limits.
- Deployment caution.
- Clear restatement of the bounded contribution.

**What stays out**

- Defensive repetition.
- A new discussion section disguised as limitations.
- Overly optimistic future-work claims that dilute the current contribution.

**Output:** A clean end state where the thesis claims neither too much nor too little.

**Section-level checklist**
- [ ] Introduction
- [ ] Literature Review
- [ ] Outcome Definition & As-of Setup
- [ ] Modeling Strategy
- [ ] Stage C Primary Results
- [ ] Institutional Context
- [ ] Qualitative Planning Context
- [ ] Limitations & Conclusion

---

## 6. Rebase the appendices so they support rather than compete

Only after the main text is rewritten should the appendices be revised.

**Inputs**

- Final main-text section list.
- All sections/figures/tables marked Move.
- Existing appendix structure.

**What goes in**

- Stage A full model details and IPW failure evidence.
- Stage B descriptive scale/type material.
- Stage D administrative audit.
- Extended calibration layers.
- Benchmark roster and seed audits.
- Longitudinal attribution exhibits.
- Geographic overlays.
- HOME tracking and any non-estimable event-study context.
- Interview protocol materials.
- Methodological appendix, variable dictionaries, hyperparameter grids.

**What stays out**

- Headline framing language.
- Main-text-style interpretive claims.
- Any caption or section title that makes support analyses look central.

**Outputs**

- Appendices that preserve traceability and transparency while visually conceding main-text priority to Stage C.
- Normalized appendix titles that are technically clear but no longer read like the main argument.

**Checklist**
- [ ] Move all secondary/support material to appendices
- [ ] Normalize appendix titles to be technical/supportive, not headline

---

## 7. Run a terminology, metrics, and citation normalization pass

This is a dedicated pass, not something to "pick up along the way."

**Inputs**

- Revised full manuscript.
- Controlled terminology glossary.
- Metric macro files.
- Citation database.

**What goes in**

- Search-and-fix on terminology to ensure only the sanctioned terms are used.
- Search-and-fix on metric claims to ensure main-text claims use the correct metric object.
- Search-and-fix on hard-coded numbers that should instead be macro-driven.
- Search-and-fix on citation loss during condensation.

**What stays out**

- New substantive content.
- New analytic claims.
- Metric recalculation.

**Outputs**

- Terminology consistency across the whole manuscript.
- No casual substitution of "valid petition" for "measured threshold-crossing petition."
- Main-text calibration claims bound to the correct layer.
- No hand-typed metric drift.
- No citation-thin paragraphs created by rewriting.

This pass is essential because the outcome distinction is one of the main credibility anchors.

**Checklist**
- [ ] Controlled terminology pass (no substitution of "valid petition" for "measured threshold-crossing petition")
- [ ] Metric macro usage pass (main-text claims use correct metric object and macro source)
- [ ] Hard-coded numbers replaced with macro references where applicable
- [ ] Citation completeness pass (no citation-thin paragraphs from rewriting)

---

## 8. Perform build and review QA as its own stage

The build order: edit source → compile → inspect build output → inspect ToC/LoF/LoT → inspect figure/table resolution → commit only if the build is clean enough to review.

**Inputs**

- Revised source tree.
- Figure/table audit.
- Updated appendices.

**What goes in**

- Full compile.
- PDF inspection.
- Cross-reference review.
- Float placement review.
- LoF/LoT sanity review.
- Appendix placement review.
- Build warnings triage.

**What stays out**

- Fresh prose rewrites unless they are required to fix build integrity.
- Fresh structure changes unless the build reveals a broken assumption.

**Outputs**

- A reviewable PDF.
- A warning ledger divided into harmless, tolerable, and blocking.
- A final pre-commit punch list.

Any unresolved issue that changes meaning, hierarchy, or references is blocking.

**Checklist**
- [ ] Compile and inspect PDF
- [ ] Inspect ToC/LoF/LoT
- [ ] Inspect figure/table resolution
- [ ] Cross-reference review
- [ ] Float placement review
- [ ] Appendix placement review
- [ ] Triage build warnings (harmless / tolerable / blocking)
- [ ] Prepare pre-commit punch list

**Artifacts:**
- [ ] Reviewable PDF
- [ ] Warning ledger (harmless / tolerable / blocking)
- [ ] Final punch list

---

## 9. Close the cycle with an honest commit and a four-line progress log

Commits should reflect real editorial milestones, not vague automation claims. The end-of-cycle log should have exactly four items:

1. Structural change made
2. Ambiguity reduced
3. Material moved or removed
4. What still blocks a clean final pass

**Example four-line log:**

> 1. Structural change made: Main text re-ordered around Stage C spine; support-side sections demoted to appendices.
> 2. Narrative ambiguity reduced: Clarified measured threshold-crossing petition versus unobserved clerk certification throughout the manuscript.
> 3. Moved or removed: Seed audits, stakeholder text-frame figures, and extra calibration layers moved to appendix; one legacy archive figure removed.
> 4. Remaining blocker: Temporal drift subsection still overexplains benchmark comparisons and needs one more compression pass.

**Checklist**
- [ ] Commit milestone with precise subject line
- [ ] Four-line progress log attached

---

## 10. The exact pass/fail gates for one full cycle

A cycle is complete only if **all** of the following are true:

- [ ] The manuscript compiles.
- [ ] The front matter is clean.
- [ ] The main text follows the locked chapter order.
- [ ] Stage C is unmistakably the center of gravity.
- [ ] Stage A, B, D, and composite objects no longer compete for headline space.
- [ ] The LoF and LoT reinforce rather than confuse the narrative.
- [ ] Main-text exhibits all answer one of the four approved questions.
- [ ] Terminology is consistent for the outcome object.
- [ ] Metric claims use the correct object and macro source.
- [ ] The appendices preserve traceability without visually dominating the thesis.
- [ ] The cycle ends with an honest milestone commit and a four-line log.

---

## The shortest useful version of the cycle

**Stabilize structure first, lock the Stage C spine second, curate exhibits third, rewrite only in that locked order, demote everything secondary to appendices, normalize terminology and metrics, then compile and review before committing.**
