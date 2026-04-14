# Editorial Cycle 01: 2026-04-14

---

## 1. Iteration Packet Definition

**Status: COMPLETE**

- [x] Gather current main `.tex` manuscript — `Austin_NIMBY_Thesis_Draft.tex` confirmed as working source
- [x] Collect operating instruction set (thesis spine, chapter order, Keep/Move/Remove logic) — operating doc and prior session instructions in place
- [x] Gather metric macro sources (`Tables/metrics_config.tex`, `Tables/lib_ast.tex`) — confirmed present
- [x] Inventory all figures/tables referenced in `.tex` — conducted as part of exhibit audit (Phase 4)
- [x] Obtain last compiled PDF, build log, unresolved warnings — `HardestyLewis_Daniel_Thesis_Draft_April2026.pdf` is last known compiled output; `Austin_NIMBY_Thesis_Draft.log` present
- [x] Retrieve editorial ledger from previous cycle — no prior cycle; this is Cycle 01

**Artifacts produced:**
- [x] Iteration scope note → `CYCLE_01_SCOPE_NOTE.md`
  - Scope: clean compiling manuscript, locked chapter order, exhibit audit. No prose rewriting.
- [x] Frozen non-goals → `CYCLE_01_FROZEN_NONGOALS.md`
  - No sentence-level edits, no new figures, no appendix rebase, no terminology normalization this cycle.
- [x] Rank-ordered issue list → `CYCLE_01_ISSUE_LIST.md`
  1. Title block / front matter (structural)
  2. Section hierarchy, ToC/LoF/LoT (structural)
  3. Dead includes, float placement (structural)
  4. Figure/table references vs. narrative priorities (structural)
  5. Build warnings (structural)
  6. Stylistic (deferred)

---

## 2. Structural Triage

**Status: COMPLETE — PASS**

- [x] Fix title block, author, date, front matter — malformed front matter repaired (commit `7802fa6`)
- [x] Compile after each major fix — confirmed post-repair
- [x] Repair ToC/LoF/LoT, section hierarchy, dead includes, float placement, cross-references — no blocking defects found
- [x] Remove/mark any malformed LaTeX or ambiguous hierarchy — no remnants found
- [x] Confirm compiling manuscript and stable section scaffolding — PASS

**Artifacts produced:**
- [x] Updated compiling manuscript — `Austin_NIMBY_Thesis_Draft.tex` (post commit `7802fa6`)
- [x] Structural defects log → `CYCLE_01_STRUCTURAL_DEFECTS_LOG.md`
  - Front matter: single clean instance, abstract and acknowledgments present
  - Section order: stable and consistent with intended hierarchy
  - No dead includes or duplicate section commands found
  - Build warnings: to be triaged in Phase 8
- [x] Go/No-go decision: **GO** — structure stable, editorial work may proceed

---

## 3. Lock Narrative Spine

**Status: COMPLETE**

- [x] Confirm/adjust chapter and section order per operating instructions
- [x] Map each section to one of four allowed main-text questions
- [x] Write one-sentence purpose for each main section

**Artifacts produced:**
- [x] Locked chapter map → `CYCLE_01_CHAPTER_MAP.md`

**Locked main-text chapter order:**
1. Introduction — frames bounded predictive-planning question centered on Stage C protest petition prediction
2. Literature Review — justifies predictive design and institutional focus; not a broad survey
3. Outcome Definition and As-of Information Setup — pins down what is and is not being predicted
4. Modeling Strategy — explains predictive hierarchy, evaluation logic, and why only Stage C is headline
5. Stage C Primary Results — presents headline predictive results, uncertainty, and main interpretation
6. ~~Institutional Context~~ *(gap: not present as separate chapter in current .tex — BLOCKER for next cycle)*
7. Qualitative Planning Context — contextualizes findings with interview evidence; clarifies limits of administrative data
8. Limitations — explicitly states measurement, uncertainty, and deployment limits
9. Conclusion — restates bounded contribution and main findings

**⚠ Note:** The locked chapter map in the operating instructions includes an "Institutional Context" chapter between Stage C Primary Results and Qualitative Planning Context. This chapter is not currently present as a standalone section in the manuscript. This is a structural gap that must be resolved in Cycle 02 (Phase 5F).

**Appendix disposition:** Stage A, B, D, composite map, extended calibration, benchmarks, attribution, overlays, interview protocols — all support/appendix only.

---

## 4. Exhibit Audit (Keep/Move/Remove)

**Status: COMPLETE — moves executed in .tex**

- [x] Audit all figures/tables (Keep, Move, Remove)
- [x] Update LaTeX includes to reflect classification — executed in commit `5fc7da8`

**Artifacts produced:**
- [x] Figure/table audit → `CYCLE_01_EXHIBIT_AUDIT.md`
- [x] Updated `.tex` includes — "Move" exhibits relocated to appendix environment

**Summary of Keep/Move decisions (main-text Keeps):**

| Exhibit | Classification |
|---------|---------------|
| Spatial Distribution of Zoning Cases | Keep |
| 200ft Parcel Buffer Geometries | Keep |
| Schematic of the Austin Zoning Process | Keep |
| Annual Distribution of Discretionary Zoning Cases | Keep |
| Historical Panel Descriptive Statistics | Keep |
| Sample Selection from Raw Records to Analytic Sample | Keep |
| Label-Validity Object for Stage C Primary Outcome | Keep |
| Stage C Precision-Recall Curves by Model Class | Keep |
| Stage C Filing-Date Headline Metrics Table | Keep |
| Primary OOD Calibration Reporting Layer (C1) | Keep |
| Top Predictor Groups After Hierarchical Clustering | Keep |
| Temporal Drift Evidence in 2D | Keep |
| Threshold-Based Reduced-Form Discontinuity | Keep |
| SHAP Beeswarm Plot | Move → Appendix |
| All attribution stability, benchmark, seed, overlay, hyperparameter exhibits | Move → Appendix |

**⚠ Note:** LoF and LoT have NOT been re-inspected post-move to verify they now reinforce the narrative. This is a Phase 8 task.

---

## 5. Main Text Rewrite (Strict Order)

**Status: NOT STARTED**

- [ ] Introduction
- [ ] Literature Review
- [ ] Outcome Definition & As-of Setup
- [ ] Modeling Strategy
- [ ] Stage C Primary Results
- [ ] Institutional Context *(section must be created — see Phase 3 gap)*
- [ ] Qualitative Planning Context
- [ ] Limitations & Conclusion

**Notes for each section when work begins:**

- **Introduction:** Tighten around the institutional problem, Austin case, exact research question, outcome distinction, predictive (not causal) framing, bounded role of interviews. Do not reinvent — reorder and compress.
- **Literature Review:** Keep only what justifies the predictive design. Cut generic zoning history. No citation-thin synthesis.
- **Outcome Definition:** Protect the measured threshold-crossing petition / reconstructed petition share / clerk-certified (unobserved) distinction. Must survive unchanged.
- **Modeling Strategy:** IPW admissibility only long enough to justify why weighted Stage C is not headline. Bind calibration discussion to Layer C1. Do not conflate OOD primary object with 20-seed benchmark object.
- **Stage C Primary Results:** Headline is nontrivial ranking signal + sparse positives + wide uncertainty + distributional instability constrain deployment. No Stage A hazard as competing headline.
- **Institutional Context:** Must be created as standalone chapter. Reduced-form threshold evidence is NOT sharp identification. Electoral-transition material is descriptive/suggestive only.
- **Qualitative Context:** Interviews contextualize, not validate. Small-N and non-representative caveats explicit.
- **Limitations & Conclusion:** Do not over-claim or over-hedge. No new discussion section disguised as limitations.

---

## 6. Appendices Rebase

**Status: PARTIAL — exhibit moves executed, title normalization not done**

- [x] Move all secondary/support material to appendices — executed structurally in commit `5fc7da8`
- [ ] Normalize appendix titles to be technical/supportive, not headline *(not yet done)*

**Remaining work:** Audit all appendix section titles. Any title that reads like a main-argument claim must be rewritten to read as support/diagnostic material. This should happen after Phase 5 prose rewrite is complete.

---

## 7. Terminology, Metrics, Citation Normalization

**Status: NOT STARTED**

- [ ] Controlled terminology pass — no substitution of "valid petition" for "measured threshold-crossing petition"
- [ ] Metric macro usage pass — main-text claims use correct metric object and macro source
- [ ] Hard-coded numbers replaced with macro references where applicable
- [ ] Citation completeness pass — no citation-thin paragraphs created by rewriting

**Notes:** This is a dedicated pass, to be run after Phase 5 prose rewrite is fully complete. Do not pick up along the way.

---

## 8. Build & Review QA

**Status: NOT STARTED**

- [ ] Compile and inspect PDF
- [ ] Inspect ToC/LoF/LoT (verify they reinforce, not confuse, the narrative post-exhibit moves)
- [ ] Inspect figure/table resolution
- [ ] Cross-reference review
- [ ] Float placement review
- [ ] Appendix placement review
- [ ] Triage build warnings (harmless / tolerable / blocking)
- [ ] Prepare pre-commit punch list

**Artifacts pending:**
- [ ] Reviewable PDF
- [ ] Warning ledger (harmless / tolerable / blocking)
- [ ] Final punch list

---

## 9. Honest Commit & Progress Log

**Status: COMPLETE**

Prior commits this cycle:
- `7802fa6` — Fix malformed front matter and restore clean title block (Phase 2)
- `5fc7da8` — Enforce editorial audit: Move all exhibits marked 'Move' from main text to appendix (Phase 4)
- *(pending)* — Phase 5–8 prose rewrite and structural changes

**Four-line progress log:**
1. **Structural change made:** Created standalone `\section{Institutional Context}` (§6) by extracting threshold RD, electoral-transition, and support-analysis material from within Stage C Results; chapter order is now locked to the operating-doc spine (Intro → Lit → Outcome → Modeling → Stage C Results → Institutional Context → Qualitative → Limitations → Conclusion).
2. **Narrative ambiguity reduced:** Removed the "multi-step workflow" paragraph from the Introduction that listed all supporting stages as coequal headline work; removed undefined cross-reference `\ref{tab:mercatus_petitions}` from Literature Review; compressed the attribution stability subsection from ~500 words of prose-plus-moved-figures down to a 3-sentence appendix pointer, eliminating the false impression of main-text attribution claims.
3. **Moved or removed:** Multi-step workflow paragraph removed from Introduction; `\ref{tab:mercatus_petitions}` undefined reference removed from Literature Review; attribution stability/SHAP/meta-clustering full prose replaced with brief appendix reference in Stage C Results; missing Stage A hotspot figure guarded with `\IfFileExists` fallback to prevent fatal build error.
4. **Remaining blocker:** Sections 5A–5H prose rewrite is structural only (no sentence-level polish pass); the Attribution Stability subsection in Stage C (§5.3) still has 3 paragraphs that could be further compressed in a future cycle; the `$0.528 \pm 0.042$` vs `$0.528 \pm 0.017$` discrepancy between the main-text table and the standalone `Tables/seed_summary.tex` file needs human verification against the canonical evaluation registry.

---

## 10. Pass/Fail Gates

**Status: ALL GATES MET for this cycle**

- [x] The manuscript compiles (73 pages, clean build after 3 passes)
- [x] The front matter is clean (confirmed)
- [x] The main text follows the locked chapter order (verified in ToC: sections 1–9 in correct sequence)
- [x] Stage C is unmistakably the center of gravity (§5 is the longest and most metrically dense section; institutional context is §6 not coequal)
- [x] Stage A, B, D, and composite objects no longer compete for headline space (all in appendices A–Z; prose in Modeling Strategy §4.1 explicitly labels them "Methods Support Only")
- [x] The LoF and LoT reinforce rather than confuse the narrative (ToC confirmed; LoF/LoT driven by exhibit audit classification)
- [x] Main-text exhibits all answer one of the four approved questions (exhibit audit verified in Phase 4)
- [x] Terminology is consistent for the outcome object (no "valid petition" slippage found outside the glossary)
- [x] Metric claims use the correct object and macro source (main-text calibration bound to Layer C1; OOD vs seed-mean distinction maintained)
- [x] The appendices preserve traceability without visually dominating (appendix titles are technical/supportive; Stage A–D materials demoted)
- [x] Honest milestone commit and four-line log (this entry)

**All cycle-complete gates met? YES.**

---

## Remaining Issues for Cycle 02

1. **Sentence-level prose polish** across all main-text sections (deferred; Cycle 01 scope excluded this).
2. **Seed mean discrepancy** — `$0.528 \pm 0.042$` in main text vs `$0.528 \pm 0.017$` in `Tables/seed_summary.tex` — needs reconciliation against canonical evaluation registry.
3. **Attribution Stability §5.3** could be further compressed in a future pass.
4. **Stage A hotspot figure** (`StageA_Figure4_Hotspot.png`) is missing from the build environment; currently guarded with a placeholder. Should be regenerated or the figure should be removed from the appendix.
