# Thesis Editorial Cycle: Agent Operating Document

This document governs the editorial workflow for each thesis editing cycle. Each cycle must:
- Be bounded by a fixed scope and non-goals
- Prioritize structure, then narrative spine, then exhibits, then prose
- Demote all secondary/support material to appendices
- Normalize terminology, metrics, and citations in a dedicated pass
- Compile and review before any commit
- End with a four-line progress log and honest milestone commit

## Editorial Cycle Checklist

### 1. Iteration Packet Definition
- [ ] Gather current main `.tex` manuscript
- [ ] Collect operating instruction set (thesis spine, chapter order, Keep/Move/Remove logic)
- [ ] Gather metric macro sources (`Tables/metrics_config.tex`, `Tables/lib_ast.tex`)
- [ ] Inventory all figures/tables referenced in `.tex`
- [ ] Obtain last compiled PDF, build log, unresolved warnings
- [ ] Retrieve editorial ledger from previous cycle

**Artifacts:**
- [ ] “Iteration scope note” (what will change this cycle)
- [ ] “Frozen non-goals” note (what will NOT change)
- [ ] Rank-ordered issue list (structural → stylistic)

### 2. Structural Triage (Pass/Fail Gate)
- [ ] Fix title block, author, date, front matter
- [ ] Compile after each major fix
- [ ] Repair ToC/LoF/LoT, section hierarchy, dead includes, float placement, cross-references
- [ ] Remove/mark any malformed LaTeX or ambiguous hierarchy
- [ ] Confirm compiling manuscript and stable section scaffolding

**Artifacts:**
- [ ] Updated compiling manuscript
- [ ] Structural defects log
- [ ] Go/No-go decision for editorial work

### 3. Lock Narrative Spine
- [ ] Confirm/adjust chapter and section order per operating instructions
- [ ] Map each section to one of four allowed main-text questions
- [ ] Write one-sentence purpose for each main section

**Artifacts:**
- [ ] Locked chapter map
- [ ] Section disposition list (keep/move/remove/collapse)
- [ ] Purpose statements

### 4. Exhibit Audit (Keep/Move/Remove)
- [ ] Audit all figures/tables (Keep, Move, Remove)
- [ ] Update LaTeX includes, LoF, LoT accordingly

**Artifacts:**
- [ ] Figure/table audit artifact
- [ ] Updated includes, LoF, LoT

### 5. Main Text Rewrite (Strict Order)
For each section, follow “what goes in/stays out” rules:
- [ ] Introduction
- [ ] Literature Review
- [ ] Outcome Definition & As-of Setup
- [ ] Modeling Strategy
- [ ] Stage C Primary Results
- [ ] Institutional Context
- [ ] Qualitative Planning Context
- [ ] Limitations & Conclusion

### 6. Appendices Rebase
- [ ] Move all secondary/support material to appendices
- [ ] Normalize appendix titles to be technical/supportive

### 7. Terminology, Metrics, Citation Normalization
- [ ] Controlled terminology pass
- [ ] Metric macro usage pass
- [ ] Citation completeness pass

### 8. Build & Review QA
- [ ] Compile and inspect PDF, ToC/LoF/LoT, cross-references, float placement, appendices
- [ ] Triage build warnings (harmless/tolerable/blocking)
- [ ] Prepare pre-commit punch list

**Artifacts:**
- [ ] Reviewable PDF
- [ ] Warning ledger
- [ ] Final punch list

### 9. Honest Commit & Progress Log
- [ ] Commit milestone with four-line log:
  1. Structural change made
  2. Ambiguity reduced
  3. Material moved/removed
  4. Remaining blocker

### 10. Pass/Fail Gates (Cycle Complete Only If ALL True)
- [ ] Manuscript compiles
- [ ] Front matter is clean
- [ ] Main text follows locked chapter order
- [ ] Stage C is the center of gravity
- [ ] Stage A/B/D/composite objects do not compete for headline space
- [ ] LoF/LoT reinforce, not confuse, the narrative
- [ ] Main-text exhibits answer one of four approved questions
- [ ] Terminology is consistent for the outcome object
- [ ] Metric claims use correct object and macro source
- [ ] Appendices are supportive, not headline
- [ ] Honest milestone commit and four-line log
