# Editorial Cycle 01: Exhibit Audit (2026-04-14)

## Figure/Table Audit

| Exhibit | Section | Narrative Job | Metric/Data Source | Caption Status | Classification (Keep/Move/Remove) |
|---------|---------|--------------|-------------------|---------------|-------------------------------|

| Spatial Distribution of Zoning Cases (2007-2024) | Introduction | Main-text spatial context | Panel data | Accurate | Keep |
| 200ft Parcel Buffer Geometries | Introduction | Illustrate buffer logic | GIS/Panel | Accurate | Keep |
| Schematic of the Austin Zoning Process | Introduction | Institutional process | Schematic | Accurate | Keep |
| Annual Distribution of Discretionary Zoning Cases (2007--2024) | Outcome Definition | Main-text annual counts | Panel data | Accurate | Keep |
| Historical Panel Descriptive Statistics | Outcome Definition | Main-text summary stats | Panel data | Accurate | Keep |
| Sample Selection from Raw Records to Analytic Sample | Outcome Definition | Show analytic sample construction | Panel data | Accurate | Keep |
| Label-Validity Object for Stage C Primary Outcome | Outcome Definition | Anchor outcome definition | Label audit | Accurate | Keep |
| Stage C Precision-Recall Curves by Model Class | Stage C Primary Results | Headline PR curves | Metrics macro | Accurate | Keep |
| Stage C Filing-Date Headline Metrics (Primary Object + Supporting Diagnostics) | Stage C Primary Results | Headline metrics table | Metrics macro | Accurate | Keep |
| Primary OOD Calibration Reporting Layer (Layer C1) | Stage C Primary Results | Main-text calibration | Metrics macro | Accurate | Keep |
| Top Predictor Groups After Hierarchical Clustering | Stage C Primary Results | Feature importance | Metrics macro | Accurate | Keep |
| SHAP Beeswarm Plot for the Opposition Model | Stage C Primary Results | Attribution (if justified) | Metrics macro | Accurate | Move |
| Temporal Drift Evidence in 2D | Stage C Primary Results | Drift context | Metrics macro | Accurate | Keep |
| Threshold-Based Reduced-Form Discontinuity on Reconstructed Petition Share | Institutional Context | Institutional threshold evidence | Metrics macro | Accurate | Keep |
| Expanding-Window Attribution Stability Test | Appendix | Attribution stability | Metrics macro | Accurate | Move |
| Multi-Horizon Attribution Stability | Appendix | Attribution stability | Metrics macro | Accurate | Move |
| Meta-Attribution Structural Clustering | Appendix | Attribution clustering | Metrics macro | Accurate | Move |
| Placebo Estimates Across Earlier Election Cycles | Appendix | Placebo test | Metrics macro | Accurate | Move |
| HOME Phase 1 Event-Study Coefficients (Reconstructed Petition Outcome) | Appendix | Event-study | Metrics macro | Accurate | Move |
| All-Stage Seed Plots, OOD, Overlays, etc. | Appendix | Diagnostic/support | Metrics macro | Accurate | Move |
| Supplementary Hyperparameter Search Results | Appendix | Hyperparameter sweep | Metrics macro | Accurate | Move |
| Project Type and Scale | Appendix | Descriptive support | Panel data | Accurate | Move |
| Stage B Continuous Error Multi-Seed Stability | Appendix | Typology stability | Metrics macro | Accurate | Move |
| Stage A IPW Diagnostic Summary (Stabilized and Truncated Weights) | Appendix | IPW diagnostics | Metrics macro | Accurate | Move |
| Overlap and Covariate Balance Diagnostics for Stage A IPW | Appendix | IPW diagnostics | Metrics macro | Accurate | Move |
| Stage A Target Definitions | Appendix | Target definitions | Metrics macro | Accurate | Move |
| Development Occurrence Hazard Model Performance | Appendix | Hazard model | Metrics macro | Accurate | Move |
| Summary of Supplementary Quasi-Experimental Estimates for Reconstructed Petition Outcomes | Appendix | Quasi-experimental | Metrics macro | Accurate | Move |
| Temporal Integrity and Leakage Audit: Local vs. Foundation Performance | Appendix | Leakage audit | Metrics macro | Accurate | Move |
| Stage A & Stage C Evaluated at Filing: Raw vs. Calibrated Candidate Architectures | Appendix | Model comparison | Metrics macro | Accurate | Move |
| Multi-Seed Performance Summary (mean ± std across 20 seeds, benchmark roster evaluation) | Appendix | Benchmark audit | Metrics macro | Accurate | Move |
| Algorithmic Feature-Restriction Stability Matrix | Appendix | Feature restriction | Metrics macro | Accurate | Move |

# Instructions
- List every figure and table currently referenced in the manuscript.
- For each, specify its section, narrative job, metric/data source, and whether the caption overclaims.
- Classify as Keep (main text), Move (appendix), or Remove (deprecate).
- Update LaTeX includes, LoF, and LoT accordingly after audit.

---

**Next:** Update LaTeX includes and LoF/LoT to reflect audit decisions.