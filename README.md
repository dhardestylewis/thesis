# Predicting Zoning Opposition

**Daniel Hardesty Lewis**  
Master's Thesis, Urban Planning  
Columbia University, Graduate School of Architecture, Planning and Preservation

## Abstract

This thesis develops machine learning models and causal inference designs to evaluate the predictability and structural attrition of civic opposition to discretionary zoning. Utilizing a canonical universe of over 7,000 distinct rezoning and zoning map amendment filings in Austin, Texas from 2007 to 2024, the research operationalizes "measured threshold-crossing petitions" as its primary dependent variable. The study is explicitly bounded to the pre-HB24 institutional regime to ensure structural and temporal validity.

## The Auditable Pipeline (Refactored April 2026)

The repository has been refactored into a hierarchical, auditable pipeline centered on the **Stage C (Filing-Date Petition Risk)** prediction task. This architecture enforces a strict separation between data engineering, model training, and empirical auditing.

### Directory Structure

| Directory | Purpose |
|-----------|---------|
| `src/` | **Core Logic**: Modularized code for `labels` (audit-driven), `features` (filing-horizon limited), `models` (benchmarked), and `reporting` (LaTeX synchronization). |
| `scripts/` | **Pipeline Stages**: Numbered execution scripts (`01_build_labels.py` through `10_export_...`) that serve as entry points for the `src` modules. |
| `configs/` | **Task Configuration**: YAML-based definitions for model hyper-parameters, data splits, and feature clusters. |
| `registries/` | **Immutable Outputs**: Parquet and JSON artifacts representing frozen states (case universe, split definitions, and finalized metrics). |
| `Thesis_Draft/` | **Manuscript**: The active LaTeX manuscript, figures, and synchronized macro tables. |
| `Analysis/` | **Legacy Sandbox**: Exploratory backtests, diffusion-as-oversampling experiments, and generative parcel panels (archival/auxiliary). |

## Core Components

1. **Canonical Stage C Task**: A gradient-boosted (CatBoost) and tabular-transformer (TabPFN/ExcelFormer) suite predicting the binary "20% petition threshold" event.
2. **Label Fidelity Audit**: A formal validation layer comparing reconstructed administrative outcomes against a hand-validated subset (~84% agreement).
3. **Meta-Attribution Consensus**: A SUR-based attribution framework (Surrogate SHAP) mapping model gradients to semantic feature clusters to ensure interpretability.
4. **Temporal Baseline Anchors**: Comparison against naive logistic baselines and regional temporal drift metrics.

## Reproducibility

To reproduce the entire thesis empirical suite (from data spine construction to manuscript-ready LaTeX macros), execute the top-level orchestrator:

```bash
python execute_thesis_pipeline.py
```

This will sequentially execute the numbered scripts in `scripts/`, update the `registries/`, and refresh the `metrics_manifest.json` utilized by the manuscript build.

## Methodological Defenses

Following intensive committee-facing review, the infrastructure enforces:
- **Strict Leakage Protocols**: Target variables are aggressively dropped from engineering matrices.
- **YAML-Driven Tasks**: Removes hard-coded script logic in favor of versioned, auditable task configurations in `configs/tasks/`.
- **Dynamic NLP Embedding**: Vocabulary is strictly generated inside rolling cross-validation blocks to prevent future-text leakage.
- **Structural Audit**: Stage D is treated as a descriptive "Administrative Data Censorship" diagnostic rather than a predictive claim.
