# Predicting Zoning Opposition

**Daniel Hardesty Lewis**  
Master's Thesis, Urban Planning  
Columbia University, Graduate School of Architecture, Planning and Preservation

## Abstract

This thesis develops machine learning models and causal inference designs to evaluate the predictability and structural attrition of civic opposition to discretionary zoning. Utilizing a canonical universe of over 7,000 distinct rezoning and zoning map amendment filings in Austin, Texas from 2007 to 2024, the research operationalizes "measured threshold-crossing petitions" as its primary dependent variable. The study is explicitly bounded to the pre-HB24 institutional regime to ensure structural and temporal validity.

## The Empirical Universe (Dataset Hierarchy)

To prevent scope drift and methodological leakage, this thesis enforces a strict dataset hierarchy:

1. **Canonical Temporal Backbone (The Master Spine)**: A case-level discretionary zoning panel tracking 7,153 chronological cases established consistently from filing (H0) through City Council ordinance (H3), encompassing 2007–2024. 
2. **Track 1 Predictive Extract (Stage C)**: Engineered explicitly on filing-horizon constraints, stripped of future-leakage, employing rigorous missingness handling (Strict NaN constraint, no zero-imputation).
3. **Track 2 Text Embeddings (Active Learning NLP)**: Time-aware textual embeddings mapped dynamically over the corpus using expanding-window SVD pipelines to prevent forward-looking vocabulary contamination.
4. **Track 3 Causal DiD Baseline**: Evaluating structural shifts against the 2024 HOME Initiative via static TWFE models mapping explicitly observed petition outcomes.

*(Note: Earlier exploratory pipelines—such as the 2018-2025 generative parcel panels and diffusion pipelines built specifically for spatial backtesting overlays—are preserved only as auxiliary architectural experiments and do not represent the thesis's core empirical claims).*

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `Thesis_Draft/` | The active academic manuscript (`Draft_v1`), compiled via standard LaTeX processes (`pdflatex`). Encompasses all generated figures and macro tables. |
| `Data/` | The raw and assembled tracking warehouse. Contains `Warehouse_As_Of`, `Zoning_Cases`, and legacy `Panel` pipelines. |
| `Analysis/Scripts/Pipeline/` | Execution scripts governing the temporal build (`build_warehouse_as_of.py`) and rolling NLP generation (`build_tfidf_embeddings.py`) into the Master Spine. |
| `Analysis/Scripts/Modeling/Production_Models/` | The core thesis modeling suite (`StageC_opposition_risk.py`, `run_causal_track3_did_real.py`). Contains rigorous benchmarks (Calibration & Alternative Architectures). |
| `Analysis/Scripts/Visualization/Production_Figures/` | Visualization suite strictly mirroring the bounded target definitions utilized in the core thesis execution. ("plot_F17_DiD_real.py") |
| `Analysis/Scripts/Experiments/SHAP/` | Robust attribution and stability tests over expanding rolling-origin windows (`attribution_stability.py`). |

## Methodological Defenses

Following intensive committee-facing pipeline reviews, the analytical infrastructure enforces:
- **Strict Leakage Protocols**: Target variables (`signer_pct`, `signers`) are aggressively dropped from upstream (H1/H2) design matrices.
- **Dynamic NLP Embedding**: Global vocabulary fitting is banned to prevent future-text leakage. Vocabulary is strictly generated inside rolling cross-validation blocks.
- **Methodological Missingness**: Unobserved petition events (`NaN`) are explicitly dropped instead of defaulted to `0` (False Negatives), ensuring that temporal baseline rates, calibration models, and Causal Inference distributions reflect legitimate empirical measurements.

## Reproducibility

The final empirical results utilized within the manuscript can be successfully verified by executing the `Production_Models` suite inside `Analysis/Scripts/Modeling`. All graphical exhibits are generated natively via `Analysis/Scripts/Visualization`. Telemetry flows automatically via Python into `metrics_config.tex` for dynamic `pdflatex` rendering.
