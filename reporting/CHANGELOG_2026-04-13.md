# Changelog — 2026-04-13

## Run Summary
- Command: `C:/Users/dhl/data/Thesis/thesis/.venv/Scripts/python.exe execute_thesis_pipeline.py`
- Result: Full orchestrator completed (`00`–`11`) with final gate status **`FINAL BUILD UNLOCKED`**.
- Total runtime: approximately `0.3 minutes`.

## Headline Stage C Metrics (latest registry output)
- Task: `STAGE_C_FILING_MAIN`
- Split: `TEMP_OOD_2023_MAIN`
- Model: `CatBoost`
- Score column: `y_score_calibrated`
- PR-AUC: `0.7477099019122676`
- Top-decile precision: `0.24193548387096775`
- Top-decile lift: `9.370967741935484`
- Brier: `0.011600586932944705`
- ECE: `0.008324013745886424`
- ACE: `0.004739587574473661`
- Calibration slope: `1.012153817443703`
- Thresholded metrics:
  - Precision@0.30: `0.6153846153846154`
  - Recall@0.30: `0.8`
  - Precision@0.50: `0.7142857142857143`
  - Recall@0.50: `0.5`
- Sample size: `1162`
- Positive rate: `0.025817555938037865`
- Generated at: `2026-04-13T23:33:14.344599+00:00`

## Gate/Report Outputs
- `reporting/submission_report.json` generated.
- `reporting/submission_report.md` generated.
- `Thesis_Draft/Draft_v1/Tables/metrics_config.tex` exported (79 macros).

## Interpretation Sidecar Note
- Meta-attribution run completed successfully.
- Warning surfaced by the sidecar: no feature cluster met all consensus criteria; avoid asserting an "invariant core" claim in manuscript text until those quantitative thresholds are met.

## Current Status
- Pipeline state is green.
- Submission gate checks passed: glossary terms, banned phrases, placeholders, cross-references, metric integrity, submission report.
