# Predicting Zoning Opposition

**Daniel Hardesty Lewis**  
Master's Thesis, Urban Planning  
Columbia University, Graduate School of Architecture, Planning and Preservation

## Abstract

This thesis studies whether public-record information available at filing can rank Austin discretionary zoning cases by the likelihood of a reconstructed threshold-crossing petition under the pre-HB 24 regime. The repository is organized around one canonical Stage C prediction task, with Stage A retained only as a selection-correction sidecar and meta-attribution retained as a bounded interpretive sidecar.

## The Auditable Pipeline (Refactored April 2026)

The repository is organized around three auditable objects:

1. a canonical Stage C prediction registry,
2. a label-validity registry, and
3. a meta-attribution registry with uncertainty-aware cluster summaries.

### Directory Structure

|Directory|Purpose|
|---------|-------|
|`src/`|Core logic for labels, features, splits, models, interpretation, and reporting.|
|`scripts/`|Numbered entry points for the registered pipeline sequence.|
|`configs/`|Task, split, feature, and interpretation definitions.|
|`registries/`|Immutable outputs: case universe, label registry, split registry, prediction registry, interpretation registry, and metrics manifest.|
|`Thesis_Draft/`|Manuscript sources and generated tables/macros.|
|`Analysis/`|Legacy or auxiliary material retained for reference.|

## Core Components

1. **Canonical Stage C Task**: one primary filing-date prediction task with a reduced benchmark family.
2. **Label Fidelity Audit**: a registered validity check against a reconstructed threshold-crossing outcome.
3. **Meta-Attribution Sidecar**: semantic cluster aggregation with explicit uncertainty and consensus rules.
4. **Selection-Correction Sidecar**: Stage A inverse-probability weights used only for sensitivity comparison.

## Reproducibility

To reproduce the registered thesis pipeline, execute the top-level orchestrator:

```bash
python execute_thesis_pipeline.py
```

This will execute the numbered scripts in `scripts/`, update the registry artifacts, and refresh the manuscript metrics manifest.

## Methodological Defenses

The current design keeps the thesis narrow:

- Stage C is the only canonical predictive task.
- Stage A is a bounded selection-correction support track.
- Meta-attribution is treated as a registered experiment with uncertainty.
- Stage D is descriptive only and excluded from the predictive DAG.
