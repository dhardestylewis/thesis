# Predicting Zoning Opposition

**Daniel Hardesty Lewis**  
Master's Thesis, Urban Planning  
Columbia University, Graduate School of Architecture, Planning and Preservation

## Abstract

This thesis develops machine learning models to predict zoning opposition using Austin, Texas rezoning protest petition data from 2007–2025, with out-of-sample validation on 2018–2025 cases. Models include logistic regression and conditional diffusion (DDPM) for per-parcel protest risk forecasting.

## Data

- **Protest Petitions**: 2007–2025 (full history, 252 cases, 8,843 parcels from PDF parsing)
- **Appraisal Rolls (EARS)**: 2018–2025 (detailed property features, 84 AJR fields)
- **Panel v3**: 282,772 parcels × 6 years (2019–2024), balanced fixed-universe structure
- **Primary Study Period**: 2019–2025 (EARS coverage for full-feature modeling)

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `Thesis_Draft/` | Active thesis writing (Markdown, HTML, Outreach, Updates) |
| `Data/` | Research datasets (Appraisal Rolls, Protest Petitions, Zoning Cases, Panel) |
| `Analysis/` | Scripts, modeling, visualization, and results |
| `References/` | Supporting materials (Bibliography, Background, Theses, Prompts) |
| `Submitted/` | Finalized work (Proposal, Assignments, IRB Materials) |
| `Archive/` | Historical draft versions |
| `.meta/` | Project management (TODO, GUIDELINES, CHANGELOG) |

## Quick Navigation

### Active Work
- **Pipeline**: `Analysis/Scripts/Pipeline/` — Panel build, EARS parsing, centroid extraction
- **Modeling**: `Analysis/Scripts/Modeling/` — Backtests (naive, generative), benchmarks (CVAE, diffusion)
- **Visualization**: `Analysis/Scripts/Visualization/` — Heatmaps, timelapse, benchmark dashboard
- **Results**: `Analysis/Results/` — Backtests, benchmarks, visualizations (HTML)
- **Data Panel**: `Data/Panel/Output/Property_Year_Panel_v3.csv` — Primary analysis dataset

### Reference & Support
- **Bibliography**: `References/Bibliography/References.bib`
- **Background**: `References/Background_Comprehensive/`
- **Guidelines**: `.meta/GUIDELINES.md`
- **Tasks**: `.meta/TODO.md`

### Completed
- **Proposal**: `Submitted/Thesis_Proposal_Submitted/`
- **IRB**: `Submitted/IRB_Submitted/`
- **Coursework**: `Submitted/Assignments_Submitted/`

## Key Files

| File | Purpose |
|------|---------|
| `Data/Panel/Output/Property_Year_Panel_v3.csv` | Primary panel dataset (282,772 × 6 years) |
| `Data/Panel/Reference/EARS_Column_Layout.csv` | EARS field definitions with leakage flags |
| `Data/Panel/Reference/parcel_centroids.csv` | Parcel lat/lon from LUI 2024 geometry |
| `Analysis/Results/protest_timelapse.html` | Interactive backtest timelapse (LogReg) |
| `Analysis/Results/generative_timelapse.html` | Generative model timelapse (LogReg + Diffusion) |
| `.meta/TODO.md` | Prioritized task list |

## Conventions

See `.meta/GUIDELINES.md` for full documentation.

### Directory & File Casing
- **Directories**: `Title_Case` (e.g., `Appraisal_Rolls`, `Notebooks`)
- **Data Files**: `EARS_YYYY_Description.ext`
- **Reference Theses**: `Author_Year_University_Title.pdf`

## Recent Changes

- **2026-02-19**: Panel v3 rebuild with fixed parcel universe (282,772), centroid extraction, diffusion v2 model
- **2026-02-19**: Updated all modeling/visualization scripts from v1 → v3 panel paths
- **2026-01-21**: Hybrid directory reorganization
- **2026-01-21**: Imported protest petitions and zoning data from Google Drive

## Project Status

- ✅ Proposal submitted and approved
- ✅ IRB approved
- ✅ Data collection complete (protest petitions 2007–2025)
- ✅ Panel v3 built (282,772 parcels × 6 years)
- ✅ Backtest pipeline running (LogReg + Diffusion)
- 🔄 Thesis drafting in progress
- 📊 Diffusion v2 model under active development
