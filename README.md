# Predicting NIMBYism

**Daniel Hardesty Lewis**  
Master's Thesis, Urban Planning  
Columbia University, Graduate School of Architecture, Planning and Preservation

## Abstract

This thesis develops machine learning models to predict neighborhood opposition (NIMBYism) to housing development using Austin, Texas rezoning protest letters data from 2007-2025, with out-of-sample validation on 2018-2025 cases.

## Data

- **Protest Petitions**: 2007–2025 (full history)
- **Appraisal Rolls (EARS)**: 2018–2025 (detailed property features)
- **Primary Study Period**: 2018–2025 (Intersection for full-feature modeling)

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `Thesis_Draft/` | Active thesis writing (Markdown, HTML, Outreach, Updates) |
| `Data/` | Research datasets (Appraisal Rolls, Protest Petitions, Zoning Cases) |
| `Analysis/` | Computational work (Notebooks, Scripts, Modeling) |
| `References/` | Supporting materials (Bibliography, Background, Theses, Prompts) |
| `Submitted/` | Finalized work (Proposal, Assignments, IRB Materials) |
| `Archive/` | Historical draft versions (standardized versioning) |
| `.meta/` | Project management (TODO, GUIDELINES, CHANGELOG, PROMPTS-LOG) |

## Quick Navigation

### Active Work
- **Writing**: `Thesis_Draft/` - Current thesis drafts, outreach strategy, and updates.
- **Analysis**: `Analysis/Notebooks/` - Jupyter notebooks for modeling (XGBoost).
- **Data**: `Data/Appraisal_Rolls/` - Standardized EARS (2018-2022) data.

### Reference & Support
- **Bibliography**: `References/Bibliography/References.bib`
- **Background**: `References/Background_Comprehensive/`
- **Guidelines**: `.meta/GUIDELINES.md` - Project conventions
- **Tasks**: `.meta/TODO.md` - Current prioritized task list

### Completed
- **Proposal**: `Submitted/Thesis_Proposal_Submitted/`
- **IRB**: `Submitted/IRB_Submitted/` - IRB protocol and CITI certificates
- **Coursework**: `Submitted/Assignments_Submitted/`


## Key Files

| File | Purpose |
|------|---------|
| `.meta/TODO.md` | Prioritized task list (P1/P2/P3/P4) |
| `.meta/GUIDELINES.md` | Project conventions and best practices |
| `.meta/CHANGELOG.md` | Session history with REVIEWED/UNREVIEWED status |
| `References/Bibliography/References.bib` | Master bibliography |

## Conventions

See `.meta/GUIDELINES.md` for full documentation. Key patterns:

### Directory & File Casing
- **Directories**: Always `Title_Case` (e.g., `Appraisal_Rolls`, `Notebooks`).
- **Data Files**: `EARS_YYYY_Description.ext`.
- **Reference Theses**: `Author_Year_University_Title.pdf`.

### Suffixes & Status
- `_Submitted`: Finalized academic work.
- `_Draft`: Active iteration.
- `-TBD`: Placeholder for future work.
- `-DEPRECATED`: Kept temporarily for reference or link validity.


## Recent Changes

- **2026-01-21**: Implemented hybrid directory reorganization for better functional organization
- **2026-01-21**: Imported protest petitions and zoning data from Google Drive
- **2026-01-21**: Migrated project metadata to `.meta/` directory

## Project Status

- ✅ Proposal submitted and approved
- ✅ IRB approved
- ✅ Data collection complete (protest petitions 2007-2025)
- 🔄 Thesis drafting in progress
- 📊 Modeling and analysis ongoing
