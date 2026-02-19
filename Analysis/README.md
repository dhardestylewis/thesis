# Analysis

Computational analysis, modeling, and figure generation for the thesis.

## Directory Structure

```
Analysis/
├── Notebooks/               # Jupyter notebooks for exploration and modeling
├── Scripts/                 # Python/R scripts for data processing (TBD)
└── README.md (this file)

```

### Notebooks/
- **01_Petition_Extraction_and_Cleaning.ipynb**: PDF text extraction and initial cleaning.
- **02_Data_Integration_and_Enrichment.ipynb**: Merging petition data with TCAD property records.
- **03_Modeling_Zoning_Opposition.ipynb**: Predictive modeling of zoning opposition.

## Workflow

1. **Data Processing**: Scripts in `Scripts/` clean and prepare raw data
2. **Exploration**: Notebooks in `Notebooks/` perform EDA and feature engineering
3. **Modeling**: Train and evaluate machine learning models for zoning opposition prediction
4. **Visualization**: Generate figures and tables for thesis

## Dependencies

See thesis proposal for computational environment details. Key dependencies likely include:
- pandas, geopandas for data manipulation
- scikit-learn for machine learning
- matplotlib, seaborn for visualization

## Related

- Data sources: `../Data/`
- Thesis writing: `../Draft/`
- Reference materials: `../References/`
