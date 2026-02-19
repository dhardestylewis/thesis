# Data

Research datasets for the thesis on predicting zoning opposition in Austin, Texas.

## Directory Structure

```
Data/
├── Appraisal_Rolls/          # TCAD Appraisal Roll Exports (EARS) - [EXCLUDED]
│   ├── 2018/--2025/          # Annual data dumps (Text/CSV + Layouts)
│   └── README.md
├── CoA_Open_Data/            # City of Austin Open Data downloads
├── Documents/                # Raw PDF Petition Files
│   └── C241282.PD.NRN.petitions.pdf
├── Panel/                    # Balanced property-year panel dataset
│   ├── Property_Year_Panel.csv      # Base panel (EARS + TCAD + zoning cases)
│   ├── Property_Year_Panel_v2.csv   # Panel with protest_zoning + protest_nearby
│   ├── ears_YYYY_clean.csv          # Per-year cleaned EARS extracts
│   ├── panel_with_*.csv             # Intermediate build stages (for debugging)
│   ├── EARS_Column_Layout.csv       # EARS AJR 84-column mapping with leakage flags
│   ├── Variable_Codebook.md         # Variable definitions + temporal leakage audit
│   └── README.md
├── Protest_Petitions/        # Derived protest data & training sets
│   ├── GeoJSON/              # Spatial petition data (protest_petitions_v1.geojson)
│   ├── petition_signers_from_pdf.csv  # 8,843 parcels parsed from petition PDF
│   ├── petition_summary_from_pdf.csv  # 252 case summaries
│   ├── Analysis_Results/     # Legacy analysis outputs
│   └── Models/               # Trained models
├── Zoning_Cases/             # City of Austin Zoning Cases
│   ├── Source_Data/          # Raw downloads from Data Portal
│   ├── Processed_Data/       # Enriched datasets
│   └── QC_Logs/              # Merge/Match quality logs
└── README.md
```

## Data Reconstruction & Sources

**Automated Setup**:
Run the included python script to automatically create directories, download public zoning data, and rename raw EARS files:
```bash
python ../Analysis/Scripts/setup_project.py
```

Since large files (>100MB) are excluded from this repository, follow these steps to reconstruct the full dataset:

### 1. Appraisal Rolls (EARS)
*   **Source**: Texas Comptroller of Public Accounts / Travis County Appraisal District (TCAD).
*   **Access**: [Property Tax Data Portal](https://comptroller.texas.gov/taxes/property-tax/) or request from TCAD.
*   **Action**: Download "Electronic Appraisal Roll Submission" (EARS) files for years 2018-2025.
*   **Placement**: Unzip into `Data/Appraisal_Rolls/{YYYY}/`.
*   **Naming**: Rename primary files to match `EARS_YYYY_Jurisdiction_Tax_Values.txt`.

### 2. Zoning Cases
*   **Source**: [Austin Open Data Portal](https://data.austintexas.gov/).
*   **Dataset**: "Zoning Cases" and "Land Use Inventory".
*   **Action**: Download as CSV/GeoJSON.
*   **Placement**: Place raw files in `Data/Zoning_Cases/Source_Data/`.

### 3. Protest Petitions
*   **Source**: Validated supermajority protest letters (City Clerk).
*   **PDF**: `Documents/C241282.PD.NRN.petitions.pdf` — 558 pages, 252 cases (2007-2024).
*   **Parsing**: Run `Analysis/Scripts/parse_petition_pdf.py` to extract signer data.
*   **Panel integration**: Run `Analysis/Scripts/rebuild_protest_panel.py` to merge into panel.

### Original File Inventory (Pre-Processing)

These files existed in the original data delivery before processing:

**Protest Petitions source files**: `2018-EARS.zip`, `2020_EARS_101620.zip`, `2021EARS092521.zip`, `227EARS092822.zip`, `227EARS093019.zip`, `Certified_Appraisal...`, `gathered_dfs_year_*`

**Zoning Cases source files**: `combined_cases_with...`, `df_matched_with_z...`, `enriched_zoning_d...`, `land_use_inventor...`, `nearby_parcels_wi...`, `parcels_within_20...`, `protest_petitions...`, `zoning_cases_with...`, `zoning_land_use_*`
