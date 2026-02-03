# Data

Research datasets for the thesis on predicting NIMBYism in Austin, Texas.

## Directory Structure

```
Data/
├── Appraisal_Rolls/          # TCAD Appraisal Roll Exports (EARS) - [EXCLUDED]
│   ├── 2018/--2025/          # Annual data dumps (Text/CSV + Layouts)
│   └── README.md
├── Protest_Petitions/        # Derived protest data & training sets
│   ├── Pickles/              # Serialized pandas dataframes
│   ├── GeoJSON/              # Spatial petition data
│   └── Models/               # Trained models
├── Zoning_Cases/             # City of Austin Zoning Cases
│   ├── Source_Data/          # Raw downloads from Data Portal
│   ├── Processed_Data/       # Enriched datasets
│   └── QC_Logs/              # Merge/Match quality logs
├── Documents/                # Raw PDF Petition Files
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
*   **Recreation**: The derived datasets (`.pkl`, `.geojson`) in `Protest_Petitions/` are the result of OCR processing in `Analysis/Notebooks/`.
*   **Backup**: Full processed datasets are available in the project's S3/G:Drive backup.
