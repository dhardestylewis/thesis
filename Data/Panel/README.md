# Panel Data — Property × Year

**Project**: Predicting Zoning Opposition  
**Author**: Daniel Hardesty Lewis  
**Last Updated**: 2026-02-19

## Overview

The panel has evolved through three versions:

| Version | Parcel Universe | Years | Total Rows | Source |
|---------|----------------|-------|------------|--------|
| v1 | 2,902 (GeoJSON only) | 2007–2024 | 52,236 | `Property_Year_Panel.csv` |
| v2 | 2,902 (+ protest_nearby) | 2007–2024 | 52,236 | `Property_Year_Panel_v2.csv` |
| **v3** | **282,772** (full TCAD) | **2019–2024** | **1,696,632** | **`Property_Year_Panel_v3.csv`** |

**v3 is the primary analysis file.** It uses a fixed parcel universe from LUI 2024 geometry.

## Directory Structure

```
Panel/
├── Output/
│   ├── Property_Year_Panel_v3.csv   # Primary (282,772 × 6 years)
│   ├── Property_Year_Panel.csv      # Legacy v1
│   └── Property_Year_Panel_v2.csv   # Legacy v2
├── Intermediate/
│   └── ears_YYYY_clean.csv          # Per-year cleaned EARS extracts
├── Reference/
│   ├── EARS_Column_Layout.csv       # 84-column AJR field mapping + leakage flags
│   ├── parcel_centroids.csv         # Lat/lon from LUI 2024 geometry
│   └── Variable_Codebook.md         # Variable definitions
├── Logs/
│   └── panel_v3_build.log           # Build log
└── README.md
```

## Panel v3 Construction

**Pipeline**: `Analysis/Scripts/Pipeline/rebuild_panel_v3.py`

### Step 1: Fixed Parcel Universe
- Source: LUI 2024 (`7vsm-dvxg`), 284,958 parcels with valid `parcel_id_10`
- After dedup: **282,772** unique parcels
- Expanded to 6 years (2019–2024) → 1,696,632 skeleton rows

### Step 2: EARS Appraisal Roll Merge
- EARS years 2019–2025 parsed from `Data/Appraisal_Rolls/YYYY/EARS_YYYY_Master.csv`
- 84 AJR fields mapped via `EARS_Column_Layout.csv`
- Match strategy: direct join on `parcel_id_10` (10-digit) or crosswalk from 6-digit `property_id`
- Backfill: years without year-matched EARS get nearest available year

### Step 3: Zoning Case Merge
- ZC CSV: 6,865 cases (1997–2024) with TCAD IDs → `zoning_case_on_parcel`
- 200ft buffer: polygon-overlap from `combined_cases_with_nearby.geojson` → `zoning_case_nearby`

### Step 4: Protest Petition Merge
- PDF signers: 8,843 parcels, 252 cases (2007–2024) → `protest_signed`

### Step 5: Spatial Data
- Centroids extracted from LUI 2024 MULTIPOLYGON WKT → `latitude`, `longitude`
- Coverage: 98.2% of parcels have valid coordinates

## Key Columns (v3)

| Column | Description |
|--------|-------------|
| `standardized_tcad_id` | 10-digit parcel ID (primary key with `year`) |
| `year` | Panel year (2019–2024) |
| `protest` | 1 if protest petition filed (from GeoJSON, legacy) |
| `protest_signed` | 1 if parcel owner signed protest petition (PDF ground truth) |
| `zoning_case_on_parcel` | 1 if a zoning case filed on this parcel |
| `zoning_case_nearby` | 1 if any zoning case within 200ft polygon overlap |
| `latitude`, `longitude` | Parcel centroid from LUI 2024 geometry |
| `market_value`, `assessed_value`, ... | EARS property valuation fields |
| `property_category_code` | EARS property type |
| `lui_general_land_use` | LUI land use classification |

## Temporal Leakage Audit

See `Reference/EARS_Column_Layout.csv` for full audit:

- **EXCLUDE** (2): `arb_protest_flag`, `arb_protest_result`
- **CAUTION** (7): ownership, sale date, new construction, zoning code
- **SAFE** (67): valuation, property characteristics, exemptions

## Design Decisions

1. **Fixed parcel universe**: LUI 2024 parcels used for all years (stable: <0.1% change 2012–2024)
2. **EARS backfill accepted**: Years without year-matched EARS get nearest year; temporal leakage acknowledged but post-2018 evaluation mitigates impact
3. **200ft buffer via polygon overlap**: Uses pre-computed area calculations, not centroid distance

## Reproduction

```bash
# Prerequisites: download CoA datasets
python Analysis/Scripts/Pipeline/download_coa_datasets.py

# Build panel v3
python Analysis/Scripts/Pipeline/rebuild_panel_v3.py

# Extract centroids
python Analysis/Scripts/Pipeline/extract_centroids.py
```
