# Data

Research datasets for the thesis on predicting NIMBYism in Austin, Texas.

## Directory Structure

```
Data/
├── Protest_Petitions/            # Rezoning protest petitions (2007-2025)
│   ├── EARS/                     # Appraisal roll zips (2019-2022)
│   ├── pickles/                  # Pickled dataframes by year
│   ├── geojson/                  # Petition geometries (v2-v12)
│   ├── models/                   # Model artifacts
│   └── analysis_results/         # Analysis outputs
├── Zoning_Cases/                 # Austin zoning cases with nearby parcels
│   ├── geojson/                  # Spatial data (cases, parcels)
│   └── csv/                      # Tabular data (land use, enriched cases)
├── documents/                    # Original petition PDFs
└── README.md (this file)
```

## Datasets

### Protest_Petitions/
**Status**: Restructured and populated

Comprehensive dataset of Austin rezoning protest petitions from 2007-2025 with Travis County Appraisal District (TCAD) property data enrichment.

**Subdirectories**:
- **EARS/**: Electronic Appraisal Roll Snapshots (zip files) for 2018-2022
- **pickles/**: Processed dataframes stored as pickle files
- **geojson/**: Progressive versions of petition data mixed with spatial info
- **models/**, **analysis_results/**: Placeholders for analysis outputs

### Zoning_Cases/
**Status**: Restructured

Austin zoning cases matched with nearby parcels and land use inventory data.

**Subdirectories**:
- **geojson/**: Spatial files including `combined_cases_with_nearby.geojson`, `parcels_within_200ft.geojson`
- **csv/**: Tabular data including `land_use_inventory_prefetched.csv`, `enriched_zoning_data.csv`

Austin zoning cases matched with nearby parcels and land use inventory data.

**Key files**:
- `combined_cases_with_nearby.geojson` - Zoning cases with nearby parcel geometries
- `land_use_inventory_prefetched.csv` - Austin land use inventory (183MB)
- `enriched_zoning_data_updated.csv` - Enriched zoning case information

### documents/
Original petition PDFs for reference and validation.

## Data Sources

- **Travis County Appraisal District (TCAD)**: Property data via EARS exports
- **City of Austin**: Zoning cases and land use inventory  
- **Austin Planning Department**: Scanned protest petition records

## Usage Notes

- Large GeoJSON files optimized for GIS software (QGIS, ArcGIS)
- Pickled dataframes require pandas to load
- Data spans 2007-2025 for training (2007-2017) and validation (2018-2025)

## Related

See `Analysis/notebooks/` for data processing and modeling notebooks.
