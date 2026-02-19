# Panel Data — Property × Year

**Project**: Predicting NIMBYism with Causal Inference
**Author**: Daniel Hardesty Lewis
**Created**: 2026-02-16
**Last Updated**: 2026-02-16

## Overview

Balanced panel: **2,902 properties × 18 years (2007–2024) = 52,236 rows**.

The panel is constructed from protest petition GeoJSON data and enriched with
EARS appraisal rolls, City of Austin Land Use / Land Database snapshots, and
ACS 5-year Census estimates.

## Pipeline

Two scripts, run in order:

1. **`build_panel.py`** — Constructs the base panel (Steps 1–7)
2. **`enrich_panel.py`** — Adds time-varying census & forward-filled covariates

### Final Outputs

| File | Rows | Description |
|------|------|-------------|
| `Property_Year_Panel.csv` | 52,236 | Base panel with EARS + static census snapshot |
| `Property_Year_Panel_Enriched.csv` | 52,236 | **Primary analysis file** with time-varying ACS, LDB, LUI |

### Intermediate Files

| File | Description |
|------|-------------|
| `property_universe.csv` | 2,902 unique properties with metadata |
| `property_year_skeleton.csv` | Balanced skeleton with `protest` outcome |
| `ears_YYYY_clean.csv` | Cleaned EARS per year (2019–2022) |
| `id_crosswalk.csv` | 269,108 parcel_id_10 → EARS account_number mappings |
| `census_tract_timeseries.csv` | 3,521 tract × vintage ACS records |

---

## Data Sources & Provenance

### Protest Petitions (Outcome Variable)
- **Source**: `protest_petitions_v1.geojson` (27,363 features)
- **Outcome**: `protest = 1` if zoning protest petition filed in that property-year, else 0
- **Base rate**: 5.75% across all property-years

### EARS Appraisal Rolls (2019–2022)
- **Source**: Travis County EARS data (`EARS_YYYY_Master.csv`)
- **Match rate**: 72.7% of eligible property-years
- **Leakage audit**: 2 columns excluded (`arb_protest_flag`, `arb_protest_result`), 7 flagged CAUTION. See `EARS_Column_Layout.csv`.

### Census / ACS (Time-Varying)
- **Source**: US Census Bureau ACS 5-year estimates via API
- **Vintages**: 2009–2023 (15 years)
- **Join key**: 11-digit tract GEOID (panel has 12-digit block group GEOIDs, truncated to tract)
- **Match rate**: 75.3% of panel rows
- **Unmatched cause**: Properties without a GEOID, or panel years 2007-2008 predating ACS 2009

> [!IMPORTANT]
> **Design decision**: Panel year *Y* is matched to the ACS vintage ≤ *Y*. E.g., panel year 2015 uses ACS 2015 (which covers survey years 2011–2015). Panel years 2007–2008 get ACS 2009 (backward-fill).

### Land Database (Forward-Filled)
- **Sources**: LDB 2016 (`4nsn-uea6`, 248K parcels) and LDB 2021 (`kk8y-6cmt`, 250K parcels)
- **Covariates**: base zoning, effective zoning, FAR, lot size, units, year built, appraised value, market value, improvement type, I-35 side, council district
- **Match rate**: 98.3%
- **Forward-fill rule**: Use most recent snapshot ≤ panel year; backward-fill from 2016 for years 2007–2015

> [!IMPORTANT]
> **Design decision**: `ldb_source_year` column tracks which snapshot each row came from.

### Land Use Inventory (Forward-Filled)
- **Sources**: LUI 2012 (`3k7r-w54d`, 283K parcels), LUI 2022 (`6qkk-xgys`, 258K parcels), LUI 2024 (`7vsm-dvxg`, 285K parcels)
- **Match rate**: 99.8%
- **Forward-fill rule**: Same as Land Database

> [!IMPORTANT]
> **Design decision**: `lui_source_year` column tracks which snapshot each row came from.

### All Downloaded Datasets
See `Data/CoA_Open_Data/MANIFEST.md` for full provenance listing with dataset IDs and portal URLs.

---

## ID Crosswalk

**Problem**: GeoJSON uses 10-digit `standardized_tcad_id`, EARS uses 6-digit `account_number` — **zero direct overlap**.

**Solution**: Land Use Inventory's `property_id` field matches EARS `account_number`.
- 257,764 `property_id` values overlap with EARS accounts
- Crosswalk: `parcel_id_10` (= GeoJSON TCAD ID) → `property_id` (= EARS account)

---

## Temporal Leakage Audit

See `EARS_Column_Layout.csv` and `Variable_Codebook.md` for the full audit.

- **EXCLUDE** (2): `arb_protest_flag`, `arb_protest_result` — directly encode outcomes
- **CAUTION** (7): ownership, sale date, new construction, zoning code — require lagging
- **SAFE** (67): valuation, property characteristics, exemptions — set before protest period

---

## Forward-Fill Strategy

For data sources that provide periodic snapshots (not annual), we forward-fill:

| Source | Snapshot Years | Fill Strategy | Tracking Column |
|--------|---------------|---------------|-----------------|
| Land Database | 2016, 2021 | Most recent ≤ panel year; backward from 2016 for 2007–2015 | `ldb_source_year` |
| LUI | 2012, 2022, 2024 | Most recent ≤ panel year; backward from 2012 for 2007–2011 | `lui_source_year` |
| ACS Census | 2009–2023 (annual) | Exact vintage ≤ panel year; backward from 2009 for 2007–2008 | `acs_vintage` |

---

## Reproduction

```bash
cd thesis/Analysis/Scripts
python build_panel.py      # ~4 min, produces base panel
python enrich_panel.py     # ~1 min, adds time-varying data
```

Prerequisite: Download CoA datasets first:
```bash
python download_coa_datasets.py  # ~10 min, downloads 8 datasets to Data/CoA_Open_Data/
```
