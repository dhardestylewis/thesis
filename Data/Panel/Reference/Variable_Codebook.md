# Variable Codebook — Panel Dataset

**Project**: Predicting NIMBYism
**Created**: 2026-02-16
**Author**: Daniel Hardesty Lewis

## Temporal Leakage Audit Legend

| Risk Level | Meaning | Action |
|------------|---------|--------|
| **SAFE** | Variable determined before protest outcome; no leakage risk | Include as predictor |
| **CAUTION** | Potential indirect leakage; requires lagging or domain justification | Use lagged value or exclude |
| **EXCLUDE** | Direct outcome leakage; encodes the outcome itself | Never use as predictor |
| **N/A** | Metadata / identifier field | Not used as predictor |

---

## 1. Panel Identifiers

| Variable | Source | Type | Years | Time-Varying? | Leakage | Notes |
|----------|--------|------|-------|---------------|---------|-------|
| `standardized_tcad_id` | GeoJSON | Text | All | No | N/A | Primary property identifier |
| `year` | Constructed | Int | All | N/A | N/A | Panel year dimension |
| `protest` | GeoJSON | Binary | All | Yes | N/A | **Outcome variable** (1 = protest filed, 0 = none) |

---

## 2. EARS Appraisal Roll Variables (2018–2025)

Full mapping in [EARS_Column_Layout.csv](file:///c:/Users/dhl/data/thesis/thesis/Data/Panel/EARS_Column_Layout.csv)

### 2a. SAFE — Property Valuation (assessed Jan 1, before protest period)

| Variable | Field ID | Type | Description | Leakage |
|----------|----------|------|-------------|---------|
| `land_market_value` | AJR030 | Numeric | Market value of land | SAFE |
| `improvement_market_value` | AJR033 | Numeric | Market value of improvements | SAFE |
| `total_market_value` | AJR034 | Numeric | Total market value (land + improvements) | SAFE |
| `appraised_value` | AJR035 | Numeric | Appraised value after cap adjustments | SAFE |
| `assessed_value` | AJR036 | Numeric | Assessed value for tax base | SAFE |
| `taxable_value` | AJR046 | Numeric | Taxable value after exemptions | SAFE |
| `prior_year_taxable_value` | AJR081 | Numeric | Previous year taxable value | SAFE |

**Justification**: Texas appraisals are set as of January 1 of the tax year. Protest petitions against *zoning changes* are filed later. The appraisal process is independent of the zoning protest process.

### 2b. SAFE — Property Characteristics

| Variable | Field ID | Type | Description | Leakage |
|----------|----------|------|-------------|---------|
| `property_category_code` | AJR031 | Text | Category (A=SFR, B=MFR, C=Vacant, etc.) | SAFE |
| `subcategory_code` | AJR032 | Text | Detailed subcategory | SAFE |
| `year_built` | AJR023 | Numeric | Construction year | SAFE |
| `deed_acreage` | AJR026 | Numeric | Property acreage | SAFE |
| `land_acres` | AJR029 | Numeric | Acreage for valuation | SAFE |
| `homesite_flag` | AJR024 | Text | Homestead designation (Y/N) | SAFE |
| `improvement_sq_ft` | AJR025 | Text | Building square footage | SAFE |

### 2c. SAFE — Exemption Flags and Amounts

| Variable | Field ID | Type | Description | Leakage |
|----------|----------|------|-------------|---------|
| `exemption_flag_hs` | AJR039 | Text | Homestead exemption flag | SAFE |
| `exemption_flag_ov65` | AJR040 | Text | Over-65 exemption flag | SAFE |
| `exemption_flag_dp` | AJR041 | Text | Disabled person flag | SAFE |
| `exemption_flag_dv` | AJR042 | Text | Disabled veteran flag | SAFE |
| `exemption_amount_hs` | AJR037 | Numeric | Homestead exemption amount | SAFE |
| `exemption_amount_ov65` | AJR038 | Numeric | Over-65 exemption amount | SAFE |
| `total_exemption_amount` | AJR066 | Numeric | Total exemptions | SAFE |
| `freeze_flag` | AJR068 | Text | Tax ceiling/freeze | SAFE |

### 2d. CAUTION — Ownership and Transactions

| Variable | Field ID | Type | Description | Leakage | Mitigation |
|----------|----------|------|-------------|---------|------------|
| `owner_name` | AJR017 | Text | Owner name | CAUTION | Ownership changes may post-date protests; use for owner-type classification only |
| `most_recent_sale_date` | AJR027 | Date | Most recent sale | CAUTION | Use lagged (t-1) or derive `years_since_sale` |
| `new_construction_value` | AJR082 | Numeric | New construction value | CAUTION | Construction may be related to zoning; use lagged |
| `zoning_code` | AJR075 | Text | Zoning designation | CAUTION | Zoning changes are the *subject* of protests; use lagged (t-1) only |

### 2e. EXCLUDE — Direct Outcome Leakage

| Variable | Field ID | Type | Description | Leakage | Reason |
|----------|----------|------|-------------|---------|--------|
| `arb_protest_flag` | AJR078 | Text | ARB protest flag | EXCLUDE | **This is the outcome** — encodes whether appraisal was protested |
| `arb_protest_result` | AJR079 | Text | ARB protest result | EXCLUDE | Encodes protest hearing outcome (post-treatment) |

> [!CAUTION]
> **Critical distinction**: `arb_protest_flag` captures **appraisal protests** (property owner vs. appraisal district), which is a *different* process from **zoning protest petitions** (neighborhood opposition to rezoning). However, including it risks confounding: properties that protest appraisals may also be in neighborhoods that protest zoning. Exclude to be safe.

---

## 3. Zoning Case Variables (from GeoJSON, ~2007–2024)

| Variable | Source | Type | Time-Varying? | Leakage | Notes |
|----------|--------|------|---------------|---------|-------|
| `case_number` | Zoning Cases | Text | Yes | N/A | Case identifier |
| `case_type` | Zoning Cases | Text | Yes | SAFE | Type of zoning case |
| `proposed_zoning` | Zoning Cases | Text | Yes | SAFE | What rezoning is requested |
| `existing_zoning` | Zoning Cases | Text | Yes | SAFE | Current zoning at filing |
| `proposed_land_use` | Zoning Cases | Text | Yes | SAFE | Proposed land use |
| `existing_land_use` | Zoning Cases | Text | Yes | SAFE | Current land use at filing |
| `gross_site_area_acres` | Zoning Cases | Numeric | No | SAFE | Site area |
| `council_district` | Zoning Cases | Int | No | SAFE | City council district |
| `status_date` | Zoning Cases | Date | Yes | **EXCLUDE** | Encodes post-filing timing |
| `final_date` | Zoning Cases | Date | Yes | **EXCLUDE** | Encodes outcome timing |
| `approval_date` | Zoning Cases | Date | Yes | **EXCLUDE** | Encodes approval (post-treatment) |

---

## 4. Land Use Inventory Variables (City of Austin, biennial)

| Variable | Source | Type | Time-Varying? | Leakage | Notes |
|----------|--------|------|---------------|---------|-------|
| `land_use` | Land Use Inventory | Text | Biennial | SAFE | Parcel-level land use code |
| `general_land_use` | Land Use Inventory | Text | Biennial | SAFE | General category |
| `shape_area` | Land Use Inventory | Numeric | No | SAFE | Parcel geometry area |

---

## 5. Census / ACS Variables (tract-level, 5-year estimates)

| Variable | Source | Vintage | Time-Varying? | Leakage | Notes |
|----------|--------|---------|---------------|---------|-------|
| `total_population` | ACS 5yr | Annual | Yes | SAFE | Tract total population |
| `median_age` | ACS 5yr | Annual | Yes | SAFE | Tract median age |
| `race_white` | ACS 5yr | Annual | Yes | SAFE | White share |
| `race_black` | ACS 5yr | Annual | Yes | SAFE | Black share |
| `race_asian` | ACS 5yr | Annual | Yes | SAFE | Asian share |
| `race_hispanic` | ACS 5yr | Annual | Yes | SAFE | Hispanic share |
| `median_income` | ACS 5yr | Annual | Yes | SAFE | Tract median household income |
| `poverty_count` | ACS 5yr | Annual | Yes | SAFE | Persons below poverty line |
| `median_home_value` | ACS 5yr | Annual | Yes | SAFE | Median home value |
| `owner_occupied` | ACS 5yr | Annual | Yes | SAFE | Owner-occupied share |
| `renter_occupied` | ACS 5yr | Annual | Yes | SAFE | Renter-occupied share |
| `commute_time` | ACS 5yr | Annual | Yes | SAFE | Median commute time |

> [!NOTE]
> **ACS vintage matching**: The 2019 ACS 5-year estimate covers survey years 2015–2019. Assign to panel year = end year (2019). This prevents look-ahead bias because the estimate is released in the year following the end year (i.e., 2019 ACS released in late 2020, but the data describes conditions *up to* 2019).

---

## 6. Derived Variables (to construct)

| Variable | Formula | Leakage | Notes |
|----------|---------|---------|-------|
| `value_change_pct` | $(V_t - V_{t-1}) / V_{t-1}$ | SAFE | Year-over-year market value change (uses lagged) |
| `years_since_sale` | $t - \text{most\_recent\_sale\_date}$ | SAFE | Time since last transaction |
| `is_absentee_owner` | `owner_city != situs_city` | CAUTION | Investor proxy; use with care |
| `improvement_to_land_ratio` | $V_{\text{imp}} / V_{\text{land}}$ | SAFE | Development intensity |
| `age_of_structure` | $t - \text{year\_built}$ | SAFE | Building age |
