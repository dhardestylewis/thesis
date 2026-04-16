import pandas as pd
import numpy as np
import os
import warnings

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data")
OUT_DIR = os.path.join(DATA, "Warehouse_As_Of")
os.makedirs(OUT_DIR, exist_ok=True)

# Empirical core sources
ZONING_LDC_CSV = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "CSV", "enriched_zoning_data_causal.csv")
PETITIONS_CSV = os.path.join(DATA, "Protest_Petitions", "Backfilled", "petition_summary_backfilled.csv")
AGENDA_TEXT_CSV = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "CSV", "scraped_agenda_text_embeddings.csv")
STAFF_CSV = os.path.join(DATA, "Scraped_Agendas", "staff_recommendations.csv")

def validate_merge(df_before, df_after, df_right_name, merge_key, expected_increment=0):
    """Rigorous assertion checking for joins in the warehouse."""
    if len(df_after) != len(df_before) + expected_increment:
        raise ValueError(f"Merge with {df_right_name} changed row count! Before: {len(df_before)}, After: {len(df_after)}")
    if df_after[merge_key].duplicated().any():
        raise ValueError(f"Merge with {df_right_name} introduced duplicates on key {merge_key}.")

def build_horizons():
    print("Building As-Of Warehouse using strictly empirical data and timing constraints...")
    
    # ---------------------------------------------------------
    # Base structural and geometry
    # ---------------------------------------------------------
    df = pd.read_csv(ZONING_LDC_CSV, low_memory=False)
    
    # Normalize case number uniformly
    df['case_number'] = df['Case Number'].fillna(df.get('case_number', pd.Series(dtype=str))).astype(str).str.strip().str.upper()
    
    # Sort by application date so we keep the most recent if there are duplicates, then define unit of observation
    df['application_start_date'] = pd.to_datetime(df['application_start_date'], errors='coerce')
    df['final_date'] = pd.to_datetime(df['final_date'], errors='coerce')
    df = df.sort_values(['case_number', 'application_start_date']).drop_duplicates(subset=['case_number'], keep='last').copy()
    
    # Extract Year explicitly; DO NOT default unknown years to 2020
    df['year'] = pd.to_numeric(df['case_number'].str.extract(r'C\d+[A-Z]*-(\d{4})')[0], errors='coerce')
    df['gross_site_area_acres'] = pd.to_numeric(df['gross_site_area_acres'], errors='coerce').fillna(0)
    
    # ---------------------------------------------------------
    # Structural Missingness handling (do not fillna(0) blindly)
    # ---------------------------------------------------------
    ldc_cols = ['delta_max_height_ft', 'delta_max_far', 'delta_max_bldg_cov_pct', 'delta_min_lot_sqft']
    for col in ldc_cols:
        if col in df.columns:
            df[ col + '_is_missing' ] = df[col].isna().astype(int)
    # Keep missing as NaN. Downstream models will use imputation or support natively.
    ldc_features = ldc_cols + [col + '_is_missing' for col in ldc_cols]
    
    # ---------------------------------------------------------
    # 1. EMPIRICAL PETITIONS (Missingness aware)
    # ---------------------------------------------------------
    pet = pd.read_csv(PETITIONS_CSV)
    pet['case_number'] = pet['case_number'].astype(str).str.strip().str.upper()
    
    # Validate uniqueness before merge
    pet = pet.groupby('case_number').last().reset_index()
    
    pre_merge_len = len(df)
    df = df.merge(pet[['case_number', 'signers', 'signer_pct']], on='case_number', how='left')
    validate_merge(df.iloc[:pre_merge_len], df, 'Petitions', 'case_number')
    
    # Do not treat missing data as 'no petition'
    df['petition_record_found'] = df['signer_pct'].notna().astype(int)
    # Target label: 1 if >= 20%, 0 if < 20%, NaN if petition not recorded
    df['is_protested'] = np.where(df['petition_record_found'] == 1, (df['signer_pct'] >= 0.20).astype(int), np.nan)
    
    base_cols = ['case_number', 'year', 'application_start_date', 'final_date', 'gross_site_area_acres', 'council_district', 'is_protested']
    
    # ---------------------------------------------------------
    # H0: Filing (Static geometry, Math LDC Constraints)
    # ---------------------------------------------------------
    h0_cols = base_cols + ldc_features
    df[h0_cols].to_csv(os.path.join(OUT_DIR, "H0_Filing.csv"), index=False)
    print(f"H0 Built: {len(df)} rows")
    
    # ---------------------------------------------------------
    # H1: Notice
    # ---------------------------------------------------------
    # Do NOT include signers/signer_pct in the feature matrix since it leaks the target.
    # For now, H1 is identical to H0 since we don't have notice-specific features yet.
    h1_cols = h0_cols.copy() # Reserved for future notice-horizon features
    df[h1_cols].to_csv(os.path.join(OUT_DIR, "H1_Notice.csv"), index=False)
    print("H1 Built (Target-Leakage Removed)")
    
    # ---------------------------------------------------------
    # H2: Pre-Commission (Staff Recommendations)
    # ---------------------------------------------------------
    if os.path.exists(STAFF_CSV):
        staff_df = pd.read_csv(STAFF_CSV)
        staff_df['case_number'] = staff_df['CASE_NUMBER'].astype(str).str.strip().str.upper()
        
        # Aggregate to one row per case explicitly
        # Instead of pseudo-continuous friction, keep actual categories, default to 'Missing'
        staff_df['staff_recommendation_cat'] = staff_df['STAFF_RECOMMENDATION'].fillna('Missing')
        staff_agg = staff_df.groupby('case_number')['staff_recommendation_cat'].first().reset_index()
        
        pre_merge_len = len(df)
        df = df.merge(staff_agg, on='case_number', how='left')
        validate_merge(df.iloc[:pre_merge_len], df, 'Staff Recommendations', 'case_number')
        
        df['staff_recommendation_cat'] = df['staff_recommendation_cat'].fillna('Missing')
        h2_cols = h1_cols + ['staff_recommendation_cat']
    else:
        h2_cols = h1_cols

    df[h2_cols].to_csv(os.path.join(OUT_DIR, "H2_Pre_Commission.csv"), index=False) 
    print("H2 Built (Staff categories aggregated natively)")
    
    # ---------------------------------------------------------
    # H3: Pre-Council (Raw NLP for Rolling Windows)
    # ---------------------------------------------------------
    # Instead of global SVD, join raw text directly. The modeling loop will handle rolling TF-IDF.
    if os.path.exists(AGENDA_TEXT_CSV):
        embed_df = pd.read_csv(AGENDA_TEXT_CSV)
        embed_df['case_number'] = embed_df['CASE_NUMBER'].astype(str).str.strip().str.upper()
        
        # Need to fix date leakage conceptually. We use Meeting_Date if available.
        embed_df['Meeting_Date'] = pd.to_datetime(embed_df['Meeting_Date'], errors='coerce')
        embed_df['agenda_text_raw'] = embed_df['agenda_text_raw'].fillna("")
        
        # Filter raw text out if it exceeds case final date, assuming it's post-treatment
        # If dates are missing, keep it but issue a warning
        # Since we merge to main DF, we do it after merge
        
        # Aggregate to 1 row per case: string concatenation of all relevant pre-council agendas
        text_agg = embed_df.groupby('case_number').agg({
            'agenda_text_raw': lambda x: ' | '.join(x),
            'Meeting_Date': 'max' # use latest meeting date for tracking
        }).reset_index()
        
        pre_merge_len = len(df)
        df = df.merge(text_agg, on='case_number', how='left')
        validate_merge(df.iloc[:pre_merge_len], df, 'Agenda Texts', 'case_number')
        
        # Time-based leakage protection: Drop text if it is demonstrably generated AFTER the case finalized
        # (Though agenda texts should ideally be validated as BEFORE council)
        future_leak = (df['Meeting_Date'].notna()) & (df['final_date'].notna()) & (df['Meeting_Date'] > df['final_date'])
        if future_leak.sum() > 0:
            warnings.warn(f"Found {future_leak.sum()} agenda texts occurring after final_date! Masking text to prevent horizon leakage.")
            df.loc[future_leak, 'agenda_text_raw'] = ""
            
        h3_cols = h2_cols + ['agenda_text_raw']
    else:
        h3_cols = h2_cols
        
    df[h3_cols].to_csv(os.path.join(OUT_DIR, "H3_Pre_Council.csv"), index=False)
    print("H3 Built (Raw Text Merged for Rolling Evaluation)")
    
    print("Warehouse Built Successfully. Horizons now enforce explicit structural rules.")

if __name__ == '__main__':
    build_horizons()
