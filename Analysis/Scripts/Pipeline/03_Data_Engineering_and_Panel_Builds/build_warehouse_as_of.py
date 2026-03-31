import pandas as pd
import numpy as np
import os

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data")
OUT_DIR = os.path.join(DATA, "Warehouse_As_Of")
os.makedirs(OUT_DIR, exist_ok=True)

# Empirical core sources
ZONING_LDC_CSV = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "CSV", "enriched_zoning_data_causal.csv")
PETITIONS_CSV = os.path.join(DATA, "Protest_Petitions", "Backfilled", "petition_summary_backfilled.csv")
AGENDA_TEXT_CSV = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "CSV", "scraped_agenda_text_embeddings.csv")

def build_horizons():
    print("Building As-Of Warehouse using strictly empirical data...")
    
    # Base structural and geometry
    df = pd.read_csv(ZONING_LDC_CSV, low_memory=False)
    # Ensure uniform case string
    df['case_number'] = df['Case Number'].fillna(df.get('case_number', pd.Series(dtype=str))).astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=['case_number']).copy()
    
    # 1. Join Empirical Petitions (Ground truth target and Notice volume)
    pet = pd.read_csv(PETITIONS_CSV)
    pet['case_number'] = pet['case_number'].astype(str).str.strip().str.upper()
    
    df = df.merge(pet[['case_number', 'signers', 'signer_pct']], on='case_number', how='left')
    df['signers'] = df['signers'].fillna(0)
    df['signer_pct'] = df['signer_pct'].fillna(0)
    
    # Explicit target
    df['is_protested'] = (df['signer_pct'] >= 0.20).astype(int)
    
    # Fix Year from case_number
    df['year'] = pd.to_numeric(df['case_number'].str.extract(r'C\d+[A-Z]*-(\d{4})')[0], errors='coerce').fillna(2020)
    df['gross_site_area_acres'] = pd.to_numeric(df['gross_site_area_acres'], errors='coerce').fillna(0)
    
    # 2. Join Empirical NLP Embeddings (H3 Pre-Council Signal)
    EMBEDDINGS_CSV = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "CSV", "agenda_tfidf_embeddings.csv")
    if os.path.exists(EMBEDDINGS_CSV):
        embed_df = pd.read_csv(EMBEDDINGS_CSV)
        embed_df['case_number'] = embed_df['case_number'].astype(str).str.strip().str.upper()
        
        # Merge the 20 dimensions
        df = df.merge(embed_df, on='case_number', how='left')
        
        nlp_cols = [c for c in embed_df.columns if c.startswith('nlp_svd_')]
        df[nlp_cols] = df[nlp_cols].fillna(0)
    else:
        nlp_cols = []
    
    # Baseline columns
    base_cols = ['case_number', 'year', 'gross_site_area_acres', 'council_district', 'is_protested']
    
    # H0: Filing (Static geometry, Math LDC Constraints)
    # From zoning_delta_calculator.py
    ldc_cols = ['delta_max_height_ft', 'delta_max_far', 'delta_max_bldg_cov_pct', 'delta_min_lot_sqft']
    h0_cols = base_cols + ldc_cols
    
    # Drop rows missing structural Math (otherwise baseline models fail on NaNs)
    df[h0_cols] = df[h0_cols].fillna(0) # or drop, but fill is safer for baseline execution
    
    print(f"H0 Target Dimension: {len(df)}")
    df[h0_cols].to_csv(os.path.join(OUT_DIR, "H0_Filing.csv"), index=False)
    
    # H1: Notice (Institutional Entry)
    h1_cols = h0_cols + ['signers', 'signer_pct']
    df[h1_cols].to_csv(os.path.join(OUT_DIR, "H1_Notice.csv"), index=False)
    
    # H2: Pre-Commission (Merge Empirical Staff Recommendations)
    STAFF_CSV = os.path.join(DATA, "Scraped_Agendas", "staff_recommendations.csv")
    if os.path.exists(STAFF_CSV):
        staff_df = pd.read_csv(STAFF_CSV)
        # Map categorical text to structural baseline: Approval=0.0 (baseline support), Disapproval=1.0 (friction), missing=0.5
        staff_df['staff_friction_index'] = staff_df['STAFF_RECOMMENDATION'].map({'Approval': 0.0, 'Disapproval': 1.0}).fillna(0.5)
        df = df.merge(staff_df[['CASE_NUMBER', 'staff_friction_index']], left_on='case_number', right_on='CASE_NUMBER', how='left')
        df['staff_friction_index'] = df['staff_friction_index'].fillna(0.5)
        h2_cols = h1_cols + ['staff_friction_index']
    else:
        h2_cols = h1_cols

    df[h2_cols].to_csv(os.path.join(OUT_DIR, "H2_Pre_Commission.csv"), index=False) 
    
    # H3: Pre-Council (Empirical TF-IDF/SVD Text Embeddings)
    h3_cols = h2_cols + nlp_cols
    df[h3_cols].to_csv(os.path.join(OUT_DIR, "H3_Pre_Council.csv"), index=False)
    
    print("Warehouse Built Successfully integrating LDC Deltas, Signatures, and Agenda constraints.")

if __name__ == '__main__':
    build_horizons()
