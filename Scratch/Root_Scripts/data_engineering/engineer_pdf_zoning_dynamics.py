import pandas as pd
import numpy as np
import re

# Austin LDC Ch. 25-2 dimensional standards
AUSTIN_LDC_TABLE = {
    "RR":   {"max_height_ft": 35,  "max_far": 0.05},
    "LA":   {"max_height_ft": 35,  "max_far": 0.15},
    "DR":   {"max_height_ft": 35,  "max_far": 0.15},
    "SF-1": {"max_height_ft": 35,  "max_far": 0.20},
    "SF-2": {"max_height_ft": 35,  "max_far": 0.35},
    "SF-3": {"max_height_ft": 35,  "max_far": 0.40},
    "SF-4A":{"max_height_ft": 35,  "max_far": 0.45},
    "SF-4B":{"max_height_ft": 35,  "max_far": 0.45},
    "SF-5": {"max_height_ft": 35,  "max_far": 0.50},
    "SF-6": {"max_height_ft": 35,  "max_far": 0.40},
    "MH":   {"max_height_ft": 35,  "max_far": 0.50},
    "MF-1": {"max_height_ft": 40,  "max_far": 0.50},
    "MF-2": {"max_height_ft": 40,  "max_far": 0.60},
    "MF-3": {"max_height_ft": 40,  "max_far": 0.75},
    "MF-4": {"max_height_ft": 60,  "max_far": 1.00},
    "MF-5": {"max_height_ft": 60,  "max_far": 1.00},
    "MF-6": {"max_height_ft": 90,  "max_far": 3.00},
    "NO":   {"max_height_ft": 35,  "max_far": 0.35},
    "LO":   {"max_height_ft": 40,  "max_far": 0.70},
    "GO":   {"max_height_ft": 60,  "max_far": 1.00},
    "CR":   {"max_height_ft": 35,  "max_far": 0.35},
    "LR":   {"max_height_ft": 40,  "max_far": 0.50},
    "GR":   {"max_height_ft": 60,  "max_far": 1.00},
    "CS":   {"max_height_ft": 60,  "max_far": 2.00},
    "CS-1": {"max_height_ft": 60,  "max_far": 2.00},
    "CH":   {"max_height_ft": 120, "max_far": 3.00},
    "IP":   {"max_height_ft": 60,  "max_far": 1.00},
    "LI":   {"max_height_ft": 60,  "max_far": 1.00},
    "MI":   {"max_height_ft": 60,  "max_far": 2.00},
    "HI":   {"max_height_ft": 60,  "max_far": 2.00},
    "CBD":  {"max_height_ft": 400, "max_far": 8.00},
    "DMU":  {"max_height_ft": 120, "max_far": 5.00},
}

def get_ldc_metrics(zone_str):
    if pd.isna(zone_str) or zone_str == 'Unknown':
        return np.nan, np.nan
    zone_str = re.sub(r"\s+", "", str(zone_str).upper())
    base = re.match(r"^([A-Z]{1,5}(?:-[0-9A-Z]+)?)", zone_str)
    if not base:
        return np.nan, np.nan
    b = base.group(1)
    if b in AUSTIN_LDC_TABLE:
        return float(AUSTIN_LDC_TABLE[b]["max_height_ft"]), float(AUSTIN_LDC_TABLE[b]["max_far"])
    b2 = re.sub(r"[A-Z]$", "", b)
    if b2 in AUSTIN_LDC_TABLE:
        return float(AUSTIN_LDC_TABLE[b2]["max_height_ft"]), float(AUSTIN_LDC_TABLE[b2]["max_far"])
    return np.nan, np.nan

def clean_zoning(z_str):
    if pd.isna(z_str) or str(z_str).strip() == '':
        return 'Unknown'
    z = str(z_str).upper().strip()
    z = z.split('-CO')[0].split('-NP')[0].split('-H')[0]
    return z

def engineer_pdf_zoning():
    panel_path = r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv'
    pdf_path = r'c:\Users\dhl\data\Thesis\thesis\Data\interim\pdf_height_features.csv'
    
    print("Loading data...")
    panel = pd.read_csv(panel_path)
    pdf_df = pd.read_csv(pdf_path)
    
    # We want these columns
    cols_to_keep = [
        'case_number', 'source_date', 
        'pdf_requested_zoning', 'pdf_staff_recommended_zoning', 'pdf_approved_zoning',
        'pdf_reduced_to_ft', 'pdf_staff_recommends_ht'
    ]
    pdf_df = pdf_df[cols_to_keep].dropna(subset=['source_date'])
    
    # Clean textual zoning
    for col in ['pdf_requested_zoning', 'pdf_staff_recommended_zoning', 'pdf_approved_zoning']:
        pdf_df[col] = pdf_df[col].apply(clean_zoning)
        
    print("Converting dates...")
    panel['period_start_dt'] = pd.to_datetime(panel['period_start'])
    pdf_df['source_date_dt'] = pd.to_datetime(pdf_df['source_date'])
    
    # Drop existing overlapping columns from panel if they exist
    overlap_cols = [c for c in panel.columns if c.startswith('pdf_') or c.startswith('dynamic_') or c.startswith('ohe_')]
    if overlap_cols:
        print(f"Dropping overlapping columns: {overlap_cols}")
        panel = panel.drop(columns=overlap_cols)
    
    panel = panel.sort_values('period_start_dt')
    pdf_df = pdf_df.sort_values('source_date_dt')
    
    print("Executing pandas.merge_asof forward-filling time-join...")
    merged = pd.merge_asof(
        panel, 
        pdf_df, 
        left_on='period_start_dt', 
        right_on='source_date_dt', 
        by='case_number', 
        direction='backward'
    )
    
    # Impute missing strings with Unknown
    for col in ['pdf_requested_zoning', 'pdf_staff_recommended_zoning', 'pdf_approved_zoning']:
        merged[col] = merged[col].fillna('Unknown')
        
    print("Mapping explicit LDC numeric characteristics...")
    # Lookup numeric heights and FARs based on the semantic strings!
    def map_ldc(df, col_prefix):
        txt_col = f'pdf_{col_prefix}_zoning'
        ht_col = f'dynamic_{col_prefix}_ht'
        far_col = f'dynamic_{col_prefix}_far'
        
        metrics = df[txt_col].apply(get_ldc_metrics)
        df[ht_col] = [m[0] for m in metrics]
        df[far_col] = [m[1] for m in metrics]
        return df

    merged = map_ldc(merged, 'requested')
    merged = map_ldc(merged, 'staff_recommended')
    merged = map_ldc(merged, 'approved')
    
    # Override inferred height with EXPLICIT reduced/staff heights if they exist
    # e.g., if staff recommended CS but explicitly said 40 ft (staff_recommends_ht = 40)
    merged['dynamic_staff_recommended_ht'] = merged['pdf_staff_recommends_ht'].combine_first(merged['dynamic_staff_recommended_ht'])
    merged['dynamic_requested_ht'] = merged['pdf_reduced_to_ft'].combine_first(merged['dynamic_requested_ht'])
    
    # OHE Top categories
    top_n = 10
    top_categories = merged['pdf_requested_zoning'].value_counts().index[:top_n].tolist()
    if 'Unknown' in top_categories: top_categories.remove('Unknown')
        
    print(f"One-hot encoding dynamic PDF features: {top_categories}")
    for cat in top_categories:
        merged[f'ohe_req_zoning_{cat}'] = (merged['pdf_requested_zoning'] == cat).astype(int)
        merged[f'ohe_staff_zoning_{cat}'] = (merged['pdf_staff_recommended_zoning'] == cat).astype(int)
        
    # Drop temp cols
    cols_to_drop = ['period_start_dt', 'source_date_dt', 'source_date']
    merged = merged.drop(columns=cols_to_drop)
    
    merged = merged.sort_values(['case_number', 'period_seq'])
    
    print("Saving updated biweekly_panel.csv...")
    merged.to_csv(panel_path, index=False)
    
    case_proof = 'C14-2007-0262'
    proof_df = merged[merged['case_number'] == case_proof][
        ['period_seq', 'period_start', 'pdf_requested_zoning', 'dynamic_requested_ht', 'pdf_staff_recommended_zoning', 'dynamic_staff_recommended_ht']
    ]
    
    # Show periods where stuff changes
    proof_df['lag_req'] = proof_df['pdf_requested_zoning'].shift(1)
    proof_df['lag_staff'] = proof_df['pdf_staff_recommended_zoning'].shift(1)
    jumps = proof_df[(proof_df['pdf_requested_zoning'] != proof_df['lag_req']) | (proof_df['pdf_staff_recommended_zoning'] != proof_df['lag_staff'])]
    
    print(f"\nProof of LDC Metrics + Time-Aware Join (Case {case_proof}):")
    print(jumps[['period_seq', 'period_start', 'pdf_requested_zoning', 'dynamic_requested_ht', 'pdf_staff_recommended_zoning', 'dynamic_staff_recommended_ht']].to_string(index=False))

if __name__ == "__main__":
    engineer_pdf_zoning()
