import os
import re
import pandas as pd

ROOT = r"C:\Users\dhl\data\thesis\thesis"
IN_PATH = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV", "scraped_agenda_text_embeddings.csv")
OUT_PATH = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV", "engineered_agenda_features.csv")

def parse_features(text, case_num):
    text = str(text)
    if text == "nan" or not text.strip():
        return pd.Series([None]*8)
    
    valid_petition = int(bool(re.search(r'valid petition', text, re.IGNORECASE)))
    
    disagree = 0
    if ('Staff Recommendation:' in text and 'Commission Recommendation:' in text):
        staff = re.search(r'Staff Recommendation:(.*?)(?=Planning|Zoning|Board|$)', text, re.IGNORECASE|re.DOTALL)
        comm = re.search(r'Commission Recommendation:(.*?)(?=First Reading|Vote|Owner|$)', text, re.IGNORECASE|re.DOTALL)
        if staff and comm:
            if staff.group(1).strip() != comm.group(1).strip():
                disagree = 1
                
    agent_match = re.search(r'Agent:\s*([^\.]+)', text)
    agent = agent_match.group(1).strip() if agent_match else None
    
    watershed_match = re.search(r'\(([^)]*[Ww]atershed[^)]*)\)', text)
    watershed = watershed_match.group(1).strip() if watershed_match else None
    
    is_npa = int(str(case_num).upper().startswith("NPA"))
    
    acreage_match = re.search(r'(\d+\.?\d*)\s*acres?', text, re.IGNORECASE)
    acreage = float(acreage_match.group(1)) if acreage_match else None
    
    orig_zoning = None
    target_zoning = None
    z_match = re.search(r'from\s+(?:.*?\()?([A-Z0-9-]{2,10})\)?\s*(?:district zoning|to)', text)
    if z_match:
        orig_zoning = z_match.group(1)
        
    t_match = re.search(r'to\s+(?:.*?\()?([A-Z0-9-]{2,10})\)?\s*(?:district|combining)', text)
    if t_match:
        target_zoning = t_match.group(1)
        
    return pd.Series([
        valid_petition, disagree, agent, watershed, 
        is_npa, acreage, orig_zoning, target_zoning
    ])

def main():
    print(f"Loading {IN_PATH}...")
    df = pd.read_csv(IN_PATH)
    
    df_valid = df[df['agenda_text_raw'].notna() & (df['agenda_text_raw'] != "")]
    total = len(df_valid)
    print(f"Engineering covariates via Regex for {total} cases with valid text...")
    
    cols = ['valid_petition', 'commission_disagree', 'agent', 'watershed', 
            'is_npa', 'acreage', 'orig_zoning', 'target_zoning']
    df_valid[cols] = df_valid.apply(lambda row: parse_features(row['agenda_text_raw'], row['CASE_NUMBER']), axis=1)
    
    print("\nExtraction Success Rates (from valid text blocks):")
    for col in cols:
        val_count = df_valid[col].notna().sum()
        pct = (val_count / total) * 100
        print(f"  {col}: {pct:.1f}% ({val_count}/{total})")
        
        if col in ['valid_petition', 'commission_disagree']:
            positive = df_valid[col].sum()
            print(f"      -> {positive} cases triggered Positive (1)")
            
    df = df.merge(df_valid[['CASE_NUMBER'] + cols], on='CASE_NUMBER', how='left')
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved engineered features to {OUT_PATH}")

if __name__ == "__main__":
    main()
