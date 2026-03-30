import os
import pandas as pd

ROOT = r"C:\Users\dhl\data\thesis\thesis"
WORK_DIR = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV")

votes_path = os.path.join(WORK_DIR, "scraped_votes_all_years.csv")
features_path = os.path.join(WORK_DIR, "engineered_agenda_features.csv")
output_path = os.path.join(WORK_DIR, "first_draft_icp_covariates.csv")

def build_first_draft():
    print(f"[*] Loading raw structured data...")
    df_votes = pd.read_csv(votes_path)
    df_features = pd.read_csv(features_path)
    
    print(f"    -> Raw Votes: {len(df_votes)} rows")
    print(f"    -> Raw Features: {len(df_features)} rows")

    # 1. Filter out cases that missed the heuristic outcome match
    df_votes = df_votes[df_votes['matched'] == True].copy()
    print(f"[*] Filtered to successfully mapped target vote cases: {len(df_votes)}")

    # 2. De-duplicate features strictly based on exact Case/Date composite keys
    df_features_dedup = df_features.drop_duplicates(subset=['CASE_NUMBER', 'Meeting_Date']).copy()
    print(f"[*] Deduped HTML ML Features matrix from {len(df_features)} -> {len(df_features_dedup)}")

    # 3. Join horizontally
    print(f"[*] Merging the Target Output Variables with the Input Covariates...")
    df_final = pd.merge(
        df_votes, 
        df_features_dedup, 
        on=['CASE_NUMBER', 'Meeting_Date'], 
        how='inner'
    )
    
    # Fill NAs in nay string just as a safety net
    df_final['nay_members'] = df_final['nay_members'].fillna("")
    
    print(f"[+] Final Causal Engineering Matrix: {len(df_final)} exact matches.")

    cols_ordered = [
        'CASE_NUMBER', 'Meeting_Date', 
        'vote_yes', 'vote_no', 'nay_members', 'matched',
        'valid_petition', 'commission_disagree', 'agent', 
        'acreage', 'orig_zoning', 'target_zoning', 'watershed', 'is_npa',
        'agenda_text_raw'
    ]
    # Reorder columns and drop any extranenous artifacts
    existing_cols = [c for c in cols_ordered if c in df_final.columns]
    df_final = df_final[existing_cols]

    # Save artifact
    df_final.to_csv(output_path, index=False)
    print(f"\n[+] SUCCESS: Clean ICP Data Matrix physically saved perfectly to disk at:\n    -> {output_path}")

if __name__ == "__main__":
    build_first_draft()
