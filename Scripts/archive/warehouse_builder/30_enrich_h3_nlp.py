import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT_DIR, "Data")
IN_FILE = os.path.join(DATA, "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
SPEECH_FILE = os.path.join(DATA, "Warehouse_As_Of", "Build", "speech_comment.csv")
OUT_FILE_NLP = os.path.join(DATA, "Warehouse_As_Of", "H3_Filing_Master_NLP.csv")
OUT_FILE_FINAL = os.path.join(DATA, "Warehouse_As_Of", "H3_Pre_Council.csv")

def fold_nlp_transcriptions():
    print("Loading V2 Master Warehouse Baseline...")
    if not os.path.exists(IN_FILE):
        print(f"Error: {IN_FILE} not found.")
        return
    df = pd.read_csv(IN_FILE, low_memory=False)
    
    print("Loading textual structural representation matrix (TF-IDF)...")
    if not os.path.exists(SPEECH_FILE):
        print(f"Error: {SPEECH_FILE} not found. Run transcript ingestion first.")
        return
    speech_df = pd.read_csv(SPEECH_FILE)
    
    # Process the target variables
    speech_df['case_number'] = speech_df['CASE_NUMBER'].astype(str).str.strip().str.upper()
    speech_df.drop(columns=['CASE_NUMBER'], inplace=True)
    
    # Aggregate to case-level to handle multi-parcel filings / multiple hearings
    # For count data, max or sum is fine. We take max for binary flags and mean for tfidf
    agg_funcs = {}
    for col in speech_df.columns:
        if col != 'case_number':
            agg_funcs[col] = 'max' if col == 'has_transcribed_opposition' else 'mean'
            
    speech_grouped = speech_df.groupby('case_number').agg(agg_funcs).reset_index()
    
    df['case_number'] = df['case_number'].astype(str).str.strip().str.upper()
    print(f"Folding {len(speech_grouped)} textual vectors into Master panel...")
    df_h3 = df.merge(speech_grouped, on='case_number', how='left')
    
    # Fill NAs for missing transcripts
    for col in speech_grouped.columns:
        if col != 'case_number':
            df_h3[col] = df_h3[col].fillna(0)
            
    print(f"Exporting H3 Master NLP panel to {OUT_FILE_NLP} and {OUT_FILE_FINAL}...")
    df_h3.to_csv(OUT_FILE_NLP, index=False)
    df_h3.to_csv(OUT_FILE_FINAL, index=False)
    print("Authentic NLP Fold Complete!")

if __name__ == "__main__":
    fold_nlp_transcriptions()
