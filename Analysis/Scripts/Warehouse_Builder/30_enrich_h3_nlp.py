import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT_DIR, "Data")
IN_FILE = os.path.join(DATA, "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv") # Upgrade to OSMnx later if finished
QUEUE_PATH = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "CSV", "transcription_queue_full.csv")
OUT_FILE = os.path.join(DATA, "Warehouse_As_Of", "H3_Filing_Master_NLP.csv")

def fold_nlp_transcriptions():
    print("Loading V2 Master Warehouse...")
    df = pd.read_csv(IN_FILE, low_memory=False)
    
    print("Loading active NLP Transcription Queue...")
    queue_df = pd.read_csv(QUEUE_PATH)
    
    # Process the queue target variables
    queue_df['case_number'] = queue_df['CASE_NUMBER'].astype(str).str.strip().str.upper()
    
    # Example proxy metrics for H3 (what we have right now before full semantic parsing)
    if 'Transcription_Status' in queue_df.columns:
        queue_df['has_audio_record'] = (queue_df['Transcription_Status'] == 'Complete').astype(int)
    
    # Aggregate to case-level to handle multi-parcel filings
    queue_grouped = queue_df.groupby('case_number').agg(
        has_audio_record=('has_audio_record', 'max') if 'has_audio_record' in queue_df.columns else ('case_number', 'count')
    ).reset_index()
    
    print(f"Folding {len(queue_grouped)} transcription status vectors into Master panel...")
    df_h3 = df.merge(queue_grouped, on='case_number', how='left')
    df_h3['has_audio_record'] = df_h3['has_audio_record'].fillna(0).astype(int)
    
    print(f"Exporting H3 Master NLP panel to {OUT_FILE}...")
    df_h3.to_csv(OUT_FILE, index=False)
    print("Intermediate NLP Fold Complete.")

if __name__ == "__main__":
    fold_nlp_transcriptions()
