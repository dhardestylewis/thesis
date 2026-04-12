import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data")
AGENDA_TEXT_CSV = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "CSV", "scraped_agenda_text_embeddings.csv")
OUT_EMBEDDINGS = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "CSV", "agenda_tfidf_embeddings.csv")

def generate_embeddings():
    print(f"Loading raw textual agenda records from {AGENDA_TEXT_CSV}...")
    df = pd.read_csv(AGENDA_TEXT_CSV)
    
    # Ensure uniform case mapping
    df['case_number'] = df['CASE_NUMBER'].astype(str).str.strip().str.upper()
    df['text'] = df['agenda_text_raw'].fillna("").astype(str)
    
    # Drop empty records
    df = df[df['text'].str.len() > 10].copy()
    
    print(f"Extracting TF-IDF NLP semantic features for {len(df)} agendas...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df['text'])
    
    print("Compressing text dimensions via Truncated SVD (PCA for sparse matrices)...")
    N_COMPONENTS = min(20, tfidf_matrix.shape[1]-1)
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    embed_matrix = svd.fit_transform(tfidf_matrix)
    
    # Structure output dataset
    embed_df = pd.DataFrame(embed_matrix, columns=[f'nlp_svd_{i}' for i in range(N_COMPONENTS)])
    embed_df['case_number'] = df['case_number'].values
    
    # Reorder
    cols = ['case_number'] + [col for col in embed_df.columns if col != 'case_number']
    embed_df = embed_df[cols]
    
    embed_df.to_csv(OUT_EMBEDDINGS, index=False)
    print(f"Generated {N_COMPONENTS}-dimensional NLP representations. Saved to {OUT_EMBEDDINGS}.")

if __name__ == '__main__':
    generate_embeddings()
