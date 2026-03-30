"""
build_submission_text_embeddings.py
===================================
Phase 3 Multimodal Architecture Component
Loads the 50+ locally downloaded Austin City Council Backup PDFs.
Parses their raw administrative semantic text into the `sentence-transformers` LLM.
Fuses the 384-dimensional conceptual matrices explicitly onto the final econometric tensor.
"""
import os
import glob
import re
import pandas as pd
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import PyPDF2
except ImportError:
    print("[-] NLP Modules not installed.")
    exit(1)

ROOT = r"C:\Users\dhl\data\thesis\thesis"
WORK_DIR = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data")
CSV_DIR = os.path.join(WORK_DIR, "CSV")
PDF_DIR = os.path.join(WORK_DIR, "PDFs")

INPUT_PATH = os.path.join(CSV_DIR, "submission_grade_icp_matrix.csv")
OUTPUT_PATH = os.path.join(CSV_DIR, "multimodal_submission_tensor.csv")

def extract_pdf_text(filepath):
    text = ""
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
    except Exception as e:
        pass
    return text[:10000]  # Cap context length at 10,000 characters to prevent OOM

def main():
    print("[*] Loading Final Synthesized Econometric Tensor...")
    df = pd.read_csv(INPUT_PATH)
    print(f"    -> Structured Records: {len(df)}")
    
    print("\n[*] Initializing SentenceTransformers Base LLM (all-MiniLM-L6-v2)...")
    # MiniLM gives high performance 384-d semantic bindings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("\n[*] Scanning Local System for Physical Backup Documentation...")
    pdf_files = glob.glob(os.path.join(PDF_DIR, "**", "*.pdf"), recursive=True)
    print(f"    -> Harvested {len(pdf_files)} Raw PDF Files.")
    
    # Dictionary of Case_Number -> Extracted Output Text
    case_texts = {}
    
    for f in pdf_files:
        filename = os.path.basename(f)
        case_match = re.match(r"(C14-[0-9]{4}-[0-9]{4}|C814-[0-9]{4}-[0-9]{4})", filename)
        if case_match:
            case_num = case_match.group(1)
            text = extract_pdf_text(f)
            if len(text) > 50:
                # Merge multiple PDFs for the same Case via string concat
                if case_num in case_texts:
                    case_texts[case_num] += " " + text
                else:
                    case_texts[case_num] = text

    print(f"    -> Bound valid internal semantics to {len(case_texts)} distinct zoning cases.")
    
    # Encode standard missing context
    print("\n[*] Executing GPU/CPU Transformer Inference mappings (384 Dimensions)...")
    default_embedding = model.encode("NO BACKUP DOCUMENTATION AVAILABLE.")
    
    embedding_matrix = []
    
    for _, row in df.iterrows():
        case_id = row['CASE_NUMBER']
        if case_id in case_texts:
            encoded = model.encode(case_texts[case_id])
            embedding_matrix.append(encoded)
        else:
            embedding_matrix.append(default_embedding)
            
    embedding_matrix = np.vstack(embedding_matrix)
    
    print(f"    -> Synthesized ({embedding_matrix.shape[0]}, {embedding_matrix.shape[1]}) Semantic NLP Matrix.")
    
    # Dynamically inject dimensions natively into Pandas dataframe
    for i in range(embedding_matrix.shape[1]):
        df[f'nlp_embed_{i}'] = embedding_matrix[:, i].astype(np.float32)
        
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[+] SUCCESS: Multimodal Semantic File correctly synthesized safely.")
    print(f"    -> Saved Master Integrated Architecture Output to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
