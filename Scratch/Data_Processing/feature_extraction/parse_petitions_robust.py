import os
import re
import pandas as pd
import fitz  # PyMuPDF

PDF_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\Protest_Petitions\raw_sample_petition_C241282.pdf"
OUTPUT_CSV = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\recovered_petitions.csv"

def extract_modern_edges(pdf_path):
    print(f"Extracting raw text from {os.path.basename(pdf_path)}...")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    edges = []
    current_case = None
    
    lines = text.split('\n')
    
    # Matches old format C14-2008-0001, modern C14-2021-0037, historic C14H-2018-0084, etc.
    case_regex = re.compile(r'((?:C14|NPA|C814|C14H)[A-Za-z0-9\-]*20[0-2]\d[A-Za-z0-9\-\.]*)', re.IGNORECASE)
    
    # Matches 10-digit TCAD (0208060108) or dashed TCAD (02-0906-0502)
    tcad_regex = re.compile(r'^(\d{10}|\d{2}-?\d{4}-?\d{4}(?:-?\d{4})?)')
    
    # Matches percentage (e.g. 10.60%)
    pct_regex = re.compile(r'(\d+\.\d+)%$')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for case numbers anywhere in the line
        case_match = case_regex.search(line)
        if case_match:
            current_case = case_match.group(1).upper()
        
        # Check if we hit a TCAD ID
        tcad_match = tcad_regex.search(line)
        if tcad_match and current_case:
            tcad_id = tcad_match.group(1)
            
            # Scan ahead a few lines to grab the area percentage
            area_pct = 0.0
            signed = 1
            for offset in range(1, 8):
                if i + offset < len(lines):
                    lookahead = lines[i+offset].strip()
                    pct_match = pct_regex.search(lookahead)
                    if pct_match:
                        area_pct = float(pct_match.group(1))
                        break
            
            edges.append({
                "case_number": current_case,
                "tcad_id": tcad_id,
                "area_pct": area_pct,
                "signed": signed
            })
            
        i += 1
        
    return pd.DataFrame(edges)

print("Starting Robust OCR/Regex Pipeline...")
edges_df = extract_modern_edges(PDF_PATH)

print(f"\nExtracted {len(edges_df)} Bipartite Edges!")

# Clean and dedup
edges_df = edges_df.drop_duplicates()
print(f"Unique Cases Recovered: {edges_df['case_number'].nunique()}")

edges_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved to {OUTPUT_CSV}")
