import os
import re
import pandas as pd
import fitz  # PyMuPDF

PDF_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\Protest_Petitions\raw_sample_petition_C241282.pdf"
OUTPUT_CSV = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\bipartite_edges.csv"

def extract_bipartite_edges(pdf_path):
    print(f"Extracting raw text from {os.path.basename(pdf_path)}...")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    edges = []
    current_case = None
    
    # We will iterate through lines to maintain state (which Case Number is active)
    lines = text.split('\n')
    
    # Regex to capture case numbers like C14-2007-0131 or NPA-2015-0015.02
    case_regex = re.compile(r'(C14-\d{4}-\d{4}|NPA-\d{4}-\d{4}(?:\.\d{2})?)')
    
    # Regex for TCAD ID: XX-XXXX-XXXX or XXXXXXXXXX
    tcad_regex = re.compile(r'^(\d{2}-?\d{4}-?\d{4}(?:-?\d{4})?)')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if we hit a Case Number block
        if "Case Number:" in line:
            # The case number is usually on the next line or the same line
            if len(line) > 12:
                match = case_regex.search(line)
                if match:
                    current_case = match.group(1)
            else:
                # Check next line
                if i + 1 < len(lines):
                    match = case_regex.search(lines[i+1])
                    if match:
                        current_case = match.group(1)
        
        # Check if we hit a TCAD ID
        tcad_match = tcad_regex.search(line)
        if tcad_match and current_case:
            tcad_id = tcad_match.group(1)
            
            # The owner name is usually on the next line
            owner_name = "UNKNOWN"
            if i + 1 < len(lines):
                owner_name = lines[i+1].strip()
                
            edges.append({
                "target_case_number": current_case,
                "source_tcad_id": tcad_id,
                "owner_name": owner_name
            })
            
        i += 1
        
    return pd.DataFrame(edges)

print("Starting OCR/Regex Pipeline...")
edges_df = extract_bipartite_edges(PDF_PATH)

print(f"\nExtracted {len(edges_df)} Bipartite Edges!")
print(edges_df.head(10).to_string(index=False))

edges_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved to {OUTPUT_CSV}")
