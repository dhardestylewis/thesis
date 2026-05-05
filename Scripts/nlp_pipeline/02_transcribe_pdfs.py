"""
Phase 2: PDF Optical Character Recognition
Iterates through the 10,000+ downloaded PDFs and uses PyMuPDF to extract text.
Outputs the raw textual data to commission_transcripts.csv for NLP parsing.
"""

import fitz
import os
import pandas as pd
import time
import re

pdf_dir = r"c:\Users\dhl\data\Thesis\thesis\Data\Commission_PDFs"
output_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv"

if not os.path.exists(pdf_dir):
    print(f"Directory {pdf_dir} not found.")
    exit(1)

pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
transcripts = []

start_time = time.time()
print(f"Starting transcription of {len(pdf_files)} Commission PDFs...", flush=True)

processed = 0
errors = 0

for filename in pdf_files:
    pdf_path = os.path.join(pdf_dir, filename)
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + " "
            
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        transcripts.append({
            'Filename': filename,
            'Raw_Text': full_text
        })
        processed += 1
        
    except Exception as e:
        errors += 1
        print(f"Error reading {filename}: {e}", flush=True)
        
    if processed % 100 == 0:
        print(f"Processed {processed}/{len(pdf_files)} PDFs...", flush=True)

df = pd.DataFrame(transcripts)
df.to_csv(output_csv, index=False)

elapsed = time.time() - start_time
print(f"\nFinished transcribing {processed} PDFs in {elapsed:.2f} seconds.", flush=True)
print(f"Total Errors: {errors}", flush=True)
print(f"Saved {len(df)} transcripts to {output_csv}", flush=True)
