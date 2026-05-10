"""
Phase 2b: Council PDF Optical Character Recognition
Iterates through the downloaded Council Minutes PDFs and uses PyMuPDF to extract text.
Outputs the raw textual data to council_transcripts.csv for NLP parsing.
"""

import fitz
import os
import pandas as pd
import time
import re

def main():
    pdf_dirs = [
        r"C:\Users\dhl\data\Thesis\thesis\Data\Council_PDFs",
        r"C:\Users\dhl\data\Thesis\thesis\Data\Council_Backups"
    ]
    output_csv = r"C:\Users\dhl\data\Thesis\thesis\Data\interim\council_transcripts.csv"

    pdf_files = []
    for d in pdf_dirs:
        if os.path.exists(d):
            pdf_files.extend([(d, f) for f in os.listdir(d) if f.endswith('.pdf')])

    transcripts = []

    start_time = time.time()
    print(f"Starting transcription of {len(pdf_files)} Council PDFs...", flush=True)

    processed = 0
    errors = 0
    
    case_pattern = re.compile(r'[Cc]8?14-\d{4}-\d{4}(?:\.\d{2})?')

    for dir_path, filename in pdf_files:
        pdf_path = os.path.join(dir_path, filename)
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + " "
                
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            # Extract case number from filename first
            case_match = case_pattern.search(filename)
            if not case_match:
                # Fallback to early text
                case_match = case_pattern.search(full_text[:4000])
                
            case_num = case_match.group(0).upper() if case_match else "UNKNOWN"
            
            transcripts.append({
                'Filename': filename,
                'Case_Number': case_num,
                'Vote_Transcript': full_text
            })
            processed += 1
            
        except Exception as e:
            errors += 1
            print(f"Error reading {filename}: {e}", flush=True)
            
        if processed % 50 == 0:
            print(f"Processed {processed}/{len(pdf_files)} PDFs...", flush=True)

    df = pd.DataFrame(transcripts)
    df.to_csv(output_csv, index=False)

    elapsed = time.time() - start_time
    print(f"\nFinished transcribing {processed} PDFs in {elapsed:.2f} seconds.", flush=True)
    print(f"Total Errors: {errors}", flush=True)
    print(f"Saved {len(df)} transcripts to {output_csv}", flush=True)

if __name__ == '__main__':
    main()
