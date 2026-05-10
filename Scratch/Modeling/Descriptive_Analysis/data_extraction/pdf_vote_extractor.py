import os
import re
import glob
import pandas as pd
import pdfplumber
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

DATA_DIR = r"C:\Users\dhl\data\Thesis\thesis\Data"
OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"

def parse_pdf(pdf_path):
    """
    Extracts text from a PDF, locates zoning case numbers (C14-XXXX-XXXX), 
    and searches the surrounding text block for explicit X-Y vote margins.
    """
    case_pattern = re.compile(r'(C14-\d{4}-\d{4}(?:\.\d+)?)', re.IGNORECASE)
    vote_pattern = re.compile(r'\b(\d{1,2})-(\d{1,2})\s*vote\b', re.IGNORECASE)
    
    extracted = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            # Find all case numbers
            cases_found = list(set(case_pattern.findall(full_text)))
            
            for case in cases_found:
                # Find the index of the case
                # Look at a window of 1000 characters after the case number mention
                for match in re.finditer(re.escape(case), full_text, re.IGNORECASE):
                    start_idx = match.end()
                    window = full_text[start_idx:start_idx+1500]
                    
                    vote_match = vote_pattern.search(window)
                    if vote_match:
                        yea = int(vote_match.group(1))
                        nay = int(vote_match.group(2))
                        
                        extracted.append({
                            "source_file": os.path.basename(pdf_path),
                            "case_number": case.upper(),
                            "yea_votes": yea,
                            "nay_votes": nay,
                            "total_votes": yea + nay,
                            "margin": yea - nay,
                            "vote_type": "Council" if "Council" in pdf_path else "Commission"
                        })
                        # Break out after finding the first vote block for this case in this document
                        break
                        
    except Exception as e:
        # Ignore malformed PDFs
        pass
        
    return extracted

def main():
    print("[1/3] Locating PDF Transcripts...")
    council_pdfs = glob.glob(os.path.join(DATA_DIR, "Council_Minutes_PDFs", "*.pdf"))
    commission_pdfs = glob.glob(os.path.join(DATA_DIR, "Commission_PDFs", "*Minutes*.pdf"))
    
    all_pdfs = council_pdfs + commission_pdfs
    print(f"  > Found {len(council_pdfs)} Council Minutes")
    print(f"  > Found {len(commission_pdfs)} Commission Minutes")
    print(f"  > Total PDFs to process: {len(all_pdfs)}")
    
    print("\n[2/3] Extracting Vote Margins via Multiprocessing (This will take a few minutes)...")
    all_votes = []
    
    # Use max cores minus 1 for stability
    cores = max(1, mp.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = {executor.submit(parse_pdf, pdf): pdf for pdf in all_pdfs}
        
        for future in tqdm(as_completed(futures), total=len(all_pdfs), desc="Parsing PDFs"):
            res = future.result()
            if res:
                all_votes.extend(res)
                
    print("\n[3/3] Saving Extracted Margins...")
    if len(all_votes) > 0:
        df = pd.DataFrame(all_votes)
        
        # Deduplicate: if a case is voted on multiple times in the SAME meeting, take the final vote
        # For simplicity across files, we will keep all records, but sort them
        df = df.sort_values(by=["case_number", "source_file"]).drop_duplicates()
        
        out_csv = os.path.join(DATA_DIR, "interim", "engineered_vote_margins.csv")
        df.to_csv(out_csv, index=False)
        print(f"  > Successfully extracted {len(df)} discrete vote events!")
        print(f"  > Saved to {out_csv}")
    else:
        print("  > No votes found.")

if __name__ == "__main__":
    main()
