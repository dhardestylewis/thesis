import os
import glob

try:
    import pdfplumber
    READER = "pdfplumber"
except ImportError:
    try:
        import fitz
        READER = "fitz"
    except ImportError:
        try:
            import PyPDF2
            READER = "pypdf2"
        except ImportError:
            READER = None

ROOT = r"C:\Users\dhl\data\thesis\thesis\Data\Zoning_Cases\Processed_Data\PDFs"

def main():
    print(f"Checking for downloaded Staff Reports in {ROOT}...")
    # Case insensitive search for Staff Report
    all_pdfs = glob.glob(os.path.join(ROOT, "*.pdf"))
    pdf_files = [p for p in all_pdfs if 'staff' in p.lower() and 'report' in p.lower()]
    
    if not pdf_files:
        print(f"No Staff Reports downloaded yet. Downloaded {len(all_pdfs)} total PDFs.")
        return
        
    print(f"Found {len(pdf_files)} Staff Reports. Selecting the first one: {os.path.basename(pdf_files[0])}")
    target_pdf = pdf_files[0]
    
    if not READER:
        print("No PDF parsing library installed (pdfplumber, PyMuPDF/fitz, or PyPDF2). Please install one.")
        print(f"File is successfully waiting on disk: {target_pdf}")
        return
        
    print(f"\n--- EXTRACTING USING {READER.upper()} ---")
    
    text = ""
    num_pages = 0
    
    try:
        if READER == "pdfplumber":
            with pdfplumber.open(target_pdf) as pdf:
                num_pages = len(pdf.pages)
                for i in range(min(3, num_pages)):
                    text += pdf.pages[i].extract_text() + "\n"
        elif READER == "fitz":
            doc = fitz.open(target_pdf)
            num_pages = len(doc)
            for i in range(min(3, num_pages)):
                text += doc[i].get_text() + "\n"
        elif READER == "pypdf2":
            with open(target_pdf, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                for i in range(min(3, num_pages)):
                    text += reader.pages[i].extract_text() + "\n"
                    
        print(f"Total Pages (The Friction Proxy): {num_pages}")
        
        text_lower = text.lower()
        print("\n--- ML FEATURE KEYWORD HITS (First 3 Pages) ---")
        print(f"TIA Triggered: {'traffic impact analysis' in text_lower or 'tia' in text_lower}")
        print(f"Neighborhood Plan Match: {'neighborhood plan' in text_lower}")
        print(f"Environmental Overlay: {'waterfront' in text_lower or 'springs' in text_lower or 'aquifer' in text_lower}")
        print(f"Conditional Overlay (CO): {'conditional overlay' in text_lower}")
        print(f"Agent Representation: {'agent' in text_lower}")
        
        print("\n--- RAW TEXT PREVIEW (First 600 chars) ---")
        print(text[:600].replace('\n', ' | '))
                
    except Exception as e:
        print(f"Failed to read PDF: {e}")

if __name__ == "__main__":
    main()
