import os
import glob
from pypdf import PdfReader

def convert_pdfs_to_text():
    # Define paths
    base_dir = r"c:\Users\dhl\data\thesis\thesis\Blei-Invariance_Causality-2026Spring\References"
    tex_dir = os.path.join(base_dir, "Text_Converted")
    
    # Create output directory
    os.makedirs(tex_dir, exist_ok=True)
    
    # Find all PDFs
    pdf_files = glob.glob(os.path.join(base_dir, "*.pdf"))
    
    print(f"Found {len(pdf_files)} PDFs to convert.")
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        txt_filename = filename.replace('.pdf', '.txt')
        txt_path = os.path.join(tex_dir, txt_filename)
        
        print(f"Converting {filename}...")
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved to {txt_filename}")
            
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")

if __name__ == "__main__":
    convert_pdfs_to_text()
