import pandas as pd
import urllib.request
import fitz  # PyMuPDF
import os
import re

index_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\planning_commission_index.csv"
output_dir = r"c:\Users\dhl\data\Thesis\thesis\Data\Sample_Staff_Reports"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(index_csv)
# Filter for staff reports that are explicitly zoning cases (C14)
df_staff = df[(df['Doc_Text'].str.contains('Staff Report', case=False, na=False)) & (df['Doc_Text'].str.contains('C14-', case=False, na=False))]

# Sample 3
samples = df_staff.sample(3, random_state=42)

for idx, row in samples.iterrows():
    url = row['Doc_URL']
    clean_name = re.sub(r'[^A-Za-z0-9_\-\.]', '_', row['Doc_Text']) + ".pdf"
    file_path = os.path.join(output_dir, clean_name)
    
    print(f"\n[{row['Year']}] Processing: {row['Doc_Text']}")
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            with open(file_path, "wb") as f:
                f.write(response.read())
                
        # Extract text
        doc = fitz.open(file_path)
        
        # Just grab the full text of the first two pages to see what it looks like
        full_text = ""
        for page_num in range(min(2, len(doc))):
            page = doc.load_page(page_num)
            full_text += page.get_text("text") + "\n"
            
        # Search for Zoning lines
        lines = full_text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if 'zoning:' in line_lower or 'zoning from' in line_lower or 'proposed zoning' in line_lower or 'existing zoning' in line_lower:
                # print the line and the next line for context
                context = line.strip()
                if i+1 < len(lines) and lines[i+1].strip() != "":
                    context += " " + lines[i+1].strip()
                print(f"  -> Found: {context}")
                
    except Exception as e:
        print(f"  -> Error processing: {e}")
