import pandas as pd
import urllib.request
import fitz
import os
import re
import time

index_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\planning_commission_index.csv"
output_dir = r"c:\Users\dhl\data\Thesis\thesis\Data\Sample_Staff_Reports"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(index_csv)
df_staff = df[(df['Doc_Text'].str.contains('Staff Report', case=False, na=False)) & (df['Doc_Text'].str.contains('C14-', case=False, na=False))]

# Benchmark 10
samples = df_staff.sample(10, random_state=123)

start_time = time.time()
success_count = 0

for idx, row in samples.iterrows():
    url = row['Doc_URL']
    clean_name = re.sub(r'[^A-Za-z0-9_\-\.]', '_', row['Doc_Text']) + ".pdf"
    file_path = os.path.join(output_dir, clean_name)
    
    try:
        # Only download if we don't have it
        if not os.path.exists(file_path):
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as response:
                with open(file_path, "wb") as f:
                    f.write(response.read())
                    
        # Extract text
        doc = fitz.open(file_path)
        full_text = ""
        for page_num in range(min(2, len(doc))):
            page = doc.load_page(page_num)
            full_text += page.get_text("text") + "\n"
            
        success_count += 1
    except Exception as e:
        pass

end_time = time.time()
elapsed = end_time - start_time
print(f"Processed {success_count} PDFs in {elapsed:.2f} seconds.")
print(f"Average time per PDF: {elapsed/max(1, success_count):.2f} seconds.")
