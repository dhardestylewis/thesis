import pandas as pd
import fitz
import os
import re
import time

agendas_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\council_agendas_missing_cases.csv"
minutes_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\council_minutes_index.csv"
pdf_dir = r"c:\Users\dhl\data\Thesis\thesis\Data\Council_Minutes_PDFs"
output_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\zoning_cases_with_council_votes.csv"

# Load the mappings
df_cases = pd.read_csv(agendas_csv)
df_minutes = pd.read_csv(minutes_csv)

# Merge to get the Doc_ID for each case's meeting
df_merged = pd.merge(df_cases, df_minutes[['Meeting_URL', 'Doc_ID']], on='Meeting_URL', how='inner')

extracted_votes = []
grouped = df_merged.groupby(['Doc_ID', 'Year'])

print(f"Starting extraction for {len(df_merged)} missing case appearances across {len(grouped)} Minutes PDFs...", flush=True)
start_time = time.time()
processed_pdfs = 0

for (doc_id, year), group in grouped:
    pdf_filename = f"{year}_{int(doc_id)}_Minutes.pdf"
    pdf_path = os.path.join(pdf_dir, pdf_filename)
    
    if not os.path.exists(pdf_path):
        continue
        
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + " "
        
        full_text = re.sub(r'\s+', ' ', full_text)
        
        for _, row in group.iterrows():
            case_num = row['Case_Number']
            safe_case = re.escape(case_num)
            
            # Find the case and extract the 50 chars before and 1000 chars after
            match = re.search(f"(.{{0,50}}{safe_case}.{{0,1000}})", full_text, re.IGNORECASE)
            
            if match:
                extracted_text = match.group(1).strip()
                extracted_votes.append({
                    'Case_Number': case_num,
                    'Meeting_Date': row['Meeting_Date'],
                    'Year': year,
                    'Vote_Transcript': extracted_text
                })
                
        processed_pdfs += 1
        if processed_pdfs % 50 == 0:
            print(f"Processed {processed_pdfs}/{len(grouped)} PDFs...", flush=True)
                
    except Exception as e:
        print(f"Error processing {pdf_filename}: {e}", flush=True)

# Append to existing CSV
df_new = pd.DataFrame(extracted_votes)
df_new = df_new.drop_duplicates()

if os.path.exists(output_csv):
    df_existing = pd.read_csv(output_csv)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates()
    df_combined.to_csv(output_csv, index=False)
    print(f"\nExtraction complete! Appended {len(df_new)} new vote transcripts.")
    print(f"Total vote transcripts now: {len(df_combined)}", flush=True)
else:
    df_new.to_csv(output_csv, index=False)
    print(f"\nExtraction complete! Saved {len(df_new)} vote transcripts.")
    
elapsed = time.time() - start_time
print(f"Finished in {elapsed:.2f} seconds.", flush=True)
