import urllib.request
import re
import csv
import time
import os

years = list(range(2007, 2027))
headers = {"User-Agent": "Mozilla/5.0"}
output_file = r"c:\Users\dhl\data\Thesis\thesis\Data\planning_commission_index.csv"
records = []

def clean_html(raw_html):
    return re.sub('<.*?>', '', raw_html).replace("&nbsp;", " ").strip()

for year in reversed(years):
    page = 1
    while page <= 10:  # Hard limit
        if year == 2026:
            url = f"https://www.austintexas.gov/boards-commissions/meetings/40_1" if page == 1 else f"https://www.austintexas.gov/boards-commissions/meetings/2026_40_{page}"
        else:
            url = f"https://www.austintexas.gov/boards-commissions/meetings/{year}_40_{page}"
            
        print(f"Fetching {url}...", flush=True)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Error on {url}: {e}", flush=True)
            break
            
        if "Page not found" in html or "no meetings" in html.lower():
            print("  -> Page not found or empty.", flush=True)
            break
            
        meeting_chunks = html.split('<div class="bcic_mtgdate">')[1:]
        if not meeting_chunks:
            print("  -> No meeting blocks found.", flush=True)
            break
            
        docs_found = 0
        for chunk in meeting_chunks:
            date_match = re.match(r'([^<]+)</div>', chunk)
            mtg_date = date_match.group(1).strip() if date_match else "Unknown"
            
            type_match = re.search(r'<div class="bcic_mtgtype">(.*?)</div>', chunk)
            mtg_type = type_match.group(1).strip() if type_match else "Unknown"
            
            docs = re.findall(r'<div class="bcic_doc">(.*?)</div>', chunk, re.IGNORECASE | re.DOTALL)
            for doc_html in docs:
                id_match = re.search(r'document\.cfm\?id=(\d+)', doc_html, re.IGNORECASE)
                if id_match:
                    doc_id = id_match.group(1)
                    doc_url = f"https://services.austintexas.gov/edims/document.cfm?id={doc_id}"
                    
                    records.append({
                        "Year": year,
                        "Meeting_Date": mtg_date,
                        "Meeting_Type": mtg_type,
                        "Doc_ID": doc_id,
                        "Doc_Text": clean_html(doc_html),
                        "Doc_URL": doc_url
                    })
                    docs_found += 1
                    
        print(f"  -> Found {docs_found} docs on page {page}.", flush=True)
        
        # Check if next page link actually exists
        next_page_str = f'href="/boards-commissions/meetings/{year}_40_{page+1}"'
        next_page_str2 = f'href="/boards-commissions/meetings/40_{page+1}"'
        if next_page_str not in html and next_page_str2 not in html:
            break
            
        page += 1
        time.sleep(1.5)
        
    time.sleep(1.5)

print(f"Found {len(records)} documents total.", flush=True)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Year", "Meeting_Date", "Meeting_Type", "Doc_ID", "Doc_Text", "Doc_URL"])
    writer.writeheader()
    writer.writerows(records)
print("Saved to CSV.", flush=True)
