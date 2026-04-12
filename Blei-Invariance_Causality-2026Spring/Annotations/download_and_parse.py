import urllib.request
import json
import os
from pypdf import PdfReader
import re

target_dir = r"C:\Users\dhl\data\thesis\thesis\Blei-Invariance_Causality-2026Spring\Annotations\Raw_Text"

headers = {'User-Agent': 'mailto:dhl2147@columbia.edu'}

# Pre-selected priority papers based on JMLR bounds found earlier
# We're searching crossref specifically for these exact titles to get the URL
titles_to_fetch = [
    "Causal Domain Generalization",
    "The synthetic control method",
    "Causal Invariance in Reasoning and Learning"
]

def clean_filename(title):
    return re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')

for search_title in titles_to_fetch:
    print(f"Fetching metadata for: {search_title}")
    encoded_query = urllib.parse.quote(search_title)
    url = f"https://api.crossref.org/works?query.container-title=Journal+of+Machine+Learning+Research&query.title={encoded_query}&select=title,URL,link&rows=1"
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            items = data.get('message', {}).get('items', [])
            
            if items:
                item = items[0]
                actual_title = item.get('title', ['Unknown'])[0]
                
                # Try to get direct PDF link from the crossref 'link' array
                pdf_url = None
                links = item.get('link', [])
                for link in links:
                    if link.get('content-type') == 'application/pdf':
                        pdf_url = link.get('URL')
                
                # If no direct PDF link in metadata, we might have to construct it if it's JMLR
                if not pdf_url:
                    base_url = item.get('URL', '')
                    # JMLR URLs roughly look like https://jmlr.org/papers/v24/22-0001.html
                    # PDF is usually https://jmlr.org/papers/volume24/22-0001/22-0001.pdf
                    if "jmlr.org/papers/v" in base_url or "jmlr.org/papers/volume" in base_url:
                        # Best effort heuristic extraction, fallback to doing it manually later if needed
                        print(f"  No direct PDF in Crossref, will need manual download for {actual_title}")
                        continue
                        
                if pdf_url:
                    print(f"  Found PDF URL: {pdf_url}")
                    # Download PDF
                    pdf_filename = clean_filename(actual_title) + ".pdf"
                    txt_filename = clean_filename(actual_title) + ".txt"
                    pdf_path = os.path.join(target_dir, pdf_filename)
                    txt_path = os.path.join(target_dir, txt_filename)
                    
                    try:
                        urllib.request.urlretrieve(pdf_url, pdf_path)
                        print(f"  Downloaded to {pdf_path}")
                        
                        # Parse Text
                        reader = PdfReader(pdf_path)
                        full_text = ""
                        for page in reader.pages:
                            full_text += page.extract_text() + "\n"
                            
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(full_text)
                        
                        print(f"  Extracted text to {txt_path}")
                        
                    except Exception as e:
                        print(f"  Error downloading or parsing PDF: {e}")
                        pass
            else:
                print(f"  No exact match found.")
                
    except Exception as e:
        print(f"Error querying crossref: {e}")
