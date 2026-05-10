import urllib.request
import time
import random
import re
import csv
import os

years = list(range(2007, 2027))
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}
base_url = "https://www.austintexas.gov"

output_file = os.path.join("..", "Data", "austin_council_meetings_index.csv")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

meetings = []

print(f"Starting scrape of {len(years)} years...")

for i, year in enumerate(years):
    if year == 2007:
        url = f"{base_url}/content/archive-council-meetings-held-2007"
    else:
        url = f"{base_url}/council/{year}/{year}_master_index"
        
    print(f"[{i+1}/{len(years)}] Fetching {year} index...")
    
    if i > 0:
        delay = random.uniform(3.0, 6.0)
        time.sleep(delay)
        
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
            # Find all meeting links. They are inside <a href="...">...</a>
            # Example: <a class="edims" href="/council/2026/20260507-reg"><b>May 7, 2026</b>  Austin City Council Regular Meeting <img ...></a>
            # Note: 2007 uses /content/...
            pattern = r'<a[^>]*href="(/council/\d{4}/[^"]+|/content/[^"]+)"[^>]*>(.*?)</a>'
            matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
            
            count_for_year = 0
            for link, inner_html in matches:
                # Some links might be master index links themselves, filter them out
                if "master_index" in link or "council_index" in link:
                    continue
                
                # Strip HTML tags from inner_html to get clean text
                clean_text = re.sub(r'<[^>]+>', '', inner_html).strip()
                # Remove excessive whitespace
                clean_text = re.sub(r'\s+', ' ', clean_text).replace("&nbsp;", " ").strip()
                
                if not clean_text:
                    continue
                    
                full_url = base_url + link
                
                meetings.append({
                    "Year": year,
                    "Meeting_Text": clean_text,
                    "URL": full_url
                })
                count_for_year += 1
                
            print(f"  -> Found {count_for_year} meetings.")
            
    except Exception as e:
        print(f"Error fetching {year}: {e}")

# Save to CSV
print(f"\nScraping complete. Saving {len(meetings)} records to {output_file}...")
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Year", "Meeting_Text", "URL"])
    writer.writeheader()
    writer.writerows(meetings)

print("Done!")
