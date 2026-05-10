import urllib.request
import urllib.error
import random
import time
import re

years = list(range(2007, 2027))
sample_years = random.sample(years, 3)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def get_html(url):
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def get_head_size(url):
    try:
        req = urllib.request.Request(url, headers=headers, method='HEAD')
        with urllib.request.urlopen(req, timeout=15) as response:
            size = response.headers.get('Content-Length')
            if size:
                return int(size)
    except Exception as e:
        pass
    return None

base_url = "https://www.austintexas.gov"

print(f"Randomly selected years for sampling: {sample_years}")

total_meetings_sampled = 0
total_docs_in_sampled_meetings = 0
doc_sizes = []

# First count all meetings in the sampled years
total_meetings_in_sample_years = 0

for year in sample_years:
    if year == 2007:
        url = f"{base_url}/content/archive-council-meetings-held-2007"
    else:
        url = f"{base_url}/council/{year}/{year}_master_index"
        
    print(f"\nFetching index for {year}: {url}")
    html = get_html(url)
    
    # Meeting links typically look like /council/YYYY/YYYYMMDD-something
    meeting_links = re.findall(r'href="(/council/\d{4}/\d{8}[^"]*)"', html)
    meeting_links = list(set(meeting_links))
    print(f"Found {len(meeting_links)} meeting links for {year}.")
    total_meetings_in_sample_years += len(meeting_links)
    
    if len(meeting_links) > 0:
        # Sample up to 5 meetings
        sample_meetings = random.sample(meeting_links, min(5, len(meeting_links)))
        
        for meeting_url in sample_meetings:
            full_meeting_url = base_url + meeting_url
            print(f"  Sampling meeting: {full_meeting_url}")
            m_html = get_html(full_meeting_url)
            
            # Find document links
            doc_links = re.findall(r'document\.cfm\?id=(\d+)', m_html)
            doc_links = list(set(doc_links))
            print(f"    Found {len(doc_links)} documents in this meeting.")
            
            total_meetings_sampled += 1
            total_docs_in_sampled_meetings += len(doc_links)
            
            # Sample document sizes
            sample_docs = random.sample(doc_links, min(3, len(doc_links)))
            for doc_id in sample_docs:
                doc_url = f"https://services.austintexas.gov/edims/document.cfm?id={doc_id}"
                size = get_head_size(doc_url)
                if size:
                    doc_sizes.append(size)
            time.sleep(1) # polite delay

if total_meetings_sampled > 0:
    avg_docs_per_meeting = total_docs_in_sampled_meetings / total_meetings_sampled
    print(f"\nAverage documents per meeting: {avg_docs_per_meeting:.1f}")
else:
    avg_docs_per_meeting = 0

if doc_sizes:
    avg_doc_size = sum(doc_sizes) / len(doc_sizes)
    print(f"Average document size: {avg_doc_size/1024/1024:.2f} MB (based on {len(doc_sizes)} samples)")
else:
    avg_doc_size = 0

avg_meetings_per_year = total_meetings_in_sample_years / len(sample_years) if len(sample_years) > 0 else 0
print(f"Average meetings per year: {avg_meetings_per_year:.1f}")

# Extrapolate for 20 years (2007-2026)
total_years = 20
est_total_meetings = avg_meetings_per_year * total_years
est_total_docs = est_total_meetings * avg_docs_per_meeting
est_total_size_bytes = est_total_docs * avg_doc_size

print("\n--- Extrapolation for 2007-2026 (20 years) ---")
print(f"Estimated total meetings: {est_total_meetings:,.0f}")
print(f"Estimated total documents: {est_total_docs:,.0f}")
print(f"Estimated total dataset size: {est_total_size_bytes / 1024 / 1024 / 1024:.2f} GB")

# Assuming 4.6 seconds average per document (4s sleep + 0.6s download)
download_time_sec = est_total_docs * 4.6
print(f"Estimated time to download (single threaded, 4.6s per doc): {download_time_sec / 60 / 60 / 24:.2f} days")
print(f"Estimated time to download (3 threads, 4.6s per doc): {download_time_sec / 3 / 60 / 60 / 24:.2f} days")
