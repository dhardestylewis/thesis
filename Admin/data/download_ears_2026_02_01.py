import os
import requests
import zipfile
import io

# EARS Download Links from TCAD Public Information
DOWNLOAD_LINKS = {
    "2025": "https://traviscad.org/wp-content/largefiles/227EARS090425.zip",
    "2024": "https://traviscad.org/wp-content/largefiles/227EARS082824%20%282%29.zip",
    "2023": "https://traviscad.org/wp-content/largefiles/227EARS082923%20%282%29.zip"
}

BASE_PATH = r"c:\Users\dhl\data\thesis\thesis\Data\Appraisal_Rolls"

def download_and_extract(year, url):
    target_dir = os.path.join(BASE_PATH, year)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")
    
    print(f"--- Starting Download for {year} ---")
    print(f"URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        # Stream the download
        response = requests.get(url, stream=True, timeout=120, headers=headers)
        response.raise_for_status()
        
        # Use io.BytesIO to handle the zip in memory if not too large, 
        # but for very large files, it's safer to download to disk first.
        # Given TCAD zips are usually ~150-200MB, we'll download to disk.
        temp_zip = os.path.join(target_dir, f"temp_{year}.zip")
        
        with open(temp_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"SUCCESS: Download complete for {year}. Extracting...")
        
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        
        print(f"SUCCESS: Extraction complete for {year}.")
        
        # Cleanup
        os.remove(temp_zip)
        print(f"Cleaned up temp zip for {year}.")
        
    except Exception as e:
        print(f"ERROR: during {year} processing: {e}")

def main():
    print("Initiating Batch EARS Download (2023-2025)...")
    for year, url in DOWNLOAD_LINKS.items():
        download_and_extract(year, url)
    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()
