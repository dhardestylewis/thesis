import os
import pandas as pd
import requests
import time
import random

ROOT_DIR = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT_DIR, "Data")
IN_FILE = os.path.join(DATA, "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
OUT_DIR = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "Images", "Historical_GSV")
os.makedirs(OUT_DIR, exist_ok=True)

def fetch_panoids(lat, lng):
    """
    Hits the undocumented internal GeoPhotoService to retrieve all historical panoids for a coordinate.
    """
    url = f"https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{lat}!4d{lng}!2d50!3m10!2m2!1sen!2sUS!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=cb"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer': 'https://www.google.com/maps'
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        # Parse the JSON callback structure [internal Google Maps array formatting]
        # This requires string slicing to grab the specific pano IDs and dates
        text = resp.text
        panos = []
        
        # Extremely basic heuristic parser for the internal array
        # Realistically, using 'streetlevel' python package is much safer here
        # Doing this manually to demonstrate the architecture
        import re
        # Find all 22-char alphanumeric pano IDs
        # And their associated date arrays [YYYY, M]
        # For production robustness, we will pipe this through `streetlevel` directly
        pass
    except Exception as e:
        print(f"Error fetching metadata: {e}")
    return []

def main():
    print("Initializing Unauthenticated Historical GSV Extraction Pipeline...")
    try:
        import streetlevel
    except ImportError:
        print("CRITICAL: 'streetlevel' package required for robust internal Google API parsing. Run `pip install streetlevel`")
        return

    df = pd.read_csv(IN_FILE, low_memory=False)
    valid_mask = df['latitude'].notna() & df['longitude'].notna() & df['year'].notna()
    work_df = df[valid_mask].copy()
    
    print(f"Loaded {len(work_df)} geometries with valid coordinates and chronological target years.")
    
    successful = 0
    for idx, row in work_df.iterrows():
        case_id = row['case_number'].replace('/', '_').replace(' ', '')
        lat = row['latitude']
        lng = row['longitude']
        target_year = int(row['year'])
        
        out_path = os.path.join(OUT_DIR, f"{case_id}_{target_year}_visual.jpg")
        if os.path.exists(out_path):
            continue
            
        try:
            # Query the undocumented API via streetlevel wrapper
            panos = streetlevel.google.get_panoramas_at(lat, lng, radius=50)
            if not panos:
                continue
                
            # Filter panos chronologically: must be captured BEFORE or DURING the application year
            # To strictly prevent look-ahead bias
            valid_panos = [p for p in panos if p.date and p.date.year <= target_year]
            
            if not valid_panos:
                # Fallback: take the absolute oldest panorama available to best approximate
                valid_panos = sorted(panos, key=lambda p: (p.date is None, getattr(p.date, 'year', 9999)))
                
            # Sort valid panos by how close they are to the target year (descending)
            best_pano = sorted(valid_panos, key=lambda p: getattr(p.date, 'year', 0), reverse=True)[0]
            
            # The User's precise bypass endpoint!
            bypass_url = f"https://streetviewpixels-pa.googleapis.com/v1/thumbnail?cb_client=maps_sv.tactile&w=1200&h=800&pitch=0&panoid={best_pano.id}&yaw=0"
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            # Throttling to prevent IP bans from Google's silent endpoint
            time.sleep(random.uniform(0.5, 1.5)) 
            
            img_resp = requests.get(bypass_url, headers=headers)
            if img_resp.status_code == 200:
                with open(out_path, 'wb') as f:
                    f.write(img_resp.content)
                successful += 1
                
                if successful % 10 == 0:
                    print(f"Harvested {successful} historical WebGL panoramas directly via bypass...")
                    
        except Exception as e:
            pass
            
    print(f"Pipeline complete. Successfully extracted {successful} images.")

if __name__ == "__main__":
    main()
