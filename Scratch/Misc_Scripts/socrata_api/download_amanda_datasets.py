import os
import urllib.request
import pandas as pd

datasets = {
    'Plan_Review_Cases': 'n8ck-xkda',
    'Site_Plan_Cases': '2u4n-hmgw',
    'Issued_Building_Permits': 'quv8-5ckq'
}

out_dir = r'C:\Users\dhl\data\Thesis\thesis\Data\CoA_Open_Data'
os.makedirs(out_dir, exist_ok=True)

for name, id in datasets.items():
    out_path = os.path.join(out_dir, f"{name}.csv")
    print(f"Downloading {name} ({id}) to {out_path}...")
    
    # We use the Socrata CSV endpoint directly, grabbing up to 200,000 rows.
    # Note: For production use with huge datasets, sodapy with pagination is safer,
    # but the direct CSV export is fastest if it's less than limits.
    url = f"https://data.austintexas.gov/api/views/{id}/rows.csv?accessType=DOWNLOAD"
    try:
        req = urllib.request.Request(url, headers={'X-App-Token': 'lx2R84KkVNVYLjYaihYyaksbw'})
        with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
            out_file.write(response.read())
        
        # Verify file
        df = pd.read_csv(out_path, low_memory=False)
        print(f"  Successfully downloaded {len(df)} rows and {len(df.columns)} columns.")
    except Exception as e:
        print(f"  Failed to download {name}: {e}")
