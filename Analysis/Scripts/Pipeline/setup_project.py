import os
import requests
import shutil
import glob

# ==========================================
# Configuration
# ==========================================

# Base data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Data")

# Directory Scaffold to Create
DIRS_TO_CREATE = [
    f"{DATA_DIR}/Appraisal_Rolls",
    *[f"{DATA_DIR}/Appraisal_Rolls/{year}" for year in range(2018, 2026)],
    f"{DATA_DIR}/Protest_Petitions/Pickles",
    f"{DATA_DIR}/Protest_Petitions/GeoJSON",
    f"{DATA_DIR}/Protest_Petitions/Models",
    f"{DATA_DIR}/Zoning_Cases/Source_Data",
    f"{DATA_DIR}/Zoning_Cases/Processed_Data",
    f"{DATA_DIR}/Zoning_Cases/QC_Logs",
    f"{DATA_DIR}/Documents",
]

# EARS Code Mapping (Technical -> Human Readable)
EARS_MAPPING = {
    "AJR": "Jurisdiction_Tax_Values",
    "ACD": "Category_Details",
    "APL": "Property_Address_Legal",
    "AND": "Owner_Metadata",
    "AUD": "Ag_Timber_Details",
    "ATO": "Top_Taxpayers",
    "AAR": "Arduino_Records", # Example, adjust as needed
}

# Public Data Sources (Austin Open Data)
DATA_URLS = {
    "Zoning_Cases": "https://data.austintexas.gov/api/views/5mps-88a7/rows.csv?accessType=DOWNLOAD",
    "Land_Use_Inventory": "https://data.austintexas.gov/api/views/3k7r-w54d/rows.csv?accessType=DOWNLOAD", # 2012 Land Use
}

# ==========================================
# Functions
# ==========================================

def create_scaffold():
    print("--- 1. Creating Directory Scaffold ---")
    for directory in DIRS_TO_CREATE:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created: {directory}")
        else:
            print(f"Exists: {directory}")

def download_public_data():
    print("\n--- 2. Downloading Public Data (Zoning) ---")
    target_dir = f"{DATA_DIR}/Zoning_Cases/Source_Data"
    
    for name, url in DATA_URLS.items():
        destination = os.path.join(target_dir, f"{name}.csv")
        if os.path.exists(destination):
            print(f"Skipping {name}, file already exists.")
            continue
            
        print(f"Downloading {name} from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved to {destination}")
        except Exception as e:
            print(f"Failed to download {name}: {e}")

def standardize_ears_names():
    print("\n--- 3. Standardizing EARS Filenames ---")
    print("Looking for files in Data/Appraisal_Rolls/YYYY/ ...")
    
    # Iterate through year directories
    for year in range(2018, 2026):
        year_dir = f"{DATA_DIR}/Appraisal_Rolls/{year}"
        if not os.path.exists(year_dir):
            continue
            
        files = os.listdir(year_dir)
        for filename in files:
            # Check for standard EARS patterns
            # Pattern A: 2018_AJR_Records.txt or 2018_0000_AJR.txt
            # We look for the code (AJR, ACD, etc.)
            
            upper_name = filename.upper()
            
            # Skip valid names
            if "EARS_" in upper_name and any(m in upper_name for m in EARS_MAPPING.values()):
                continue

            for code, readable in EARS_MAPPING.items():
                if code in upper_name:
                    # Construct new name: EARS_YYYY_Description.ext
                    ext = os.path.splitext(filename)[1]
                    new_name = f"EARS_{year}_{readable}{ext}"
                    
                    old_path = os.path.join(year_dir, filename)
                    new_path = os.path.join(year_dir, new_name)
                    
                    if old_path != new_path:
                        try:
                            os.rename(old_path, new_path)
                            print(f"Renamed: {filename} -> {new_name}")
                        except OSError as e:
                            print(f"Error renaming {filename}: {e}")
                    break

def main():
    print("Starting Setup Script for Austin Zoning Thesis...")
    create_scaffold()
    download_public_data()
    standardize_ears_names()
    print("\nSetup Complete!")

if __name__ == "__main__":
    main()
