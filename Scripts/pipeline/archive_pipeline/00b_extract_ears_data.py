import os
import glob
import zipfile
import pandas as pd
from pathlib import Path

print("==================================================")
print(" 00b: EXTRACTING EARS APPRAISAL DATA FROM GDRIVE  ")
print("==================================================")

BASE = Path(r"c:\Users\dhl\data\Thesis\thesis")
GDRIVE = Path(r"G:\My Drive\protest_petitions_project")
OUT_DIR = BASE / "Data" / "Raw" / "EARS"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Comptroller AJR Format Mapping
# Derived from empirical analysis of 2020/2022 Electronic Appraisal Roll Submissions
COL_MAP = {
    6: "account_number",
    9: "address",
    14: "exemption_flag_hs",
    21: "improvement_sq_ft",
    22: "year_built",
    28: "deed_acreage", # usually sqft, panel normalizes it
    29: "total_market_value",
    30: "land_use_code",
    32: "appraised_value",
    34: "improvement_market_value",
    35: "land_market_value"
}

years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

for year in years:
    print(f"\nProcessing EARS {year}...")
    
    yr_2 = str(year)[-2:]
    all_zips = (glob.glob(str(GDRIVE / f"*{year}*EARS*.zip")) + 
            glob.glob(str(GDRIVE / "**" / f"*{year}*EARS*.zip"), recursive=True) + 
            glob.glob(str(GDRIVE / f"EARS_{year}" / "*.zip")) +
            glob.glob(str(GDRIVE / f"*EARS*{yr_2}*.zip")) +
            glob.glob(str(GDRIVE / f"**" / f"*EARS*{yr_2}*.zip"), recursive=True))
    
    # Filter out false positives (e.g. 2021 having '25' in its month/day)
    zips = []
    for z in set(all_zips):
        n = Path(z).name
        if str(year) in n or n.endswith(f"{yr_2}.zip") or n.endswith(f"{yr_2} (2).zip") or n.endswith(f"{yr_2} (3).zip"):
            zips.append(z)
    
    if not zips:
        print(f"  [!] No zip archive found for {year}.")
        # Try finding a raw CSV just in case
        csvs = glob.glob(str(GDRIVE / f"EARS_{year}" / "**" / "*.csv"), recursive=True)
        if not csvs:
            print(f"  [!] No raw CSV found for {year} either. Skipping.")
            continue
        target_csv = csvs[0]
        needs_unzip = False
    else:
        target_zip = zips[0]
        needs_unzip = True
        print(f"  Found archive: {target_zip}")

    try:
        if needs_unzip:
            with zipfile.ZipFile(target_zip, 'r') as z:
                # Check for nested zips first
                nested_zips = [f for f in z.namelist() if f.endswith('.zip') and 'EARS' in f.upper()]
                
                if nested_zips:
                    inner_zip_name = nested_zips[0]
                    print(f"  Found nested zip: {inner_zip_name}")
                    import io
                    inner_zip_data = z.read(inner_zip_name)
                    with zipfile.ZipFile(io.BytesIO(inner_zip_data), 'r') as inner_z:
                        csv_files = [f for f in inner_z.namelist() if f.endswith('.csv')]
                        if not csv_files:
                            print(f"  [!] No CSV found in nested {inner_zip_name}")
                            continue
                        target_ajr = sorted(csv_files, key=lambda x: inner_z.getinfo(x).file_size, reverse=True)[0]
                        print(f"  Extracting {target_ajr} from nested zip...")
                        with inner_z.open(target_ajr) as f:
                            df = pd.read_csv(f, encoding='latin1', header=None, low_memory=False, usecols=list(COL_MAP.keys()))
                else:
                    # Find the AJR (Appraisal Jurisdiction Record) file
                    ajr_files = [f for f in z.namelist() if 'AJR' in f.upper() and f.endswith('.csv')]
                    if not ajr_files:
                        # Fallback to the largest CSV
                        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                        if not csv_files:
                            print(f"  [!] No CSV found in {target_zip}")
                            continue
                        ajr_files = sorted(csv_files, key=lambda x: z.getinfo(x).file_size, reverse=True)
                    
                    target_ajr = ajr_files[0]
                    print(f"  Extracting {target_ajr}...")
                    
                    with z.open(target_ajr) as f:
                        df = pd.read_csv(f, encoding='latin1', header=None, low_memory=False, usecols=list(COL_MAP.keys()))
        else:
            print(f"  Loading {target_csv}...")
            df = pd.read_csv(target_csv, encoding='latin1', header=None, low_memory=False, usecols=list(COL_MAP.keys()))
            
        # Rename to formal EARS columns
        df = df.rename(columns=COL_MAP)
        
        # Clean up account_number
        df['account_number'] = df['account_number'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
        
        # Keep only Travis County (227) CAD records if multiple counties are mixed
        out_file = OUT_DIR / f"ears_{year}.csv"
        df.to_csv(out_file, index=False)
        print(f"  [+] Saved {len(df):,} records to {out_file.name} ({(os.path.getsize(out_file)/1e6):.1f} MB)")
        
    except Exception as e:
        print(f"  [!] Failed to process {year}: {e}")

print("\nEARS Extraction Complete!")
