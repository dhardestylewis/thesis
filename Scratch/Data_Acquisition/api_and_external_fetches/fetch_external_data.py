import os
import pandas as pd
import urllib.request
import zipfile
import io

BASE_DIR = r"c:\Users\dhl\data\Thesis\thesis\Data\Panel"
MACRO_DIR = os.path.join(BASE_DIR, "macro")
FEMA_DIR = os.path.join(BASE_DIR, "fema")
os.makedirs(MACRO_DIR, exist_ok=True)
os.makedirs(FEMA_DIR, exist_ok=True)

print("1. Fetching FRED Macro Data (High-Frequency)...")
series = {
    "MORTGAGE30US": "mortgage_rate_30yr",
    "FEDFUNDS": "fed_funds_rate",
    "DGS10": "treasury_10yr_yield",
    "AUST448URN": "local_unemployment_rate" # Austin MSA unemployment
}

dfs = []
for sid, name in series.items():
    print(f"   Downloading {sid} ({name})...")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2000-01-01"
    try:
        df = pd.read_csv(url, na_values=["."])
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df[name] = pd.to_numeric(df[sid], errors="coerce")
        df = df[["observation_date", name]].dropna()
        df = df.sort_values("observation_date")
        dfs.append(df)
    except Exception as e:
        print(f"   Failed to fetch {sid}: {e}")

if dfs:
    # Outer join all series on observation_date
    fred_out = dfs[0]
    for df in dfs[1:]:
        fred_out = fred_out.merge(df, on="observation_date", how="outer")
    fred_out = fred_out.sort_values("observation_date")
    # Forward fill to handle daily vs weekly vs monthly mismatches
    fred_out = fred_out.ffill()
    
    fred_path = os.path.join(MACRO_DIR, "fred_timeseries.csv")
    fred_out.to_csv(fred_path, index=False)
    print(f"   Saved {fred_path} with shape {fred_out.shape}")


print("\n2. Fetching FEMA National Risk Index (Travis County subset)...")
fema_url = "https://hazards.fema.gov/nri/Content/StaticDocuments/DataDownload//NRI_Table_CensusTracts/NRI_Table_CensusTracts.zip"
try:
    print(f"   Downloading NRI from {fema_url}...")
    req = urllib.request.Request(fema_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            with z.open(z.namelist()[0]) as f:
                nri = pd.read_csv(f, low_memory=False)
    
    # Filter for Travis County, Texas (FIPS 48453)
    travis = nri[(nri["STATEABBR"] == "TX") & (nri["COUNTY"] == "Travis")].copy()
    
    keep_cols = [
        "TRACTFIPS", "RISK_SCORE", "RISK_RATNG", "EAL_SCORE", "EAL_VALT",
        "SOVI_SCORE", "SOVI_RATNG", "RESL_SCORE", "RESL_RATNG",
        "HRCN_RISKS", "CFLD_RISKS", "WFIR_RISKS", "HWAV_RISKS"
    ]
    keep_cols = [c for c in keep_cols if c in travis.columns]
    travis = travis[keep_cols]
    travis = travis.rename(columns={"TRACTFIPS": "census_tract"})
    
    fema_path = os.path.join(FEMA_DIR, "travis_nri.csv")
    travis.to_csv(fema_path, index=False)
    print(f"   Saved {fema_path} with shape {travis.shape}")
except Exception as e:
    print(f"   Failed to fetch/process FEMA NRI: {e}")

print("\nDone.")
