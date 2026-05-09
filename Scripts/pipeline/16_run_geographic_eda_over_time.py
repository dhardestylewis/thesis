import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

ROOT_DIR = Path(r"c:\Users\dhl\data\Thesis\thesis")
REGISTRY_DIR = ROOT_DIR / "registries"
ARTIFACTS_DIR = Path(r"c:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts")
PRIMARY_LABEL_VERSION = "label_v1_reconstructed_threshold_crossing"
WAREHOUSE_MASTER = ROOT_DIR / "Data" / "Warehouse_As_Of" / "canonical" / "H0_Filing_Master_Enriched_v2_OmniLagged.csv"

def main():
    print("1. Loading Data for Geographic EDA...")
    labels = pd.read_parquet(REGISTRY_DIR / "label_registry.parquet")
    labels = labels[labels["label_version"] == PRIMARY_LABEL_VERSION]
    
    # Load cases with coordinates
    master = pd.read_csv(WAREHOUSE_MASTER)
    master["case_id"] = master["case_number"].astype(str)
    
    # Merge threshold label
    master = master.merge(labels[["case_id", "threshold_crossed"]], on="case_id", how="inner")
    
    # Filter to cases with coordinates
    master = master.dropna(subset=["longitude", "latitude"])
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(master, geometry=gpd.points_from_xy(master.longitude, master.latitude), crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857) # Web mercator for mapping
    
    date_col = next((col for col in ["filing_date", "as_of_date", "status_date", "date"] if col in gdf.columns), None)
    if date_col:
        gdf["year"] = pd.to_datetime(gdf[date_col]).dt.year
    else:
        # If all else fails, look for any date column
        date_cols = [c for c in gdf.columns if 'date' in c.lower()]
        if date_cols:
            gdf["year"] = pd.to_datetime(gdf[date_cols[0]]).dt.year
        else:
            raise ValueError("No date column found in dataset")
        
    years_to_plot = [2020, 2021, 2022, 2023]
    
    print("2. Downloading Travis County Census Tracts for Choropleth...")
    try:
        tracts = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2021/TRACT/tl_2021_48_tract.zip")
        travis_tracts = tracts[tracts['COUNTYFP'] == '453']
        travis_tracts = travis_tracts.to_crs(epsg=3857)
    except Exception as e:
        print(f"Warning: Could not download tracts, skipping background. Error: {e}")
        travis_tracts = None
        
    print("3. Generating Geographic EDA FacetGrid over Time...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, year in enumerate(years_to_plot):
        ax = axes[i]
        
        if travis_tracts is not None:
            # Just simple boundaries for EDA
            travis_tracts.boundary.plot(ax=ax, linewidth=0.5, color='gray', alpha=0.5)
            
        year_data = gdf[gdf["year"] == year]
        
        # Plot non-protested cases
        non_protested = year_data[year_data["threshold_crossed"] == 0]
        if len(non_protested) > 0:
            non_protested.plot(ax=ax, color='gray', alpha=0.3, markersize=15, label='Non-Protested', marker='o')
            
        # Plot protested cases
        protested = year_data[year_data["threshold_crossed"] == 1]
        if len(protested) > 0:
            protested.plot(ax=ax, color='red', alpha=0.8, markersize=60, label='Protested (>20%)', marker='X', edgecolor='black')
            
        ax.set_title(f"Zoning Cases in {year} (Total: {len(year_data)} | Protested: {len(protested)})", fontsize=18, pad=15)
        ax.axis('off')
        
        if i == 0:
            ax.legend(fontsize=14, loc='upper left')
            
    plt.tight_layout()
    out_path = ARTIFACTS_DIR / "geographic_eda_over_time.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Done! Saved Geographic EDA map to: {out_path}")

if __name__ == "__main__":
    main()
