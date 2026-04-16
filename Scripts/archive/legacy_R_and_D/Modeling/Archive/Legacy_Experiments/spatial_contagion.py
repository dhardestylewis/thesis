"""
spatial_contagion.py
======================
Priority 2: Spatial & Temporal Contagion Analysis (NIMBYism wildfire)
- Did a successful protest trigger nearby protests?
- Where was this tool used nearby recently?

Outputs: 
  - Analysis/Output/Descriptive/fig_spatial_knearest.png
  - Analysis/Output/Descriptive/spatial_contagion_stats.csv
"""

import pandas as pd
import numpy as np
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
try:
    # Attempt to locate the root Scripts directory
    _curr = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(_curr) != 'Scripts' and os.path.dirname(_curr) != _curr:
        _curr = os.path.dirname(_curr)
    if _curr not in sys.path:
        sys.path.insert(0, _curr)
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

from sklearn.neighbors import BallTree

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data")
OUT_DIR = os.path.join(ROOT, "Analysis", "Output", "Descriptive")

PET_GEO = os.path.join(OUT_DIR, "protest_timeline_geo.csv")
ZONING = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "enriched_zoning_data_full.csv")

def haversine_miles(d_rad):
    # Earth radius in miles
    return d_rad * 3958.8

def analyze_contagion():
    print("=== SPATIAL CONTAGION ANALYSIS ===")
    
    # 1. Load data
    pet = pd.read_csv(PET_GEO)
    pet = pet.dropna(subset=['latitude', 'longitude', 'year'])
    pet['year'] = pet['year'].astype(int)
    
    # Load background zoning cases to serve as "baseline" density
    zd = pd.read_csv(ZONING)
    zd = zd.dropna(subset=['latitude', 'longitude'])
    zd['app_year'] = pd.to_datetime(zd['application_start_date'], errors='coerce').dt.year
    zd = zd.dropna(subset=['app_year'])
    zd['app_year'] = zd['app_year'].astype(int)
    
    # Exclude protested cases from background
    protested_cases = set(pet['case_number'].unique())
    zd_not_protested = zd[~zd['case_number'].isin(protested_cases)]
    
    # 2. Convert to radians for BallTree
    pet['lat_rad'] = np.radians(pet['latitude'])
    pet['lon_rad'] = np.radians(pet['longitude'])
    
    zd_not_protested['lat_rad'] = np.radians(zd_not_protested['latitude'])
    zd_not_protested['lon_rad'] = np.radians(zd_not_protested['longitude'])
    
    results = []
    
    for _, row in pet.iterrows():
        yr = row['year']
        lat, lon = row['lat_rad'], row['lon_rad']
        
        # Protests in previous 1-3 years
        prior_pet = pet[(pet['year'] >= yr - 3) & (pet['year'] < yr)]
        
        # Background zoning cases in previous 1-3 years
        prior_zd = zd_not_protested[(zd_not_protested['app_year'] >= yr - 3) & (zd_not_protested['app_year'] < yr)]
        
        dist_pet = np.nan
        dist_zd = np.nan
        
        if len(prior_pet) > 0:
            tree_pet = BallTree(prior_pet[['lat_rad', 'lon_rad']].values, metric='haversine')
            dist, _ = tree_pet.query([[lat, lon]], k=1)
            dist_pet = haversine_miles(dist[0][0])
            
        if len(prior_zd) > 0:
            tree_zd = BallTree(prior_zd[['lat_rad', 'lon_rad']].values, metric='haversine')
            dist, _ = tree_zd.query([[lat, lon]], k=1)
            dist_zd = haversine_miles(dist[0][0])
            
        results.append({
            'case_number': row['case_number'],
            'year': yr,
            'dist_to_nearest_prior_protest_miles': dist_pet,
            'dist_to_nearest_prior_dev_miles': dist_zd
        })
        
    df_res = pd.DataFrame(results).dropna()
    
    # 3. Summary stats
    mean_dist_protest = df_res['dist_to_nearest_prior_protest_miles'].mean()
    mean_dist_dev = df_res['dist_to_nearest_prior_dev_miles'].mean()
    median_dist_protest = df_res['dist_to_nearest_prior_protest_miles'].median()
    median_dist_dev = df_res['dist_to_nearest_prior_dev_miles'].median()
    
    print(f"Distance to nearest prior PROTEST (within 3 yrs): Mean={mean_dist_protest:.2f} mi, Median={median_dist_protest:.2f} mi")
    print(f"Distance to nearest prior DEV (within 3 yrs):     Mean={mean_dist_dev:.2f} mi, Median={median_dist_dev:.2f} mi")
    
    # Calculate % of protests within X miles of a prior protest
    pct_within_05 = (df_res['dist_to_nearest_prior_protest_miles'] <= 0.5).mean() * 100
    pct_within_10 = (df_res['dist_to_nearest_prior_protest_miles'] <= 1.0).mean() * 100
    
    print(f"% of new protests within 0.5 miles of prior protest: {pct_within_05:.1f}%")
    print(f"% of new protests within 1.0 miles of prior protest: {pct_within_10:.1f}%")
    
    df_res.to_csv(os.path.join(OUT_DIR, "spatial_contagion_distances.csv"), index=False)
    
    # 4. Plot distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, 5, 25)
    ax.hist(df_res['dist_to_nearest_prior_protest_miles'], bins=bins, alpha=0.6, 
            label=f'Nearest Prior Protest (Median: {median_dist_protest:.2f} mi)', color='red', density=True)
    
    # We expect protests to be closer to development in general simply because development is clustered. 
    # But are they EVEN CLOSER to prior protests?
    ax.hist(df_res['dist_to_nearest_prior_dev_miles'], bins=bins, alpha=0.6, 
            label=f'Nearest Prior Zoning Case - Unprotested (Median: {median_dist_dev:.2f} mi)', color='blue', density=True)
            
    ax.set_title("NIMBY Contagion: Distance to Nearest Prior Event (Previous 3 Years)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Distance (Miles)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig9_spatial_contagion.png"), dpi=150)
    plt.close()
    print("Saved fig9_spatial_contagion.png")
    
    # Provide the temporal autocrrelation: does having a protest in year Y make district D more likely to have one in Y+1?
    print("\n--- Temporal/Spatial Correlation by Council District ---")
    pet_b = pet.copy()
    pet_b['protest_count'] = 1
    district_year = pet_b.groupby(['council_district', 'year'])['protest_count'].sum().reset_index()
    
    # create full panel
    districts = list(range(1, 11))
    years = list(range(2007, 2025))
    ix = pd.MultiIndex.from_product([districts, years], names=['council_district', 'year'])
    dy_full = pd.DataFrame(index=ix).reset_index()
    dy_full = dy_full.merge(district_year, on=['council_district', 'year'], how='left').fillna(0)
    
    dy_full = dy_full.sort_values(['council_district', 'year'])
    dy_full['prev_year_count'] = dy_full.groupby('council_district')['protest_count'].shift(1)
    
    corr = dy_full[['protest_count', 'prev_year_count']].corr().iloc[0, 1]
    print(f"Autocorrelation of protest volume in same district (Year over Year): r = {corr:.3f}")

if __name__ == "__main__":
    analyze_contagion()
