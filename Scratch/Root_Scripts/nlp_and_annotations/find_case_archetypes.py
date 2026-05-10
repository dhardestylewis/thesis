import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    print("Loading biweekly panel...")
    panel = pd.read_csv(r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv', low_memory=False)
    protested_cases = panel[panel['cumulative_petition_pct'] >= 20]['case_number'].unique()
    
    print(f"Found {len(protested_cases)} protested cases. Loading master data...")
    master = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\canonical\H0_Filing_Master_Enriched_v2_OmniLagged.csv', low_memory=False)
    master['case_id'] = master['case_number'].astype(str)
    
    protested_features = master[master['case_id'].isin(protested_cases)].copy()
    
    print("Calculating missingness...")
    protested_features['missing_ratio'] = protested_features.isnull().sum(axis=1) / len(protested_features.columns)
    
    # Take the 30 cases with the lowest missingness (best coverage)
    good_cases = protested_features.sort_values('missing_ratio').head(30).copy()
    
    # Define interesting covariate clusters (location, wealth, scale)
    # dist_to_cbd might be distance_to_cbd or similar. We'll find actual column names.
    potential_cols = ['longitude', 'latitude', 'improvement_sq_ft', 'appraised_val', 'median_hh_inc']
    cluster_cols = [c for c in protested_features.columns if any(x in c.lower() for x in ['long', 'lat', 'dist', 'inc', 'sq_ft', 'val']) and protested_features[c].dtype in ['float64', 'int64']]
    
    # Prune to just a few core ones to keep clustering sane
    cluster_cols = cluster_cols[:10]
    print(f"Clustering on: {cluster_cols}")
    
    X = good_cases[cluster_cols].fillna(good_cases[cluster_cols].median())
    X = X.dropna(axis=1, how='any')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    good_cases['cluster'] = kmeans.fit_predict(X_scaled)
    
    prototypes = []
    for i in range(3):
        centroid = kmeans.cluster_centers_[i]
        cluster_points = X_scaled[good_cases['cluster'] == i]
        cluster_cases = good_cases[good_cases['cluster'] == i]
        
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        best_idx = np.argmin(distances)
        best_case = cluster_cases.iloc[best_idx]
        prototypes.append(best_case)
        
    for i, p in enumerate(prototypes):
        print(f"\n--- Archetype {i+1} ---")
        print(f"Case ID: {p['case_id']}")
        print(f"Missing Ratio: {p['missing_ratio']:.2f}")
        for c in cluster_cols[:5]:
            print(f"{c}: {p[c]}")

if __name__ == "__main__":
    main()
