import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

WORK_DIR = r"C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\Build"
FIGURES_DIR = r"C:\Users\dhl\data\thesis\thesis\Thesis_Draft\Draft_v1\Figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper")

def detect_temporal_regimes():
    print("Loading Council Voting Records for Unsupervised Regime Detection...")
    
    vr = pd.read_csv(os.path.join(WORK_DIR, "vote_record.csv"))
    cm = pd.read_csv(os.path.join(WORK_DIR, "case_master.csv"))
    
    # We must join by CASE_NUMBER to get the council_date
    df = vr.merge(cm[['CASE_NUMBER', 'council_date']], on='CASE_NUMBER', how='inner')
    df['date'] = pd.to_datetime(df['council_date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Create an active dissent matrix: proportion of NO/NAY votes per month per Council Member
    df['is_dissent'] = df['vote'].astype(str).str.upper().isin(['NAY', 'NO', 'AGAINST'])
    
    monthly_activity = df.groupby(['year_month', 'council_member'])['is_dissent'].mean().unstack(fill_value=0)
    
    if len(monthly_activity) < 3:
        print("Not enough longitudinal data to cluster.")
        return
        
    months = monthly_activity.index.to_timestamp()
    # StandardScaler requires a 2D array, which unstack provides, but ensure no implicit NaNs
    X = StandardScaler().fit_transform(monthly_activity.fillna(0).values)
    
    print("Applying K-Means to identify structural breaks in Voting Distributions...")
    # Assume 3 latent regimes over the decade constraint
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Plot the temporal regime map
    plt.figure(figsize=(9, 4))
    
    # Plot the clusters across the timeline
    scatter = plt.scatter(months, [1]*len(months), c=clusters, cmap='viridis', s=100, marker='|')
    
    # Format the timeline
    plt.yticks([])
    plt.xlabel('Timeline (Months)')
    plt.title('Unsupervised Matrix Detection of Council Policy Regimes (Latent Clusters)')
    
    legend1 = plt.legend(*scatter.legend_elements(), loc="upper left", title="Latent Regime")
    plt.gca().add_artist(legend1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig11_Unsupervised_Regimes.png"), dpi=300)
    plt.close()
    
    # Identify the largest transitions (structural breaks)
    print("\n--- DETECTED REGIME SHIFTS ---")
    prev_cluster = clusters[0]
    for i, cluster in enumerate(clusters):
        if cluster != prev_cluster:
            print(f"Structural Break Detected at: {months[i].strftime('%Y-%m')}")
            prev_cluster = cluster
            
    print("\nUnsupervised Regime Map Rendered natively to Fig11.")

if __name__ == "__main__":
    detect_temporal_regimes()
