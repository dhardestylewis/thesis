import os
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT_DIR, "Data")
IN_FILE = os.path.join(DATA, "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
OUT_FILE = os.path.join(DATA, "Warehouse_As_Of", "H0_Filing_Master_OSMnx.csv")

def run_osmnx_routing():
    print("Loading V2 Master Warehouse...")
    df = pd.read_csv(IN_FILE, low_memory=False)
    
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("CRITICAL ERROR: Missing coordinate matrix for OSMnx routing.")
        return
        
    print("Downloading Austin, TX topological walking network graph...")
    # Request the full routable pedestrian infrastructure graph
    G = ox.graph_from_place('Austin, Texas, USA', network_type='walk')
    
    # Congress Ave (Texas State Capitol) acts as standard Urban Core epicenter
    core_lat, core_lon = 30.2747, -97.7404
    print("Snapping Urban Core epicenter to nearest physical grid node...")
    core_node = ox.distance.nearest_nodes(G, X=core_lon, Y=core_lat)
    
    network_distances = []
    
    print(f"Initializing spatial translation for {len(df)} historical case polygons...")
    valid_mask = df['latitude'].notna() & df['longitude'].notna()
    work_df = df[valid_mask].copy()
    
    X = work_df['longitude'].values
    Y = work_df['latitude'].values
    
    print("Vectorized snapping of active zoning cases to nearest OSM transit nodes...")
    case_nodes = ox.distance.nearest_nodes(G, X=X, Y=Y)
    
    print("Calculating absolute topological shortest network paths to Urban Core...")
    for idx, node in enumerate(case_nodes):
        try:
            dist = nx.shortest_path_length(G, node, core_node, weight='length')
        except nx.NetworkXNoPath:
            dist = None
        network_distances.append(dist)
        
        if (idx + 1) % 1000 == 0:
            print(f"Successfully routed {idx + 1} / {len(case_nodes)} topologies...")
            
    work_df['osmnx_core_walk_dist_m'] = network_distances
    
    # Bind back to the base dataframe
    df['osmnx_core_walk_dist_m'] = None
    df.loc[valid_mask, 'osmnx_core_walk_dist_m'] = work_df['osmnx_core_walk_dist_m']
    
    print(f"Exporting topological warehouse to {OUT_FILE}...")
    df.to_csv(OUT_FILE, index=False)
    print("OSMnx Routing Matrix Complete.")

if __name__ == "__main__":
    run_osmnx_routing()
