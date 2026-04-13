import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
import sys

# src/interpretation/run_meta_attribution.py
sys.path.append(str(Path(r"c:\Users\dhl\data\thesis\thesis") / "src"))
from src.data_io.schema import ROOT_DIR, REGISTRY_DIR, save_registry

def run_meta_attribution():
    print("[+] Running Formal Meta-Attribution Experiment...")
    
    # 1. Load Registry
    interp_path = REGISTRY_DIR / "interpretation_registry.parquet"
    if not interp_path.exists():
        print("    [!] Error: No interpretation data found. Run training first.")
        return
        
    df_attr = pd.read_parquet(interp_path)
    
    # 2. Load Semantic Mapping
    mapping_path = ROOT_DIR / "configs" / "features" / "semantic_clusters.yaml"
    with open(mapping_path, 'r') as f:
        config = yaml.safe_load(f)
        
    clusters = config['clusters']
    # Reverse mapping: feature -> cluster
    feat_to_cluster = {}
    for cluster_name, features in clusters.items():
        for feat in features:
            feat_to_cluster[feat] = cluster_name
            
    # 3. Map to Clusters
    # Note: Some features might be unmapped. We'll group them as "Other"
    df_attr['semantic_cluster'] = df_attr['feature_name'].map(feat_to_cluster).fillna('Other')
    
    # 4. Aggregate to Cluster Shares
    cluster_attr = df_attr.groupby(['model_family', 'seed', 'split_id', 'semantic_cluster'])['attribution_value'].sum().reset_index()
    
    # 5. Apply Consensus Rules
    # Definition: Top quartile attribution share, present in X% of model-period cells
    summary = cluster_attr.groupby('semantic_cluster')['attribution_value'].agg(['mean', 'median', 'std', 'count']).reset_index()
    
    # Consensus Feature Rule: median >= 0.10 and present in multiple models
    summary['is_consensus'] = (summary['median'] >= 0.10) & (summary['count'] >= 2)
    
    # 6. Save formal objects
    save_registry(cluster_attr, "interpretation_registry_clustered")
    
    output_obj = summary.to_dict(orient='records')
    with open(REGISTRY_DIR / "meta_attribution_object.json", 'w') as f:
        json.dump(output_obj, f, indent=4)
        
    print("\n>>> META-ATTRIBUTION CONSENSUS (REAL DATA) <<<")
    print(summary.sort_values('median', ascending=False))
    
    # Export identified feature sets
    consensus = summary[summary['is_consensus']]['semantic_cluster'].tolist()
    print(f"\n[*] Identified Consensus Features: {consensus}")

if __name__ == "__main__":
    run_meta_attribution()
