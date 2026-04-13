import pandas as pd
import json
from pathlib import Path

# Paths
ROOT = Path(r"c:\Users\dhl\data\thesis\thesis")
PIPELINE_DATA = ROOT / "thesis_pipeline" / "data" / "final"

def build_manifest():
    print("[+] Building Final Metrics Manifest...")
    
    manifest = {}
    
    # 1. Load Prediction Metrics
    try:
        preds = pd.read_parquet(PIPELINE_DATA / "prediction_registry.parquet")
        # Get headline metric for CatBoost
        cb_preds = preds[preds['model_family'] == 'CatBoost']
        from sklearn.metrics import average_precision_score
        prauc = average_precision_score(cb_preds['y_true'], cb_preds['y_prob'])
        
        manifest['headline_pr_auc'] = {
            'value': f"{prauc:.3f}",
            'model': 'CatBoost',
            'split': 'TEMP_OOD_2023_MAIN',
            'task': 'Stage C Canonical'
        }
    except Exception as e:
        print(f"    [-] Error loading prediction metrics: {e}")

    # 2. Load Label Validity
    try:
        with open(PIPELINE_DATA / "label_validity_object.json", 'r') as f:
            label_audit = json.load(f)
        manifest['label_fidelity_score'] = {
            'value': f"{label_audit['v1_vs_v3_agreement']:.1%}",
            'audit_n': label_audit['v3_audit_n']
        }
    except Exception as e:
        print(f"    [-] Error loading label validity: {e}")

    # 3. Load Meta-Attribution
    try:
        with open(PIPELINE_DATA / "meta_attribution_object.json", 'r') as f:
            meta_attr = json.load(f)
        # Sort by mean importance
        meta_attr.sort(key=lambda x: x['mean'], reverse=True)
        top_cluster = meta_attr[0]['cluster']
        
        manifest['top_consensus_factor'] = {
            'value': top_cluster.replace('_', ' ').title(),
            'share': f"{meta_attr[0]['mean']:.1%}"
        }
    except Exception as e:
        print(f"    [-] Error loading meta-attribution: {e}")

    # Save Manifest
    with open(ROOT / "thesis_pipeline" / "reporting" / "final_metrics_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=4)
    
    print(f"    Manifest created with {len(manifest)} metric entries.")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    build_manifest()
