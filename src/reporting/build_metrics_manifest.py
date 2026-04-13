import pandas as pd
import json
from pathlib import Path
import sys
from sklearn.metrics import average_precision_score

# src/reporting/build_metrics_manifest.py
sys.path.append(str(Path(r"c:\Users\dhl\data\thesis\thesis") / "src"))
from src.data_io.schema import ROOT_DIR, REGISTRY_DIR

def build_metrics_manifest():
    print("[+] Building Metrics Manifest (Single Source of Truth)...")
    
    # Load Prediction Registry
    preds = pd.read_parquet(REGISTRY_DIR / "prediction_registry.parquet")
    
    # Extract specific headline result (e.g. CatBoost on TEMP_OOD_2023_MAIN)
    headline = preds[
        (preds['model_family'] == 'CatBoost') & 
        (preds['split_id'] == 'TEMP_OOD_2023_MAIN')
    ]
    
    if headline.empty:
        print("    [!] Error: No headline predictions found in registry.")
        return
        
    prauc = average_precision_score(headline['y_true'], headline['y_score_raw'])
    
    manifest = {
        "metricHeadlinePRAUC": {
            "value": f"{prauc:.3f}",
            "task_id": "STAGE_C_FILING_MAIN",
            "split_id": "TEMP_OOD_2023_MAIN",
            "model_family": "CatBoost",
            "calibration_method": "none",
            "source": str(REGISTRY_DIR / "prediction_registry.parquet")
        }
    }
    
    # Pull label fidelity from audit results
    audit_results_path = REGISTRY_DIR / "label_audit_results.json"
    if audit_results_path.exists():
        with open(audit_results_path, 'r') as f:
            audit_res = json.load(f)
        agreement_val = f"{audit_res['agreement']:.1%}"
    else:
        agreement_val = "N/A"

    manifest["metricLabelFidelity"] = {
        "value": agreement_val,
        "task_id": "LABEL_AUDIT_MAIN",
        "split_id": "FULL_UNIVERSE",
        "description": "Agreement between v1 reconstructed and v3 hand-validated subset"
    }
    
    # Save manifest
    manifest_path = REGISTRY_DIR / "metrics_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"    Manifest created with {len(manifest)} entries. Output: {manifest_path}")

if __name__ == "__main__":
    build_metrics_manifest()
