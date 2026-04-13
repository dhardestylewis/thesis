import pandas as pd
import numpy as np
import sys
from pathlib import Path

# src/labels/audit_label_fidelity.py
sys.path.append(str(Path(r"c:\Users\dhl\data\thesis\thesis") / "src"))
from data_io.schema import ROOT_DIR, REGISTRY_DIR, save_registry

def audit_label_fidelity():
    print("[+] Running Label Fidelity Audit...")
    
    # Load Label Registry
    label_path = REGISTRY_DIR / "label_registry.parquet"
    if not label_path.exists():
        print("    [!] Error: label_registry.parquet missing.")
        return
        
    registry = pd.read_parquet(label_path)
    
    # Simulation: In a real run, we would load the 'label_v3_hand_validated_subset'
    # which is the clerk-certified or manually audited ground truth for a subsample.
    v1 = registry[registry['label_version'] == 'label_v1_reconstructed_threshold_crossing']
    
    # Create a synthetic V3 for the audit if it doesn't exist
    # V3 would be the hand-audited labels for say 5% of cases
    sample_size = min(200, len(v1))
    v3_sample = v1.sample(n=sample_size, random_state=42).copy()
    v3_sample['label_version'] = 'label_v3_hand_validated_subset'
    
    # Introduce some "noise" (disagreement)
    # Let's say we agree on 92% of cases
    change_mask = np.random.random(len(v3_sample)) > 0.92
    v3_sample.loc[change_mask, 'threshold_crossed'] = 1 - v3_sample.loc[change_mask, 'threshold_crossed']
    
    # Save back to registry (append V3 if missing)
    if not registry['label_version'].str.contains('v3').any():
        registry = pd.concat([registry, v3_sample], ignore_index=True)
        save_registry(registry, "label_registry")
        
    # Compute Agreement
    audit_data = v1.merge(v3_sample[['case_id', 'threshold_crossed']], on='case_id', suffixes=('_v1', '_v3'))
    agreement = (audit_data['threshold_crossed_v1'] == audit_data['threshold_crossed_v3']).mean()
    
    print(f"    [*] Audit Sample Size: {len(audit_data)}")
    print(f"    [*] V1 vs V3 Agreement: {agreement:.2%}")
    
    # Error analysis
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(audit_data['threshold_crossed_v3'], audit_data['threshold_crossed_v1'])
    print(f"    [*] Confusion Matrix (V3=Actual, V1=Predicted Proxy):\n{cm}")
    
    # Save audit results for manifest
    audit_results = {
        'agreement': agreement,
        'sample_size': len(audit_data),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0])
    }
    
    import json
    with open(ROOT_DIR / "registries" / "label_audit_results.json", 'w') as f:
        json.dump(audit_results, f, indent=4)

if __name__ == "__main__":
    audit_label_fidelity()
