import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data_io.schema import REGISTRY_DIR, ROOT_DIR


def audit_label_fidelity() -> dict:
    """Compare the reconstructed label to the audited subsample when present."""

    label_path = REGISTRY_DIR / "label_registry.parquet"
    if not label_path.exists():
        raise FileNotFoundError(f"label_registry.parquet missing at {label_path}")

    registry = pd.read_parquet(label_path)
    v1 = registry[registry["label_version"] == "label_v1_reconstructed_threshold_crossing"].copy()
    v3 = registry[registry["label_version"] == "label_v3_hand_validated_subsample"].copy()

    if v1.empty or v3.empty:
        audit_results = {"agreement": None, "sample_size": 0, "fp": None, "fn": None, "audit_mode": "missing_v3"}
    else:
        audit_data = v1.merge(v3[["case_id", "threshold_crossed"]], on="case_id", suffixes=("_v1", "_v3"))
        agreement = float((audit_data["threshold_crossed_v1"] == audit_data["threshold_crossed_v3"]).mean()) if not audit_data.empty else None
        cm = confusion_matrix(audit_data["threshold_crossed_v3"], audit_data["threshold_crossed_v1"]) if not audit_data.empty else None
        audit_results = {
            "agreement": agreement,
            "sample_size": int(len(audit_data)),
            "fp": int(cm[0, 1]) if cm is not None and cm.size >= 4 else None,
            "fn": int(cm[1, 0]) if cm is not None and cm.size >= 4 else None,
            "audit_mode": "observed",
        }

    (ROOT_DIR / "registries" / "label_audit_results.json").write_text(json.dumps([audit_results], indent=2), encoding="utf-8")
    return audit_results

if __name__ == "__main__":
    audit_label_fidelity()
