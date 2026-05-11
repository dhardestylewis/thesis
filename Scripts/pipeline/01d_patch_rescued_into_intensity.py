"""
01d_patch_rescued_into_intensity.py
Merges rescued_petition_intensity.csv back into advanced_geometric_petition_intensity.csv
and re-runs 01c injection so the biweekly panel gets updated.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\dhl\data\Thesis\thesis")
sys.path.append(str(ROOT / "Scripts"))
from config.paths import PROTEST_PETITIONS_DIR, PANEL_DIR

# Load the base intensity file
base = pd.read_csv(PROTEST_PETITIONS_DIR / "advanced_geometric_petition_intensity.csv")
rescued = pd.read_csv(PROTEST_PETITIONS_DIR / "rescued_petition_intensity.csv")

print(f"Base intensity file: {len(base)} cases")
print(f"Rescued cases:       {len(rescued)}")
print(f"Cases with >0 in base: {(base['unofficial_protest_intensity'] > 0).sum()}")

# Patch: for each rescued case, update unofficial_protest_intensity and spatial_total_parcels
for _, row in rescued.iterrows():
    cn = row['case_number']
    mask = base['case_number'] == cn
    if mask.any():
        base.loc[mask, 'unofficial_protest_intensity'] = row['unofficial_protest_intensity']
        base.loc[mask, 'spatial_total_parcels'] = row['signers_resolved']
        base.loc[mask, 'signers_within_200ft'] = row['signers_within_200ft']
        base.loc[mask, 'signers_outside_200ft'] = row['signers_outside_200ft']
    else:
        # Case not in base at all — append a minimal row
        new_row = {col: np.nan for col in base.columns}
        new_row['case_number'] = cn
        new_row['unofficial_protest_intensity'] = row['unofficial_protest_intensity']
        new_row['spatial_total_parcels'] = row['signers_resolved']
        new_row['signers_within_200ft'] = row['signers_within_200ft']
        new_row['signers_outside_200ft'] = row['signers_outside_200ft']
        base = pd.concat([base, pd.DataFrame([new_row])], ignore_index=True)

print(f"Cases with >0 after patch: {(base['unofficial_protest_intensity'] > 0).sum()}")

# Also handle the 19 GeoJSON-matched cases for the main signer columns 
# (they already have geometry, we need to backfill from GeoJSON match results if available)
# The rescued.csv already handles them via the rescue pipeline above.

base.to_csv(PROTEST_PETITIONS_DIR / "advanced_geometric_petition_intensity.csv", index=False)
print(f"Saved patched intensity file.")

# Now re-run 01c to inject into biweekly_panel
print("\nRe-running 01c injection into biweekly_panel.csv...")
import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "engineer_advanced_petitions",
    os.path.join(ROOT, "Scripts", "pipeline", "01c_engineer_advanced_petitions.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.engineer_advanced_petitions()
