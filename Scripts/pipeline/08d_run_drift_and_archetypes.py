import sys
import os
from pathlib import Path

# Add project root to python path dynamically
ROOT = r'c:\Users\dhl\data\thesis\thesis'
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.interpretation.drift_and_archetypes import run_drift_and_archetypes

# 1. Generate Canonical 20% tables (main text output)
print("=== GENERATING CANONICAL DRIFT TABLES (20% Threshold) ===")
# run_drift_and_archetypes(threshold=0.20, is_appendix=False)

# 2. Generate Appendix Threshold tables
print("\n=== GENERATING APPENDIX SENSITIVITY GRIDS ===")
for t in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
    print(f"\n---> Running Threshold: >{int(t*100)}% <---")
    run_drift_and_archetypes(threshold=t, is_appendix=True)

print("\n[+] Appendix sensitivity complete.")
