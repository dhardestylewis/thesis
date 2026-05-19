"""
08_run_multihorizon_pipeline.py

Full end-to-end orchestrator:
  1. Biweekly multi-model OOT eval  → artifacts/multihorizon_multicutoff_all_models.csv
  2. Annualized multi-model OOT eval → artifacts/annualized_multihorizon_multicutoff_all_models.csv
  3. Figure suite (08g)              → Figures/exhibits/fig_multihorizon_*.pdf
  4. LaTeX table update              → Tables/chapter4_performance/tbl_ch4_09_multi_horizon_results.tex

Run:  python Scripts/pipeline/08_run_multihorizon_pipeline.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).parent

STEPS = [
    ("Biweekly multi-model OOT eval",    SCRIPTS / "08e_run_multihorizon_oot.py"),
    ("Annualized multi-model OOT eval",  SCRIPTS / "08e_run_annualized_oot.py"),
    ("Render multi-horizon figure suite", SCRIPTS / "08g_render_multihorizon_figures.py"),
    ("Update LaTeX results table",        SCRIPTS / "08h_update_multihorizon_latex_table.py"),
]

def run_step(label, script_path):
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"  Script: {script_path.name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAILED] {label} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[OK] {label} completed in {elapsed/60:.1f} min")

if __name__ == "__main__":
    print("\n🚀  Multi-Horizon Pipeline Orchestrator")
    print(f"    Root: {ROOT}\n")
    for label, script in STEPS:
        if script.exists():
            run_step(label, script)
        else:
            print(f"[SKIP] {script.name} not found — skipping")
    print("\n✅  Full pipeline complete.")
