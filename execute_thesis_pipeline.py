import os
import sys
import time
from pathlib import Path

# execute_thesis_pipeline.py
# The New Top-Level Orchestrator for the Refactored Thesis

ROOT = Path(r"c:\Users\dhl\data\thesis\thesis")
SCRIPTS_DIR = ROOT / "scripts"

# Define the sequence
PIPELINE_STEPS = [
    ("00_build_case_universe.py", "Building analytic universe spine..."),
    ("01_build_labels.py", "Generating and auditing threshold labels..."),
    ("02_build_features.py", "Engineering filing-date as-of features..."),
    ("03_build_splits.py", "Freezing evaluation split registry..."),
    ("04_train_stage_a.py", "Training Stage A (IPW sidecar)..."),
    ("05_train_stage_c.py", "Executing Canonical Stage C training..."),
    # ("06_calibrate_stage_c.py", "Calibrating Stage C predictions..."),
    # ("07_evaluate_stage_c.py", "Running Stage C diagnostic suite..."),
    ("08_run_meta_attribution.py", "Running meta-attribution consensus audit..."),
    ("08b_run_ablation_suite.py", "Running cluster ablation sensitivity tests..."),
    ("09_run_audits.py", "Running structural and fairness audits..."),
    ("10_export_manuscript_artifacts.py", "Generating manifest and LaTeX macros...")
]

def run_step(script_name, description):
    print(f"\n>>> [STEP: {script_name}] {description}")
    script_path = SCRIPTS_DIR / script_name
    
    # Create the script if it doesn't exist (thin wrapper around src/)
    if not script_path.exists():
        create_script_wrapper(script_name)
        
    start = time.time()
    result = os.system(f"python {script_path}")
    end = time.time()
    
    if result != 0:
        print(f"    [!] Error in {script_name}. Aborting pipeline.")
        sys.exit(1)
    
    print(f"    [+] {script_name} completed in {end-start:.1f}s")

def create_script_wrapper(name):
    mappings = {
        "00_build_case_universe.py": "from src.labels.build_threshold_labels import build_case_universe; build_case_universe()",
        "01_build_labels.py": "from src.labels.build_threshold_labels import build_threshold_labels; build_threshold_labels()",
        "02_build_features.py": "from src.features.build_stage_c_features import build_stage_c_features; build_stage_c_features()",
        "03_build_splits.py": "from src.splits.build_split_registry import build_split_registry; build_split_registry()",
        "04_train_stage_a.py": "from src.features.build_stage_a_features import build_stage_a_features; from src.models.train_stage_a import train_stage_a_ipw; build_stage_a_features(); train_stage_a_ipw()",
        "05_train_stage_c.py": "from src.models.train_stage_c import train_stage_c; train_stage_c('CatBoost')",
        "08_run_meta_attribution.py": "from src.interpretation.run_meta_attribution import run_meta_attribution; run_meta_attribution()",
        "08b_run_ablation_suite.py": "from src.interpretation.ablation_suite import run_ablation_suite; run_ablation_suite()",
        "09_run_audits.py": "from src.labels.audit_label_fidelity import audit_label_fidelity; audit_label_fidelity()",
        "10_export_manuscript_artifacts.py": "from src.reporting.build_metrics_manifest import build_metrics_manifest; from src.reporting.export_metrics_tex import export_metrics_tex; build_metrics_manifest(); export_metrics_tex()",
    }
    
    if name in mappings:
        with open(SCRIPTS_DIR / name, 'w') as f:
            f.write("import sys\nimport os\nfrom pathlib import Path\n")
            f.write("sys.path.append(r'c:\\Users\\dhl\\data\\thesis\\thesis')\n")
            f.write(mappings[name] + "\n")

def main():
    print("="*60)
    print(" REFACTORED THESIS PIPELINE ORCHESTRATOR")
    print(f" Target Root: {ROOT}")
    print("="*60)
    
    total_start = time.time()
    
    for script, desc in PIPELINE_STEPS:
        run_step(script, desc)
        
    total_time = (time.time() - total_start) / 60
    print("\n" + "="*60)
    print(f" PIPELINE COMPLETE ({total_time:.1f} minutes)")
    print(f" Results consolidated in registries/ folder.")
    print("="*60)

if __name__ == "__main__":
    main()
