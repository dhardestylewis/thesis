import os
import sys
import time
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

# execute_thesis_pipeline.py
# Top-level orchestrator for the canonical Stage C pipeline.

ROOT = Path(r"c:\Users\dhl\data\thesis\thesis")
SCRIPTS_DIR = ROOT / "Scripts" / "pipeline"
TIMINGS_FILE = SCRIPTS_DIR / "empirical_timings.json"
DEFAULT_TIMING = 30.0

def load_timings():
    if TIMINGS_FILE.exists():
        try:
            with open(TIMINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_timing(script_name, duration_seconds):
    timings = load_timings()
    timings[script_name] = duration_seconds
    with open(TIMINGS_FILE, "w") as f:
        json.dump(timings, f, indent=2)

# Define the sequence
PIPELINE_STEPS = [
    ("00a_engineer_model_ready_zoning.py", "Engineering formal model_ready_zoning_data.csv baseline..."),
    # ("00b_extract_ears_data.py", "Extracting raw EARS AJR archives into panel-ready data..."),
    ("00_build_case_universe.py", "Building analytic universe spine..."),
    ("01_build_labels.py", "Generating the label-validity registry..."),
    ("01a1_parse_petition_pdf.py", "Extracting OCR text and tables from raw Protest PDFs..."),
    ("01a2_calculate_spatial_petitions.py", "Calculating the true GIS geographic protest footprint..."),
    ("02_build_features.py", "Engineering filing-date as-of features..."),
    ("02b_build_biweekly_panel.py", "Generating the formal biweekly causal inference panel..."),
    ("01b_build_biweekly_panel.py", "Generating the primary biweekly causal inference panel..."),
    ("01b2_patch_missing_petitions.py", "Patching missing NLP target petitions into the panel..."),
    ("01c_engineer_advanced_petitions.py", "Engineering advanced spatial petition features and injecting EDIMS dates..."),
    ("01d_merge_pdf_height_features.py", "Merging PDF extraction heights and outcomes..."),
    ("01e_build_advanced_features.py", "Engineering rolling window spatial gravity and velocities..."),
    ("03_build_splits.py", "Freezing evaluation split registry..."),
    ("04_train_stage_a.py", "Running Stage A selection-correction sidecar..."),
    ("05_train_stage_c.py", "Training the canonical Stage C model..."),
    ("06_calibrate_stage_c.py", "Calibrating Stage C predictions..."),
    ("07_evaluate_stage_c.py", "Evaluating Stage C ranking and calibration..."),
    ("08_run_meta_attribution.py", "Running meta-attribution interpretation sidecar..."),
    ("08b_run_ablation_suite.py", "Running semantic cluster ablations..."),
    ("08c_run_did_causal.py", "Running Stijn DiD causal estimators..."),
    ("08d_run_drift_and_archetypes.py", "Running temporal drift and 10-architecture meta-attribution..."),
    ("08e_run_multihorizon_oot.py", "Running Walk-Forward Multi-Horizon Out-Of-Time evaluation..."),
    ("08f_run_multihorizon_shap.py", "Generating Multi-Horizon SHAP spatial interaction heatmaps..."),
    ("09_run_audits.py", "Running label-fidelity and data-quality audits..."),
    ("09_hydrate_macro_features.py", "Hydrating macro features..."),
    ("10_link_lifecycle_stages.py", "Linking lifecycle stages..."),
    ("11_train_cradle_to_grave_model.py", "Training cradle to grave model..."),
    ("12_build_annual_lifecycle_panel.py", "Building annual lifecycle panel..."),
    ("12b_build_biweekly_lifecycle.py", "Building biweekly lifecycle panel..."),
    ("12c_merge_permit_features.py", "Merging permit features..."),
    ("13_export_tax_anomalies.py", "Exporting tax anomalies..."),
    ("10_export_manuscript_artifacts.py", "Generating metrics manifest and LaTeX macros..."),
    ("11_final_build_gate.py", "Running final submission build gate..."),
    ("12_generate_extracted_tables.py", "Exporting extracted table definitions directly to table path..."),
    ("13_render_prose_figures.py", "Regenerating all qualitative and quantitative prose figures..."),
    ("14_audit_prose_recency.py", "Running git-blame prose recency scan to flag stale paragraphs for manual review...")
]

def run_step(script_name, description, pbar, expected_duration):
    tqdm.write(f"\n>>> [STEP: {script_name}] {description}")
    script_path = SCRIPTS_DIR / script_name
    
    # Create the script if it doesn't exist (thin wrapper around src/)
    if not script_path.exists():
        create_script_wrapper(script_name)
        
    start = time.time()
    
    process = subprocess.Popen(["python", "-u", str(script_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    for line in iter(process.stdout.readline, ''):
        tqdm.write(line.strip('\n'))
        
    process.stdout.close()
    return_code = process.wait()
    
    end = time.time()
    actual_duration = end - start
    
    if return_code != 0:
        tqdm.write(f"    [!] Error in {script_name}. Aborting pipeline.")
        sys.exit(1)
    
    save_timing(script_name, actual_duration)
    tqdm.write(f"    [+] {script_name} completed in {actual_duration:.1f}s")
    pbar.update(expected_duration)

def create_script_wrapper(name):
    mappings = {
        "00_build_case_universe.py": "from src.labels.build_threshold_labels import build_case_universe; build_case_universe()",
        "01_build_labels.py": "from src.labels.build_threshold_labels import build_threshold_labels; build_threshold_labels()",
        "02_build_features.py": "from src.features.build_stage_c_features import build_stage_c_features; build_stage_c_features()",
        "01a2_calculate_spatial_petitions.py": "",
        "01b_build_biweekly_panel.py": "",
        "02b_build_biweekly_panel.py": "",
        "01c_engineer_advanced_petitions.py": "",
        "01d_merge_pdf_height_features.py": "",
        "01e_build_advanced_features.py": "",
        "03_build_splits.py": "from src.splits.build_split_registry import build_split_registry; build_split_registry()",
        "04_train_stage_a.py": "from src.features.build_stage_a_features import build_stage_a_features; from src.models.train_stage_a import train_stage_a_ipw; build_stage_a_features(); train_stage_a_ipw()",
        "05_train_stage_c.py": "from src.models.train_stage_c import train_stage_c; train_stage_c('CatBoost')",
        "06_calibrate_stage_c.py": "from src.models.calibrate_predictions import calibrate_predictions; calibrate_predictions()",
        "07_evaluate_stage_c.py": "from src.models.evaluate_predictions import evaluate_predictions; evaluate_predictions()",
        "08_run_meta_attribution.py": "from src.interpretation.run_meta_attribution import run_meta_attribution; run_meta_attribution()",
        "08b_run_ablation_suite.py": "from src.interpretation.ablation_suite import run_ablation_suite; run_ablation_suite()",
        "08c_run_did_causal.py": "import os; os.system('python Analysis/Scripts/Experiments/DiD/evaluate_stijn_did.py > results/stijn_did_results.txt')",
        "08d_run_drift_and_archetypes.py": "from src.interpretation.drift_and_archetypes import run_drift_and_archetypes; run_drift_and_archetypes()",
        "09_run_audits.py": "from src.labels.audit_label_fidelity import audit_label_fidelity; audit_label_fidelity()",
        "10_export_manuscript_artifacts.py": "from src.reporting.build_metrics_manifest import build_metrics_manifest; from src.reporting.export_metrics_tex import export_metrics_tex; build_metrics_manifest(); export_metrics_tex()",
        "11_final_build_gate.py": "from src.reporting.final_build_gate import run_final_build_gate; raise SystemExit(run_final_build_gate())",
        "12_generate_extracted_tables.py": "",
        "13_render_prose_figures.py": "import os, glob; [os.system(f'python \"{f}\"') for f in glob.glob('Analysis/Scripts/Visualization/Production_Figures/*.py') + glob.glob('Analysis/Scripts/Experiments/DiD/*.py') if 'electoral_placebo' in f or 'generate_' in f or 'plot_' in f]",
        "14_audit_prose_recency.py": "import os; os.system('python scripts/track_tex_recency.py')"
    }
    
    if name in mappings:
        with open(SCRIPTS_DIR / name, 'w') as f:
            f.write("import sys\nimport os\nfrom pathlib import Path\n")
            f.write("sys.path.append(r'c:\\Users\\dhl\\data\\thesis\\thesis')\n")
            f.write(mappings[name] + "\n")

def main():
    os.environ["USE_GPU"] = "1"
    print("="*60)
    print(" REFACTORED THESIS PIPELINE ORCHESTRATOR")
    print(f" Target Root: {ROOT}")
    print("="*60)
    
    total_start = time.time()
    timings = load_timings()
    total_expected = sum([timings.get(s[0], DEFAULT_TIMING) for s in PIPELINE_STEPS])
    
    with tqdm(total=total_expected, desc="Pipeline ETA Progress", unit="s", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for script, desc in PIPELINE_STEPS:
            expected = timings.get(script, DEFAULT_TIMING)
            run_step(script, desc, pbar, expected)
        
    total_time = (time.time() - total_start) / 60
    print("\n" + "="*60)
    print(f" PIPELINE COMPLETE ({total_time:.1f} minutes)")
    print(f" Results consolidated in registries/ folder.")
    print("="*60)

if __name__ == "__main__":
    main()
