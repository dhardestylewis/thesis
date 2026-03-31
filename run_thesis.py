"""
run_thesis.py: The Top-Level Thesis Orchestrator

This script executes the entire empirical pipeline for the Austin NIMBY thesis.
It runs the 4-Stage Predictive Pipeline, the 2-Track Causal Pipeline, and the
Visualizations/Tables pipeline using exclusively real data from the Data/Warehouse_As_Of/ directory. 
All outputs are written to the Analysis/Output/ directory and draft Figures folders.
"""
import sys
import os
import time
import datetime

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

ROOT = r"C:\Users\dhl\data\thesis\thesis"
log_dir = os.path.join(ROOT, "Analysis", "Scripts", "Modeling", "Utilities_and_Logs")
os.makedirs(log_dir, exist_ok=True)
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = DualLogger(os.path.join(log_dir, f"empirical_run_{run_id}.log"))

# Add Scripts to path so we can import modules
sys.path.append(os.path.join(ROOT, "Analysis", "Scripts", "Modeling"))
sys.path.append(os.path.join(ROOT, "Analysis", "Scripts", "Modeling", "Production_Models"))
sys.path.append(os.path.join(ROOT, "Analysis", "Scripts", "Visualization"))
sys.path.append(os.path.join(ROOT, "Analysis", "Scripts", "Visualization", "Production_Figures"))
sys.path.append(os.path.join(ROOT, "Analysis", "Scripts", "Experiments"))
sys.path.append(os.path.join(ROOT, "Analysis", "Scripts", "Experiments", "DiD"))
sys.path.append(os.path.join(ROOT, "Analysis", "Scripts", "Warehouse_Builder"))

# Import pipeline stages (Real Data Versions)
import StageA_development_hazard as stage_a
import StageB_6_Tier_Classifier as stage_b
import StageC_opposition_risk as stage_c
import StageD_institutional_outcome_real as stage_d
import StageF_generative_simulation as stage_f
import run_causal_track2_rd_real as track2
import run_causal_track3_did_real as track3
import run_multi_horizon as multi_horizon_table

# Import Visualizations & Tables (Real Data Versions)
import plot_F8_Calibration_real as f8
import plot_F12_PR_real as f12
import plot_F16_RD_real as f16
import plot_F17_DiD_real as f17
import plot_Track1_exhibits_real as t1_ex
import generate_summary_stats_real as stats_table
import plot_F19_F20_Qualitative as f19_f20
import plot_F22_HexMap as f22
import generate_stageA_exhibits as stage_a_exhibits
import generate_thesis_figures as fig1_and_more
import importlib
try:
    sweeps = importlib.import_module("18_real_model_sweeps")
except Exception as e:
    sweeps = None

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def main():
    start_time = time.time()
    
    # Enable bypass for heavy 5M row CatBoost matrix if --fast is passed
    fast_mode = '--fast' in sys.argv
    if fast_mode:
        print_header("THESIS ORCHESTRATOR: END-TO-END EXECUTION [FAST MODE ENABLED]")
    else:
        print_header("THESIS ORCHESTRATOR: END-TO-END EXECUTION")
    
    # ---------------------------------------------------------
    # PART 0: WAREHOUSE ENGINEERING
    # ---------------------------------------------------------
    print_header("PHASE 0: WAREHOUSE ENGINEERING")
    print("[+] Rebuilding Invariant Causal Prediction (ICP) Data Matrix...")
    try:
        if fast_mode:
            print("    [--fast bypass] Skipping programmatic regeneration of the submission_grade_icp_matrix.csv...")
        else:
            icp_gen = os.path.join(ROOT, "Analysis", "Scripts", "Pipeline", "03_Data_Engineering_and_Panel_Builds", "build_submission_demographics.py")
            os.system(f'python "{icp_gen}" > NUL')
            print("    Output: Data/Zoning_Cases/Processed_Data/CSV/submission_grade_icp_matrix.csv")
    except Exception as e:
        print(f"    [!] Error running ICP Builder: {e}")

    # ---------------------------------------------------------
    # PART 1: PREDICTIVE PIPELINE (Stages A - D)
    # ---------------------------------------------------------
    print_header("PHASE 1: PREDICTIVE PIPELINE")
    
    print("[+] Stage A: Development Hazard (Computed via StageA_development_hazard.py)")
    stage_a_path = os.path.join(ROOT, "Analysis", "Output", "Track0_Predictive", "stage_a_hazard_results.csv")
    try:
        if fast_mode and os.path.exists(stage_a_path):
            print(f"    [--fast bypass] Found cached Stage A hazard probabilities ({os.path.getsize(stage_a_path) / 1e6:.1f} MB). Skipping heavy CatBoost 5M-row training!")
        else:
            stage_a.run_stage_a()
            print("    Output: Analysis/Output/Track0_Predictive/stage_a_hazard_results.csv (1.08 GB)")
    except Exception as e:
        print(f"    [!] Error running Stage A: {e}")
    
    print("\n[+] Stage B: Project Scale & Typology")
    try:
        stage_b.run_stage_b()
    except Exception as e:
        print(f"    [!] Error running Stage B: {e}")
        
    print("\n[+] Stage C: Neighborhood Opposition Risk")
    try:
        stage_c.run_track1()
    except Exception as e:
        print(f"    [!] Error running Stage C: {e}")
        
    print("\n[+] Stage D: Institutional Outcome")
    try:
        stage_d.run_stage_d()
    except Exception as e:
        print(f"    [!] Error running Stage D: {e}")

    print("\n[+] Stage F: Generative Forward Simulation (Future Work Skeleton)")
    # try:
    #     stage_f.run_generative_simulation()
    # except Exception as e:
    #     print(f"    [!] Error running Stage F: {e}")

    # ---------------------------------------------------------
    # PART 2: CAUSAL PIPELINE (Tracks 2 & 3)
    # ---------------------------------------------------------
    print_header("PHASE 2: CAUSAL PIPELINE")
    
    try:
        track2.run_track2()
    except Exception as e:
        print(f"    [!] Error running Track 2: {e}")
        
    try:
        track3.run_track3()
    except Exception as e:
        print(f"    [!] Error running Track 3: {e}")

    # ---------------------------------------------------------
    # PART 3: TABLES & VISUALIZATIONS
    # ---------------------------------------------------------
    print_header("PHASE 3: TABLES & VISUALIZATIONS")
    
    try:
        stats_table.generate_summary_stats()
    except Exception as e:
        print(f"    [!] Error generating summary stats table: {e}")

    try:
        multi_horizon_table.main()
    except Exception as e:
        print(f"    [!] Error generating Table 8 (Multi-Horizon): {e}")

    try:
        f8.plot_f8()
    except Exception as e:
        print(f"    [!] Error generating Figure 8 (Calibration): {e}")

    try:
        f12.plot_f12()
    except Exception as e:
        print(f"    [!] Error generating Figure 12 (PR Curves): {e}")

    try:
        f16.plot_f16()
    except Exception as e:
        print(f"    [!] Error generating Figure 16 (RD): {e}")
        
    try:
        f17.plot_f17()
    except Exception as e:
        print(f"    [!] Error generating Figure 17 (DiD): {e}")

    try:
        t1_ex.plot_all_track1_exhibits()
    except Exception as e:
        print(f"    [!] Error generating Track 1 exhibits: {e}")

    try:
        f19_f20.generate_exhibits()
    except Exception as e:
        print(f"    [!] Error generating Figure 19/20 (Qualitative): {e}")

    try:
        f22.generate_exhibits()
    except Exception as e:
        print(f"    [!] Error generating Figure 22 (HexMap): {e}")

    try:
        stage_a_exhibits.generate_exhibits()
    except Exception as e:
        print(f"    [!] Error generating Stage A exhibits: {e}")

    try:
        fig1_and_more.main()
    except Exception as e:
        print(f"    [!] Error generating Figure 1 (Spatial Map) and others: {e}")

    try:
        if sweeps:
            sweeps.run_real_pipelines()
        else:
            print("    [!] Could not load 18_real_model_sweeps module.")
    except Exception as e:
        print(f"    [!] Error generating Real Model Sweeps (Fig 8, 9, 10): {e}")

    # ---------------------------------------------------------
    # PART 4: AST SEMANTIC NARRATIVE GENERATION
    # ---------------------------------------------------------
    print_header("PHASE 4: EVALUATING LLM NARRATIVE AST DEPENDENCIES")
    try:
        from Analysis.Scripts.Modeling.Production_Models import StageE_narrative_generation
        StageE_narrative_generation.run_stage_e()
    except Exception as e:
        print(f"    [!] Error during semantic AST generation: {e}")

    # ---------------------------------------------------------
    # PART 5: LATEX COMPILATION
    # ---------------------------------------------------------
    print_header("PHASE 4: COMPILING THESIS DOCUMENT")
    os.chdir(os.path.join(ROOT, "Thesis_Draft", "Draft_v1"))
    result = os.system("pdflatex -interaction=nonstopmode Austin_NIMBY_Thesis_Draft.tex")
    
    if result == 0:
        print("\n[+] PDF compiled successfully.")
    else:
        print("\n[-] PDF compilation returned warnings or errors.")

    end_time = time.time()
    minutes = (end_time - start_time) / 60
    print_header(f"ORCHESTRATOR COMPLETE ({minutes:.1f} minutes)")

if __name__ == "__main__":
    main()
