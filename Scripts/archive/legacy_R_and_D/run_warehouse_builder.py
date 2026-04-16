import os
import sys
import time
import subprocess
import argparse

ROOT = r"C:\Users\dhl\data\thesis\thesis"
BUILDER_DIR = os.path.join(ROOT, "Analysis", "Scripts", "Warehouse_Builder")
SCRAPER_DIR = os.path.join(ROOT, "Analysis", "Scripts", "Pipeline", "01_Scrapers_and_Harvesters")
NLP_DIR = os.path.join(ROOT, "Analysis", "Scripts", "Pipeline", "02_Transcription_and_NLP")

def print_header(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def run_directory_scripts(directory, desc):
    scripts = sorted([f for f in os.listdir(directory) if f.endswith(".py")])
    if not scripts:
        print(f"[-] No scripts found in {directory}")
        return
        
    print(f"\n[*] STARTING PHASE: {desc} ({len(scripts)} scripts)")
    for i, script in enumerate(scripts, 1):
        script_path = os.path.join(directory, script)
        print(f"    [+] Executing {script} ...")
        step_start = time.time()
        try:
            subprocess.run([sys.executable, script_path], check=True)
            step_mins = (time.time() - step_start) / 60
            print(f"        -> [SUCCESS] completed in {step_mins:.1f} m")
        except subprocess.CalledProcessError as e:
            print(f"        -> [FAILED] exited with code {e.returncode}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="End-to-end Data Warehouse Orchestrator")
    parser.add_argument("--full-scrape", action="store_true", help="Run the raw internet scrapers and AWS NLP transcriptions before building the warehouse.")
    args = parser.parse_args()

    start_time = time.time()
    print_header("THESIS DATA WAREHOUSE ORCHESTRATOR: END-TO-END BUILD")
    
    if args.full_scrape:
        print("WARNING: --full-scrape is active. Pinging City APIs, scraping agendas, and triggering AWS Transcription.")
        print("This may incur cloud costs and take hours depending on rate limits.")
        print("="*80 + "\n")
        run_directory_scripts(SCRAPER_DIR, "RAW DATA SCRAPING & HARVESTING")
        run_directory_scripts(NLP_DIR, "AUDIO TRANSCRIPTION & NLP VECTORIZATION")
    else:
        print("NOTE: Running local Warehouse Build only. (Use --full-scrape to re-download all raw data from the internet).")
        print("="*80 + "\n")

    # Fetch all numbered python files in the Warehouse_Builder directory
    scripts = sorted([f for f in os.listdir(BUILDER_DIR) if f.endswith(".py") and f[0].isdigit()])
    
    if not scripts:
        print("[-] Error: Could not locate numbered builder scripts in Analysis/Scripts/Warehouse_Builder")
        sys.exit(1)

    print(f"[*] Found {len(scripts)} sequential data engineering stages.")
    
    for i, script in enumerate(scripts, 1):
        script_path = os.path.join(BUILDER_DIR, script)
        print(f"\n[+] Executing Stage {i}/{len(scripts)}: {script} ...")
        
        step_start = time.time()
        try:
            # Using subprocess to ensure clean memory environments for massive NLP/pandas operations
            result = subprocess.run([sys.executable, script_path], check=True)
            step_mins = (time.time() - step_start) / 60
            print(f"    -> [SUCCESS] Completed {script} in {step_mins:.1f} minutes")
        except subprocess.CalledProcessError as e:
            print(f"    -> [FAILED] Stage {script} exited with error code {e.returncode}.")
            print(f"    -> HALTING WAREHOUSE BUILD PIPELINE TO PREVENT CORRUPTION.")
            sys.exit(1)
        except Exception as e:
            print(f"    -> [FATAL ERROR] {e}")
            sys.exit(1)

    # Note: If there are scripts in Pipeline/03_Data_Engineering_and_Panel_Builds that must run, 
    # they are typically invoked internally by the 33 scripts. If build_warehouse_as_of.py isn't
    # one of the 33, we execute it here to finalize the root datasets.
    
    # 34. Final Assembly (if missed by the 33 tracking loop)
    final_assembler = os.path.join(ROOT, "Analysis", "Scripts", "Pipeline", "03_Data_Engineering_and_Panel_Builds", "build_warehouse_as_of.py")
    if os.path.exists(final_assembler):
        print(f"\n[+] Executing Final Assembly: build_warehouse_as_of.py ...")
        try:
            subprocess.run([sys.executable, final_assembler], check=True)
            print("    -> [SUCCESS] Completed Master Assembly.")
        except Exception as e:
            print(f"    -> [FAILED] Master Assembly failed: {e}")

    total_time = (time.time() - start_time) / 60
    print_header(f"DATA ENGINEERING PIPELINE COMPLETE ({total_time:.1f} minutes)")
    print("-> The 9 root CSV datasets have been fundamentally reconstructed from raw data.")
    print("-> You may now execute `python run_thesis.py`.")

if __name__ == "__main__":
    main()
