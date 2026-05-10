import os
import shutil

SCRATCH_DIR = r"C:\Users\dhl\data\Thesis\thesis\Scratch"
DATA_DIR = r"C:\Users\dhl\data\Thesis\thesis\Data"

# 1. Organize Scratch Directory
scratch_folders = {
    "Data_Acquisition": ["scrape_", "download_", "fetch_", "pull_"],
    "Data_Processing": ["build_", "extract_", "calculate_", "calc_", "engineer_", "parse_", "assign_"],
    "Modeling": ["train_", "benchmark_", "make_teacher"],
    "Visualization": ["plot_", "generate_"],
    "Diagnostics": ["check_", "investigate_", "audit_", "review_", "scan_", "test_", "print_"]
}

# DO NOT MOVE these active running scripts
EXCLUDE_FILES = ["train_gru_multi_horizon.py", "train_multi_horizon.py", "engineer_concentric_buffers.py", "parse_petitions.py", "organize_workspace.py"]

print("Organizing Scratch Directory...")
for folder in scratch_folders.keys():
    os.makedirs(os.path.join(SCRATCH_DIR, folder), exist_ok=True)

for file in os.listdir(SCRATCH_DIR):
    if file in EXCLUDE_FILES or not file.endswith(".py"):
        continue
    
    file_path = os.path.join(SCRATCH_DIR, file)
    if not os.path.isfile(file_path):
        continue
        
    moved = False
    for folder, prefixes in scratch_folders.items():
        if any(file.startswith(prefix) for prefix in prefixes):
            shutil.move(file_path, os.path.join(SCRATCH_DIR, folder, file))
            moved = True
            break
            
    if not moved:
        # Move anything else into a 'Misc_Scripts' folder
        misc_dir = os.path.join(SCRATCH_DIR, "Misc_Scripts")
        os.makedirs(misc_dir, exist_ok=True)
        shutil.move(file_path, os.path.join(misc_dir, file))

print("Scratch Organized!")

# 2. Organize Data Directory
print("\nOrganizing Data Directory...")
os.makedirs(os.path.join(DATA_DIR, "raw", "indices"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "interim"), exist_ok=True)

data_files = {
    "austin_council_meetings_index.csv": os.path.join(DATA_DIR, "raw", "indices"),
    "council_minutes_index.csv": os.path.join(DATA_DIR, "raw", "indices"),
    "planning_commission_index.csv": os.path.join(DATA_DIR, "raw", "indices"),
    "zoning_platting_commission_index.csv": os.path.join(DATA_DIR, "raw", "indices"),
    "commission_agendas_cases.csv": os.path.join(DATA_DIR, "interim"),
    "commission_reached_cases.csv": os.path.join(DATA_DIR, "interim"),
    "commission_transcripts.csv": os.path.join(DATA_DIR, "interim"),
    "council_agendas_cases.csv": os.path.join(DATA_DIR, "interim"),
    "council_agendas_missing_cases.csv": os.path.join(DATA_DIR, "interim"),
    "zoning_cases_with_council_votes.csv": os.path.join(DATA_DIR, "interim"),
    "model_ready_zoning_data.csv": os.path.join(DATA_DIR, "final") # Moving to final since it's "model ready"
}

for file, dest_folder in data_files.items():
    source = os.path.join(DATA_DIR, file)
    if os.path.exists(source):
        shutil.move(source, os.path.join(dest_folder, file))

print("Data Organized!")
